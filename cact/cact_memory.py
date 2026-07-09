"""
C-ACT Memory: Contracted Adaptive Counterfactual Trust governance layer.

Wraps XENON DecomposedMemory with the C-ACT decision-time admission gate.

9 methods compared:
  NoKnowledge             — Pure LLM, no knowledge base (E3 baseline)
  XENON-Original          — Retrieve-and-reuse without gate
  BankCuration            — Relevance-based + merge/prune curation
  LifecycleSuccessGate    — Lifecycle + success rate gate
  FixedBayes              — Fixed-threshold Bayesian gate (no adaptive)
  ACT                     — Counterfactual uplift gate (no contract)
  C-ACT-Full              — Full C-ACT: contract + adaptive + interaction
  OracleGate              — Exhaustive ON/OFF oracle (upper bound, E4)
  ShuffledKnowledge       — Random knowledge mapping (sanity check, E4)

6 log types:
  episode/      — Episode-level outcomes
  contracts/    — Knowledge contract lifecycle
  reuse/        — Reuse decision with full audit trail
  base/         — Active base logging with propensity
  interaction/  — Interaction conflict / chain failures
  lifecycle/    — Lifecycle state transitions

Knowledge lifecycle: Candidate → Quarantined → Probation → Certified → Deprecated → Disabled
No RL rollback — only knowledge-level governance.
"""

import logging, numpy as np, json, os, copy, time, random
from typing import Dict, List, Optional, Tuple

from .trust_store import TrustStore, CANDIDATE, QUARANTINED, PROBATION, CERTIFIED, DEPRECATED, DISABLED
from .trust_gate import TrustGate
from .context_bucket import ContextBucket
from .contract import ContractExtractor, ContractChecker
from .decision_controller import DecisionController
from .interaction_gate import InteractionGate
from .active_logging import ActiveBaseLogger
from .thompson_probe import SafeThompsonProber
from .bank_sanitizer import BankSanitizer
from .attribution import OutcomeAttributor, apply_attribution_to_lifecycle


class CactMemory:
    """C-ACT decision-time admission layer wrapping XENON DecomposedMemory."""

    def __init__(self, xenon_memory, method="C-ACT-Full",
                 store_path: str = None, frozen: bool = False,
                 active_calib_rate: float = 0.0, log_dir: str = None):
        self._mem = xenon_memory

        # Component flags (for ablation)
        self._use_contract = True
        self._use_adaptive_tau = True
        self._use_active_calib = True
        self._use_interaction = True
        self._use_level_prior = True
        self._use_lifecycle = True
        self._use_thompson = True
        self._use_sanitizer = True

        # Parse method name for ablation variants
        if "NoContract" in method or "no_Contract" in method:
            self._use_contract = False
        if "NoAdaptiveTau" in method or "no_AdaptiveTau" in method:
            self._use_adaptive_tau = False
        if "NoActiveCalib" in method or "no_ActiveCalib" in method:
            self._use_active_calib = False
        if "NoInteraction" in method or "no_Interaction" in method:
            self._use_interaction = False
        if "NoLevelPrior" in method or "no_LevelPrior" in method:
            self._use_level_prior = False
        if "NoLifecycle" in method or "no_Lifecycle" in method:
            self._use_lifecycle = False
        if "NoThompson" in method or "no_Thompson" in method:
            self._use_thompson = False
        if "NoSanitizer" in method or "no_Sanitizer" in method:
            self._use_sanitizer = False

        # Core C-ACT components
        self._store = TrustStore(store_path or "cact_ckpt/trust_store")
        self._gate = TrustGate()
        self._bucket = ContextBucket()
        self._extractor = ContractExtractor()
        self._checker = ContractChecker()
        self._ig = InteractionGate()
        self._al = ActiveBaseLogger(
            log_path=os.path.join(log_dir, "base", "propensity.jsonl") if log_dir else None)
        self._tp = SafeThompsonProber()
        self._sanitizer = BankSanitizer()
        self._attributor = OutcomeAttributor()
        self._controller = DecisionController(
            self._store, self._gate, self._ig, self._al, self._tp, self._checker)

        self.method = method
        self.frozen = frozen
        self.active_calib_rate = active_calib_rate
        self.log_dir = log_dir

        # Log buffers
        self.episode_logs: List[Dict] = []
        self.contract_logs: List[Dict] = []
        self.reuse_logs: List[Dict] = []
        self.base_logs: List[Dict] = []
        self.interaction_logs: List[Dict] = []
        self.lifecycle_logs: List[Dict] = []

        # State tracking
        self._prev_kid = None
        self._last_was_supervised = False
        self._registered_cnt = 0    # counter for register_knowledge
        self._generated_cnt = 0     # counter for knowledge generated in save_success_failure
        self._current_difficulty = "medium"
        self._current_group = "crafting"
        self._current_observation: Dict = {}
        self._current_info: Dict = {}
        self._pairwise_harm: Dict[Tuple[str, str], Tuple[int, int]] = {}
        self._drift_counter = 0
        self._logs_dumped = False   # track whether dump_logs has been called
        self._last_sync_hash = 0    # cache key for _sync_ablation_flags
        self._waypoint_to_kid: Dict[str, str] = {}  # waypoint → registered knowledge_id

    # ── Pass-through to XENON DecomposedMemory ──
    @property
    def succeeded_waypoints(self): return self._mem.succeeded_waypoints
    def retrieve_similar_succeeded_waypoints(self, w, k=3): return self._mem.retrieve_similar_succeeded_waypoints(w, k)
    def retrieve_failed_subgoals(self, w): return self._mem.retrieve_failed_subgoals(w)
    def retrieve_total_failed_counts(self, w): return self._mem.retrieve_total_failed_counts(w)
    def save_plan(self, *a, **kw): return self._mem.save_plan(*a, **kw)
    def reset_success_failure_history(self, i): return self._mem.reset_success_failure_history(i)
    def set_history_index(self, i): return self._mem.set_history_index(i)
    def add_succeeded_waypoint(self, wp, action): return self._mem.add_succeeded_waypoint(wp, action)
    def add_failed_waypoint(self, wp, action): return self._mem.add_failed_waypoint(wp, action)
    def save_reflection(self, wp, reflect, is_success): return self._mem.save_reflection(wp, reflect, is_success)
    def save_replan(self, wp, replan): return self._mem.save_replan(wp, replan)
    def save_decomposed_plan(self, wp, plan, is_success): return self._mem.save_decomposed_plan(wp, plan, is_success)
    def set_current_environment(self, env_name): pass
    @property
    def current_environment(self): return None
    @current_environment.setter
    def current_environment(self, v): pass

    def _sync_ablation_flags(self):
        """Pass ablation flags to all affected components. Cached per state hash."""
        # Build a hash of current ablation + threshold state to skip redundant syncs
        tg = self._gate
        state_key = hash((
            self._use_contract, self._use_active_calib, self._use_interaction,
            self._use_thompson, self._use_lifecycle, self._use_level_prior,
            self._use_adaptive_tau,
            tuple(sorted(tg.tau.items())), tuple(sorted(tg.harm.items())),
        ))
        if state_key == self._last_sync_hash:
            return
        self._last_sync_hash = state_key

        c = self._controller
        c.abl_contract = self._use_contract
        c.abl_active_calib = self._use_active_calib
        c.abl_interaction = self._use_interaction
        c.abl_thompson = self._use_thompson
        self._store.abl_lifecycle = self._use_lifecycle
        self._store.abl_level_prior = self._use_level_prior
        self._gate.abl_adaptive = self._use_adaptive_tau

        # Sync per-group TrustGate thresholds to TrustStore for lifecycle transitions
        per_group = {}
        for grp in ["crafting", "mining", "exploration", "tech_tree",
                     "failure_recovery", "interaction_stress"]:
            per_group[grp] = {
                "tau": tg.tau.get(grp, 0.88),
                "delta": tg.delta.get(grp, 0.05),
                "harm": tg.harm.get(grp, 0.10),
            }
        self._store.sync_calibration(per_group)

    def set_observation(self, observation: Dict = None, info: Dict = None):
        """Store current game state for contract precondition checking."""
        if observation is not None:
            self._current_observation = observation
        if info is not None:
            self._current_info = info

    def update_inventory(self, item: str, count: int = 1):
        """Incrementally update inventory after craft/smelt where env.step() is not called.

        The craft/smelt helper uses direct inventory manipulation, bypassing
        env.step(), so no new observation is generated. This method manually
        patches the cached inventory to keep contract precondition checking
        accurate.
        """
        obs = self._current_observation
        if not obs:
            return
        inv = obs.get("inventory", {})
        if not isinstance(inv, dict):
            return
        current = int(inv.get(item, 0))
        inv[item] = current + count

    def set_task_info(self, difficulty: str = "medium", group: str = "crafting"):
        """Set current task metadata for episode logging."""
        self._current_difficulty = difficulty
        self._current_group = group

    def needs_supervision_check(self) -> bool:
        """Check if the last reuse decision was in supervised (Probation) mode.

        When True, the caller should run an immediate reflection check after
        each execution step. If the check fails, fall back to base policy.
        """
        return self._last_was_supervised

    # ── Log dumping ──
    def dump_logs(self):
        if not self.log_dir: return
        os.makedirs(self.log_dir, exist_ok=True)
        log_files = {
            "episode/episode.jsonl": self.episode_logs,
            "contracts/contracts.jsonl": self.contract_logs,
            "reuse/reuse_decision.jsonl": self.reuse_logs,
            "base/base_logging.jsonl": self.base_logs,
            "interaction/interaction.jsonl": self.interaction_logs,
            "lifecycle/lifecycle.jsonl": self.lifecycle_logs,
        }
        # Truncate on first call to avoid mixing with previous run data
        mode = "w" if not self._logs_dumped else "a"
        self._logs_dumped = True
        for fname, data in log_files.items():
            if data:
                path = os.path.join(self.log_dir, fname)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, mode) as f:
                    for e in data:
                        f.write(json.dumps(e, ensure_ascii=False) + "\n")
        # Clear buffers
        self.episode_logs.clear()
        self.contract_logs.clear()
        self.reuse_logs.clear()
        self.base_logs.clear()
        self.interaction_logs.clear()
        self.lifecycle_logs.clear()

    # ── Episode stats update ──
    def update_last_episode(self, total_steps=0, llm_calls=0, wall_time_sec=0.0,
                            input_tokens=0, output_tokens=0):
        if self.episode_logs:
            e = self.episode_logs[-1]
            e["total_steps"] = total_steps or e.get("total_steps", 0)
            e["llm_calls"] = llm_calls or e.get("llm_calls", 0)
            e["wall_time_sec"] = wall_time_sec or e.get("wall_time_sec", 0.0)
            e["tokens"] = e.get("tokens", 0) + input_tokens + output_tokens

    # ── Knowledge contract registration ──
    def register_knowledge(self, knowledge_dict: Dict) -> Optional[str]:
        """Extract contract from raw knowledge, sanitize, and register in TrustStore.

        Returns None if the knowledge is deduplicated or quarantined without contract.
        """
        # Pass through Bank Sanitizer before contract extraction
        if self._use_sanitizer:
            clean_list, actions = self._sanitizer.sanitize([knowledge_dict])
            if not clean_list:
                return None  # Deduplicated or discarded
            knowledge_dict = clean_list[0]

        contract = self._extractor.extract(knowledge_dict)
        kid = contract.knowledge_id
        self._store.register_contract(kid, contract.to_dict())
        self._registered_cnt += 1

        # Record waypoint→kid mapping for later lookup in save_success_failure
        wp = knowledge_dict.get("subgoal", knowledge_dict.get("waypoint", ""))
        if not wp:
            # Infer from gene: first substantive words typically contain the waypoint
            gene = contract.gene or ""
            wp = gene.split()[0] if gene else ""
        if wp:
            self._waypoint_to_kid[wp] = kid

        # Log contract
        self.contract_logs.append({
            "knowledge_id": kid,
            "type": contract.type,
            "level": contract.level,
            "gene": contract.gene,
            "claimed_context": contract.claimed_context,
            "expected_uplift": contract.expected_uplift,
            "risk_bound": contract.risk_bound,
            "status": knowledge_dict.get("status", CANDIDATE),
            "source_episode": contract.source_episode,
        })
        return kid

    # ── save_success_failure ──
    def save_success_failure(self, waypoint, language_action_str, is_success):
        """Record outcome after a knowledge use/failure with attribution."""
        self._mem.save_success_failure(waypoint, language_action_str, is_success)
        sv = 1.0 if is_success else 0.0
        # Look up the registered knowledge_id for this waypoint
        kid = self._waypoint_to_kid.get(waypoint, f"skill:{waypoint}")
        ctx = self._bucket.encode(knowledge_type="skill",
                                  subgoal_type=self._infer_subgoal_type(waypoint),
                                  task_tier=self._infer_tier(waypoint))
        task_grp = self._infer_group(waypoint)

        # Use last decision result
        last = getattr(self, '_last_decision_result', None)
        was_used = last.decision in ("reuse", "probe") if last else True

        ess = self._store.ess(kid, ctx)
        pi = self._store.uplift_probability(kid, ctx)
        hu = self._store.harm_upper_bound(kid, ctx)
        ul = self._store.uplift_lcb(kid, ctx)
        lc = self._store.get_lifecycle_state(kid)

        # Contract / interaction status
        contract = self._store.get_contract(kid)
        # Contract violation: prefer explicit postcondition check from decision result,
        # fall back to harm_ucb proxy (high harm rate suggests contract not protective)
        contract_violated = (
            last.contract_violation_after if last and last.contract_violation_after
            else ((not is_success and hu >= 0.10) if contract else False)
        )
        interaction_conflict = not last.interaction_safe if last else False
        csr_before = last.contract_satisfied_before if last else True
        is_harmful = 1 if (not is_success and hu >= 0.10) else 0

        # ── Attribution-aware lifecycle update ──
        if not self.frozen:
            attribution = self._attributor.attribute(
                success=is_success,
                is_harmful=bool(is_harmful),
                contract_violated=contract_violated,
                interaction_conflict=interaction_conflict,
                used_knowledge=was_used,
                progress_delta=0.0,  # Updated by caller if available
            )
            # Apply attribution-aware update
            attr_result = apply_attribution_to_lifecycle(
                kid, attribution, sv, float(is_harmful),
                self._store, ctx, self._attributor)
        else:
            attribution = None
            attr_result = {"action": "frozen", "attribution": "none"}

        # Reuse decision log
        self.reuse_logs.append({
            "waypoint": waypoint, "kid": kid, "ctx": ctx,
            "method": self.method, "frozen": self.frozen,
            "success": is_success, "task_group": task_grp,
            "pi_uplift": round(pi, 4), "uplift_lcb": round(ul, 4),
            "harm_ucb": round(hu, 4), "ess": round(ess, 1),
            "lifecycle": lc, "decision": last.decision if last else "reuse",
            "contract_satisfied_before": csr_before,
            "contract_violation_after": contract_violated,
            "is_harmful": is_harmful,
            "attribution": attribution.value if attribution else "frozen",
            "attr_action": attr_result.get("action", "none"),
            "outcome_success": int(is_success),
        })

        # Episode log
        self.episode_logs.append({
            "waypoint": waypoint, "method": self.method, "frozen": self.frozen,
            "task_group": task_grp, "difficulty": self._current_difficulty,
            "success": int(is_success),
            "total_steps": 0, "llm_calls": 0, "tokens": 0,
            "unrecoverable_failure": 0 if is_success else 1,
        })

        # Knowledge generation
        if is_success and ess >= 2:
            self.episode_logs[-1]["knowledge_generated"] = f"gen_{waypoint}_{self._generated_cnt:02d}"
            self._generated_cnt += 1

        # Drift tracking + periodic decay
        pred_err = abs(sv - self._store.mean(kid, ctx, "use"))
        self._store.adapt_decay(pred_err)
        self._drift_counter += 1
        if self._drift_counter % 20 == 0:
            self._store.decay_all()

        # Reset supervised flag — no longer in supervised context after recording
        self._last_was_supervised = False

    # ── is_succeeded_waypoint (MAIN GATE) ──
    def is_succeeded_waypoint(self, waypoint):
        """C-ACT admission gate for knowledge reuse."""
        ctx = self._bucket.encode(knowledge_type="skill",
                                  subgoal_type=self._infer_subgoal_type(waypoint),
                                  task_tier=self._infer_tier(waypoint))
        kid = self._waypoint_to_kid.get(waypoint, f"skill:{waypoint}")
        task_grp = self._infer_group(waypoint)
        lc = self._store.get_lifecycle_state(kid)
        ess = self._store.ess(kid, ctx)

        # Build candidate list
        contract = self._store.get_contract(kid) or {}
        candidates = [{
            "knowledge_id": kid,
            "type": contract.get("type", "skill"),
            "level": contract.get("level", "atomic_correction"),
            "gene": contract.get("gene", waypoint),
            "full_text": contract.get("full_text", ""),
            "claimed_context": contract.get("scope", contract.get("claimed_context", {})),
            "preconditions": contract.get("preconditions", []),
            "postconditions": contract.get("postconditions", []),
            "non_applicable_contexts": contract.get(
                "hard_non_applicable_contexts", contract.get("non_applicable_contexts", [])),
        }]

        # Build state from observation for contract precondition checking.
        # Safety flags are computed from available mineRL data (inventory +
        # equipment) because mineRL does not expose health/near_lava/combat
        # as direct observation fields.
        obs = self._current_observation
        info = self._current_info
        state = {
            "waypoint": waypoint,
            "task_group": task_grp,
        }
        if obs:
            inv = obs.get("inventory", {})
            # Build flat inventory: item_name -> count for precondition matching
            inv_flat = {}
            if isinstance(inv, dict):
                for k, v in inv.items():
                    try:
                        inv_flat[k] = v.item() if hasattr(v, 'item') else int(v)
                    except (TypeError, ValueError):
                        pass
            state.update(inv_flat)

            # Equipment: what tool/weapon is currently equipped
            eq = obs.get("equipped_items", {})
            if isinstance(eq, dict):
                for k, v in eq.items():
                    if isinstance(v, str) and v != "none":
                        state[k] = v

            # Compute safety flags from available data
            inv_items = set(inv_flat.keys())
            state["has_iron_pickaxe"] = "iron_pickaxe" in inv_items
            state["has_diamond_pickaxe"] = "diamond_pickaxe" in inv_items
            state["has_stone_pickaxe"] = "stone_pickaxe" in inv_items
            state["has_furnace"] = "furnace" in inv_items
            state["has_crafting_table"] = "crafting_table" in inv_items
            state["has_fuel"] = bool({"coal", "charcoal"} & inv_items)
            state["has_bucket"] = "bucket" in inv_items or "lava_bucket" in inv_items
            state["near_lava"] = "lava_bucket" in inv_items
            state["in_combat"] = bool({"iron_sword", "diamond_sword", "stone_sword",
                "wooden_sword", "shield", "bow"} & inv_items) and eq.get("mainhand", {}).get(
                "type", "") in ("sword", "axe", "bow")
            state["low_health"] = False  # Not directly observable from mineRL
            state["near_cliff"] = False  # Not directly observable
            state["irreversible_resource_constraint"] = bool(
                {"diamond", "diamond_pickaxe", "obsidian", "diamond_sword"} & inv_items)
            state["resource_critical"] = state["irreversible_resource_constraint"]

            # Scalar observation fields
            for k in ("location_stats", "life_stats"):
                if k in obs and isinstance(obs[k], dict):
                    state[k] = obs[k]
        task = {"task_id": waypoint, "group": task_grp}
        context = {"bucket": ctx, "subgoal_type": self._infer_subgoal_type(waypoint),
                   "task_group": task_grp, "task_tier": self._infer_tier(waypoint),
                   "failure_type": "none", "risk_level": self._infer_risk(waypoint)}

        if self.method.startswith("Online-"):
            mode = "online"
        elif self.active_calib_rate > 0:
            mode = "calibration"
        elif self.frozen:
            mode = "evaluation"
        else:
            mode = "accumulation"

        # C-ACT decision
        if self.method in ("C-ACT-Full", "Online-C-ACT") or "C-ACT-" in self.method:
            self._sync_ablation_flags()
            result = self._controller.decide(
                candidates, state, task, context, mode)
        else:
            result = self._legacy_decide(
                kid, ctx, candidates, state, task, context)

        self._last_decision_result = result
        trusted = result.decision in ("reuse", "probe")
        self._last_was_supervised = getattr(result, "supervised", False)

        if trusted:
            is_ok, sg = self._mem.is_succeeded_waypoint(waypoint)
            final = (True, sg) if is_ok else (False, None)
        else:
            final = (False, None)

        # Log interaction conflicts
        if not result.interaction_safe:
            self.interaction_logs.append({
                "episode_id": f"ep_{waypoint}",
                "used_knowledge_chain": [result.chosen_knowledge_id] if result.chosen_knowledge_id else [],
                "interaction_status": "conflict_detected",
                "interaction_state": result.interaction_state,
            })

        # Propensity logging
        if result.propensity_base > 0:
            self.base_logs.append({
                "decision_id": f"{waypoint}_step",
                "candidate": result.chosen_knowledge_id,
                "assigned_action": result.decision,
                "propensity_reuse": result.propensity_reuse,
                "propensity_base": result.propensity_base,
                "context_bucket": ctx,
            })

        self._prev_kid = kid if trusted else None
        return final

    # ── Legacy decision methods (for baselines) ──
    def _legacy_decide(self, kid, ctx, candidates, state, task, context):
        """Decision logic for non-C-ACT-Full methods."""
        from .decision_controller import DecisionResult
        result = DecisionResult()
        pi = self._store.uplift_probability(kid, ctx)
        up = self._store.uplift_lcb(kid, ctx)
        hu = self._store.harm_upper_bound(kid, ctx)
        ess = self._store.ess(kid, ctx)
        lc = self._store.get_lifecycle_state(kid)
        grp = task.get("group", "crafting")

        if self.method == "NoKnowledge":
            result.decision = "reuse" if np.random.random() < 0.5 else "fallback"
        elif self.method == "XENON-Original" or self.method == "Online-NoGate":
            result.decision = "reuse"
        elif self.method == "BankCuration" or self.method == "Online-BankCuration":
            result.decision = "reuse" if self._store.mean(kid, ctx, "use") >= 0.3 else "fallback"
        elif self.method == "LifecycleSuccessGate" or self.method == "Online-SuccessLifecycle":
            if lc in (QUARANTINED, DEPRECATED, DISABLED):
                result.decision = "fallback"
            else:
                result.decision = "reuse" if self._store.mean(kid, ctx, "use") >= 0.5 else "fallback"
        elif self.method == "FixedBayes" or self.method == "Online-FixedBayes":
            if self._store.total_count(kid, ctx, "use") < 1:
                result.decision = "reuse"
            else:
                result.decision = "reuse" if (pi >= 0.90 and up >= 0.05 and hu <= 0.10) else "fallback"
        elif self.method == "ACT" or self.method == "Online-ACT":
            if self._store.total_count(kid, ctx, "use") < 1:
                result.decision = "reuse"
            else:
                result.decision = "reuse" if (pi >= 0.90 and hu <= 0.10) else "fallback"
        elif self.method == "C-ACT-Full" or self.method == "Online-C-ACT":
            result.decision = "reuse"  # Full C-ACT handled by decision_controller
        elif self.method == "OracleGate":
            result.decision = "reuse"
        elif self.method == "ShuffledKnowledge":
            result.decision = "reuse" if np.random.random() < 0.5 else "fallback"
        else:
            result.decision = "reuse"

        result.pi_uplift = pi
        result.harm_ucb = hu
        result.uplift_lcb = up
        result.lifecycle_state = lc
        result.chosen_knowledge_id = kid
        return result

    # ── Helpers (inference from waypoint text) ──
    def _infer_subgoal_type(self, item):
        """Infer subgoal type from waypoint text."""
        il = str(item).lower()
        if any(t in il for t in ["craft", "make", "build", "smelt"]): return "craft"
        if any(t in il for t in ["mine", "collect", "dig", "chop"]): return "mine"
        if any(t in il for t in ["find", "explore", "locate", "village"]): return "explore"
        if any(t in il for t in ["kill", "combat", "hunt", "shoot"]): return "combat"
        return "craft"

    def _infer_tier(self, item):
        il = str(item).lower()
        if any(t in il for t in ["diamond", "netherite", "ender"]): return "diamond"
        if any(t in il for t in ["iron", "obsidian", "portal"]): return "iron"
        if any(t in il for t in ["stone", "furnace", "shield", "cake"]): return "stone"
        if any(t in il for t in ["plank", "stick", "wood", "dirt", "wool"]): return "wood"
        return "stone"

    def _infer_group(self, item):
        il = str(item).lower()
        if any(t in il for t in ["craft", "plank", "stick", "table", "ladder", "furnace",
                                  "stonecut", "shield", "cake", "potion", "enchant"]):
            return "crafting"
        if any(t in il for t in ["mine", "collect", "ore", "diamond", "iron", "obsidian",
                                  "wood", "dirt", "wool", "cobble"]):
            return "mining"
        if any(t in il for t in ["explore", "find", "village", "forest", "lava", "treasure",
                                  "bedrock", "map", "biome"]):
            return "exploration"
        if any(t in il for t in ["techtree", "nether_portal", "long", "tech_tree"]):
            return "tech_tree"
        if any(t in il for t in ["wrong_tool", "missing", "failure", "recover"]):
            return "failure_recovery"
        if any(t in il for t in ["interaction", "conflict", "limited", "compet"]):
            return "interaction_stress"
        return "crafting"

    def _infer_risk(self, item):
        il = str(item).lower()
        if any(t in il for t in ["diamond", "netherite", "lava", "combat", "witch",
                                  "portal", "ender"]):
            return "high"
        if any(t in il for t in ["iron", "obsidian", "cake", "potion", "enchant",
                                  "skeleton", "zombie"]):
            return "medium"
        return "low"

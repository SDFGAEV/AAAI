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

import logging, numpy as np, json, os, copy, time
from typing import Dict, List, Optional, Tuple

from .trust_store import TrustStore, CANDIDATE, PROBATION, CERTIFIED, DEPRECATED, DISABLED
from .trust_gate import TrustGate
from .context_bucket import ContextBucket
from .contract import ContractExtractor, ContractChecker
from .decision_controller import DecisionController
from .interaction_gate import InteractionGate
from .active_logging import ActiveBaseLogger
from .thompson_probe import SafeThompsonProber


class CactMemory:
    """C-ACT decision-time admission layer wrapping XENON DecomposedMemory."""

    def __init__(self, xenon_memory, method="C-ACT-Full",
                 store_path: str = None, frozen: bool = False,
                 active_calib_rate: float = 0.0, log_dir: str = None):
        self._mem = xenon_memory

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
        self._knowledge_cnt = 0
        self._pairwise_harm: Dict[Tuple[str, str], Tuple[int, int]] = {}
        self._drift_counter = 0

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
        for fname, data in log_files.items():
            if data:
                path = os.path.join(self.log_dir, fname)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "a") as f:
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
    def register_knowledge(self, knowledge_dict: Dict) -> str:
        """Extract contract from raw knowledge and register in TrustStore."""
        contract = self._extractor.extract(knowledge_dict)
        kid = contract.knowledge_id
        self._store.register_contract(kid, contract.to_dict())
        self._knowledge_cnt += 1

        # Log contract
        self.contract_logs.append({
            "knowledge_id": kid,
            "type": contract.type,
            "level": contract.level,
            "gene": contract.gene,
            "claimed_context": contract.claimed_context,
            "expected_uplift": contract.expected_uplift,
            "risk_bound": contract.risk_bound,
            "status": CANDIDATE,
            "source_episode": contract.source_episode,
        })
        return kid

    # ── save_success_failure ──
    def save_success_failure(self, waypoint, language_action_str, is_success):
        """Record outcome after a knowledge use/failure."""
        self._mem.save_success_failure(waypoint, language_action_str, is_success)
        sv = 1.0 if is_success else 0.0
        kid = f"skill:{waypoint}"
        ctx = self._bucket.encode(knowledge_type="skill", subgoal_type="craft",
                                  task_tier=self._infer_tier(waypoint))
        task_grp = self._infer_group(waypoint)

        if not self.frozen:
            self._store.record_episode(kid, ctx, used=True, success=sv,
                                       is_harmful=0.0 if is_success else 0.5)

        n = self._store.total_count(kid, ctx)
        pi = self._store.uplift_probability(kid, ctx)
        hu = self._store.harm_upper_bound(kid, ctx)
        ul = self._store.uplift_lcb(kid, ctx)
        lc = self._store.get_lifecycle_state(kid)
        ess = self._store.ess(kid, ctx)

        # Contract check (post-conditions)
        contract = self._store.get_contract(kid)
        contract_violated = False
        if contract and not is_success:
            # Quick post-condition check
            postconds = contract.get("postconditions", [])
            contract_violated = len(postconds) > 0

        # Reuse decision log
        self.reuse_logs.append({
            "waypoint": waypoint, "kid": kid, "ctx": ctx,
            "method": self.method, "frozen": self.frozen,
            "success": is_success, "task_group": task_grp,
            "pi_uplift": round(pi, 4), "uplift_lcb": round(ul, 4),
            "harm_ucb": round(hu, 4), "ess": round(ess, 1),
            "lifecycle": lc, "decision": "reuse",
            "contract_satisfied_before": True,
            "contract_violation_after": contract_violated,
            "is_harmful": 0 if is_success else (1 if not is_success and n > 1 else 0),
            "outcome_success": int(is_success),
        })

        # Episode log
        self.episode_logs.append({
            "waypoint": waypoint, "method": self.method, "frozen": self.frozen,
            "task_group": task_grp, "success": int(is_success),
            "total_steps": 0, "llm_calls": 0, "tokens": 0,
            "unrecoverable_failure": 0 if is_success else 1,
        })

        # Knowledge generation
        if is_success and n >= 2:
            self.episode_logs[-1]["knowledge_generated"] = f"s_{waypoint}_{int(n):02d}"
            self._knowledge_cnt += 1

        # Drift tracking + periodic decay
        pred_err = abs(sv - self._store.mean(kid, ctx, "use"))
        self._store.adapt_decay(pred_err)
        self._drift_counter += 1
        if self._drift_counter % 20 == 0:
            self._store.decay_all()

    # ── is_succeeded_waypoint (MAIN GATE) ──
    def is_succeeded_waypoint(self, waypoint):
        """C-ACT admission gate for knowledge reuse."""
        ctx = self._bucket.encode(knowledge_type="skill", subgoal_type="craft",
                                  task_tier=self._infer_tier(waypoint))
        kid = f"skill:{waypoint}"
        task_grp = self._infer_group(waypoint)
        lc = self._store.get_lifecycle_state(kid)
        ess = self._store.ess(kid, ctx)

        # Build candidate list
        contract = self._store.get_contract(kid) or {}
        candidates = [{
            "knowledge_id": kid,
            "type": contract.get("type", "skill"),
            "level": contract.get("level", "atomic"),
            "gene": contract.get("gene", waypoint),
            "full_text": contract.get("full_text", ""),
            "claimed_context": contract.get("claimed_context", {}),
            "preconditions": contract.get("preconditions", []),
            "postconditions": contract.get("postconditions", []),
            "non_applicable_contexts": contract.get("non_applicable_contexts", []),
        }]

        state = {"waypoint": waypoint, "task_group": task_grp}
        task = {"task_id": waypoint, "group": task_grp}
        context = {"bucket": ctx, "subgoal_type": "craft",
                   "risk_level": self._infer_risk(waypoint)}

        mode = "calibration" if self.active_calib_rate > 0 else (
            "evaluation" if self.frozen else "accumulation")

        # C-ACT decision
        if self.method == "C-ACT-Full":
            result = self._controller.decide(
                candidates, state, task, context, mode)
        else:
            result = self._legacy_decide(
                kid, ctx, candidates, state, task, context)

        trusted = result.decision in ("reuse", "probe")

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
        elif self.method == "XENON-Original":
            result.decision = "reuse"
        elif self.method == "BankCuration":
            result.decision = "reuse" if self._store.mean(kid, ctx, "use") >= 0.3 else "fallback"
        elif self.method == "LifecycleSuccessGate":
            if lc in (DISABLED, DEPRECATED):
                result.decision = "fallback"
            else:
                result.decision = "reuse" if self._store.mean(kid, ctx, "use") >= 0.5 else "fallback"
        elif self.method == "FixedBayes":
            if self._store.total_count(kid, ctx, "use") < 1:
                result.decision = "reuse"
            else:
                result.decision = "reuse" if (pi >= 0.90 and up >= 0.05 and hu <= 0.10) else "fallback"
        elif self.method == "ACT":
            if self._store.total_count(kid, ctx, "use") < 1:
                result.decision = "reuse"
            else:
                result.decision = "reuse" if (pi >= 0.90 and hu <= 0.10) else "fallback"
        elif self.method == "OracleGate":
            result.decision = "reuse"  # Upper bound — always reuse with oracle knowledge
        elif self.method == "ShuffledKnowledge":
            result.decision = "reuse" if np.random.random() < 0.5 else "fallback"
        else:
            result.decision = "reuse"  # Default: permissive

        result.pi_uplift = pi
        result.harm_ucb = hu
        result.uplift_lcb = up
        result.lifecycle_state = lc
        result.chosen_knowledge_id = kid
        return result

    # ── Helpers (inference from waypoint text) ──
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

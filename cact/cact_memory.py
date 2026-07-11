"""
C-ACT Memory: Contracted Adaptive Counterfactual Trust governance layer.

Wraps XENON DecomposedMemory with the C-ACT decision-time admission gate.

6 preregistered main methods:
  NoKnowledge             — Pure LLM, no knowledge base (E3 baseline)
  NoGate                  — Retrieve-and-reuse without gate
  FixedBayes              — Fixed-threshold Bayesian gate (no adaptive)
  PairwisePreferenceGate  — Trained pairwise preference gate; fail-fast if absent
  C-ACT-Pointwise         — v2 certificate without episode ledger
  C-ACT                   — v2 certificate with no-credit episode ledger

Legacy/appendix methods remain for compatibility:
  XENON-Original, BankCuration, LifecycleSuccessGate, ACT, C-ACT-Full,
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
from .preference_gate import PairwisePreferenceModel
from .protocol_v2 import (Opportunity, OpportunityLogger, RandomizedAssignment,
                          AdmissionPolicyV2, ApplicabilitySpec, SCHEMA_VERSION,
                          validate_method_name)


class CactMemory:
    """C-ACT decision-time admission layer wrapping XENON DecomposedMemory."""

    def __init__(self, xenon_memory, method="C-ACT",
                 store_path: str = None, frozen: bool = False,
                 active_calib_rate: float = 0.0, log_dir: str = None,
                 calibration_path: str = None, protocol_path: str = None,
                 protocol_seed: int = 0, branch_mode: str = "",
                 branch_target_opportunity: str = "", branch_parent_id: str = "",
                 branch_prefix_assignment: int = 0,
                 branch_prefix_trace: str = "",
                 kappa_override: str = None, snapshot_hash: str = ""):

        self._mem = xenon_memory
        requested_method = method
        method = validate_method_name(method, allow_legacy=True)

        # Component flags (for ablation)
        self._use_contract = True
        self._use_adaptive_tau = True
        self._use_active_calib = True
        self._use_interaction = True
        self._use_level_prior = True
        self._use_lifecycle = True
        self._use_attribution = True  # w/o Attribution Lifecycle ablation (doc §14)
        self._use_thompson = True
        self._use_sanitizer = True
        self._use_decay = True

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
            self._use_attribution = False  # Full lifecycle removal implies no attribution
        if "NoAttribution" in method or "no_Attribution" in method:
            self._use_attribution = False  # Only remove attribution, keep state machine
        if "NoThompson" in method or "no_Thompson" in method:
            self._use_thompson = False
        if "NoSanitizer" in method or "no_Sanitizer" in method:
            self._use_sanitizer = False
        if "NoDecay" in method or "NoTemporalDecay" in method or "no_Decay" in method:
            self._use_decay = False

        # Core C-ACT components
        self._store = TrustStore(store_path or "cact_ckpt/trust_store")
        self._gate = TrustGate()
        self._protocol_enabled = bool(protocol_path)
        self._v2_policy = None
        self._v2_logger = None
        self._v2_assigner = RandomizedAssignment(probability=0.5, seed=protocol_seed)
        self._v2_pending = None
        self._preference_model = None
        if method == "PairwisePreferenceGate":
            preference_path = os.environ.get("CACT_PREFERENCE_PATH", "")
            if not preference_path or not os.path.exists(preference_path):
                raise FileNotFoundError("PairwisePreferenceGate requires CACT_PREFERENCE_PATH")
            self._preference_model = PairwisePreferenceModel.load(preference_path)
        if protocol_path and os.path.exists(protocol_path):
            use_ledger = method not in ("C-ACT-Pointwise", "Online-C-ACT-Pointwise")
            family = "pointwise" if method in ("C-ACT-Pointwise", "Online-C-ACT-Pointwise") else "full"
            self._v2_policy = AdmissionPolicyV2.load(
                protocol_path, use_ledger=use_ledger, family=family)
            # E2 direct select: override calibrated kappa for matched-risk rollout
            if kappa_override:
                self._v2_policy.policy.kappa = float(kappa_override)
        calibration_path = calibration_path or os.environ.get("CACT_CALIBRATION_PATH")
        if calibration_path:
            self._gate.load_calibration(calibration_path)
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
        self.requested_method = requested_method
        self.frozen = frozen
        self.active_calib_rate = active_calib_rate
        self.branch_mode = str(branch_mode or "").lower()
        if self.branch_mode not in {"", "reuse", "base"}:
            raise ValueError("branch_mode must be empty, 'reuse', or 'base'")
        self.branch_target_opportunity = str(branch_target_opportunity or "")
        self.branch_parent_id = str(branch_parent_id or "")
        self.branch_prefix_assignment = int(bool(branch_prefix_assignment))
        try:
            self.branch_prefix_trace = {str(k): int(bool(v)) for k, v in json.loads(branch_prefix_trace or "{}").items()}
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("branch_prefix_trace must be a JSON object of opportunity_id to assignment") from exc
        self._snapshot_hash = str(snapshot_hash or "")
        self._branch_triggered = False
        self.log_dir = log_dir
        if self._protocol_enabled and log_dir:
            self._v2_logger = OpportunityLogger(os.path.join(log_dir, "opportunities.jsonl"))

        # Log buffers
        self.episode_logs: List[Dict] = []
        self.contract_logs: List[Dict] = []
        self.reuse_logs: List[Dict] = []
        self.base_logs: List[Dict] = []
        self.interaction_logs: List[Dict] = []
        self.lifecycle_logs: List[Dict] = []
        self.sanitizer_logs: List[Dict] = []

        # State tracking
        self._prev_kid = None
        self._last_was_supervised = False
        self._supervision_pending = False
        self._supervision_blocked = False
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
        self._inventory_snapshot: Dict[str, int] = {}  # item→count when knowledge was used
        self._current_seed = 0      # experiment seed
        self._episode_counter = 0
        self._episode_id = "episode-0"
        self._current_environment = None
        self._current_round = 0     # experiment round
        self._reuse_count = 0       # number of reuse decisions
        self._fallback_count = 0    # number of fallback decisions
        self._harmful_count = 0     # number of harmful reuse decisions

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
    def set_current_environment(self, env_name):
        self._current_environment = env_name
        setter = getattr(self._mem, "set_current_environment", None)
        return setter(env_name) if callable(setter) else None
    @property
    def current_environment(self):
        return getattr(self._mem, "current_environment", self._current_environment)
    @current_environment.setter
    def current_environment(self, value):
        self.set_current_environment(value)

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
        self._store.abl_decay = self._use_decay
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
        self._supervision_blocked = False
        self._episode_counter += 1
        self._episode_id = f"ep_{self._current_seed}_{self._episode_counter}"
        self._branch_triggered = False
        self._current_difficulty = difficulty
        self._current_group = group

    def set_seed(self, seed: int):
        """Set experiment seed for logging."""
        self._current_seed = seed

    def set_round(self, r: int):
        """Set experiment round for logging."""
        self._current_round = r

    def needs_supervision_check(self) -> bool:
        """Check if the last reuse decision was in supervised (Probation) mode.

        When True, the caller should run an immediate reflection check after
        each execution step. If the check fails, fall back to base policy.
        """
        return self._supervision_pending

    def acknowledge_supervision(self, passed: bool = True, observation: Dict = None,
                                info: Dict = None) -> bool:
        """Consume the probation verification hook from the environment loop."""
        if not self._supervision_pending:
            return True
        if observation is not None or info is not None:
            self.set_observation(observation, info)
        explicit = (info or {}).get("reflection_passed") if isinstance(info, dict) else None
        ok = bool(passed if explicit is None else explicit)
        self._supervision_pending = False
        self._last_was_supervised = False
        if not ok:
            self._supervision_blocked = True
        return ok

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
            "sanitizer/sanitizer.jsonl": self.sanitizer_logs,
        }
        # Always append — each worker has its own log_dir (seed+method unique),
        # and within a worker, episodes accumulate. Parallel workers don't conflict.
        mode = "a"
        self._logs_dumped = True
        for fname, data in log_files.items():
            if data:
                for event in data:
                    event.setdefault("schema_version", "cact.v1")
                    event.setdefault("run_id", os.path.basename(self.log_dir))
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
        self.sanitizer_logs.clear()

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
            # Log sanitizer actions
            for action in actions:
                self.sanitizer_logs.append({
                    "knowledge_id": action.knowledge_id,
                    "action": action.action,
                    "reason": action.reason,
                    "merge_group": action.merge_group,
                    "merged_into": action.merged_into,
                })
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
            "scope": "|".join(str(v) for v in contract.scope.values()) if contract.scope else "",
            "preconditions": contract.preconditions,
            "postconditions": contract.postconditions,
            "hard_non_applicable_contexts": contract.get_safety_boundaries(),
            "expected_uplift": contract.expected_uplift,
            "risk_bound": contract.risk_bound,
            "status": knowledge_dict.get("status", CANDIDATE),
            "source_episode": contract.source_episode,
        })
        return kid

    # ── save_success_failure ──
    def set_outcome_labels(self, harmful_reuse=None, postcondition_satisfied=None,
                           resource_conflict=False, chain_success=True):
        self._pending_outcome = {"harmful_reuse": harmful_reuse,
            "postcondition_satisfied": postcondition_satisfied,
            "resource_conflict": resource_conflict, "chain_success": chain_success}

    def save_success_failure(self, waypoint, language_action_str, is_success,
                             harmful_reuse=None, postcondition_satisfied=None,
                             resource_conflict=False, chain_success=True):
        """Record outcome after a knowledge use/failure with attribution."""
        if not self.frozen:
            self._mem.save_success_failure(waypoint, language_action_str, is_success)
        sv = 1.0 if is_success else 0.0
        # Look up the registered knowledge_id for this waypoint
        kid = self._waypoint_to_kid.get(waypoint, f"skill:{waypoint}")
        ctx = self._bucket.encode(knowledge_type="skill",
                                  subgoal_type=self._infer_subgoal_type(waypoint),
                                  task_tier=self._infer_tier(waypoint),
                                  risk_level=self._infer_risk(waypoint))
        task_grp = self._infer_group(waypoint)

        # Use last decision result
        last = getattr(self, '_last_decision_result', None)
        was_used = last.decision in ("reuse", "probe") if last else False

        # Increment reuse/fallback/harmful counters
        if was_used:
            self._reuse_count += 1
        else:
            self._fallback_count += 1

        ess = self._store.ess(kid, ctx)
        pi = self._store.uplift_probability(kid, ctx)
        hu = self._store.harm_upper_bound(kid, ctx)
        ul = self._store.uplift_lcb(kid, ctx)
        lc = self._store.get_lifecycle_state(kid)

        # Contract / interaction status
        contract = self._store.get_contract(kid)
        pending = getattr(self, "_pending_outcome", {})
        if harmful_reuse is None:
            harmful_reuse = pending.get("harmful_reuse")
        if postcondition_satisfied is None:
            postcondition_satisfied = pending.get("postcondition_satisfied")
        resource_conflict = pending.get("resource_conflict", resource_conflict)
        chain_success = pending.get("chain_success", chain_success)
        self._pending_outcome = {}

        # Contract violation and harm are observed labels, never posterior proxies.
        contract_violated = (postcondition_satisfied is False and contract is not None)
        interaction_conflict = not last.interaction_safe if last else bool(resource_conflict)
        csr_before = last.contract_satisfied_before if last else True
        is_harmful = int(bool(harmful_reuse) and was_used) if harmful_reuse is not None else int(bool(was_used and not is_success))

        if is_harmful:
            self._harmful_count += 1
        if self._v2_pending is not None:
            opp = self._v2_pending
            opp.y = int(bool(is_success))
            opp.h1 = int(bool(is_harmful)) if self._current_info.get("harm_code") == "H1" else 0
            opp.h2 = int(bool(is_harmful)) if self._current_info.get("harm_code") == "H2" else 0
            opp.h3 = int(bool(is_harmful)) if self._current_info.get("harm_code") == "H3" else 0
            opp.h4 = int(bool(is_harmful)) if self._current_info.get("harm_code") not in ("H1", "H2", "H3") else 0
            opp.h5 = int(float(self._current_info.get("resource_cost", 0.0) or 0.0) > float(self._current_info.get("resource_budget", float("inf"))))
            opp.h6 = int(bool(self._current_info.get("paired_reuse_failed_base_success", False)))
            opp.censor_flag = bool(self._current_info.get("censor_flag", False))
            opp.second_intervention_flag = bool(self._current_info.get("second_intervention", False))
            opp.progress_delta = float(self._current_info.get("progress_delta", 0.0) or 0.0)
            opp.resource_cost = float(self._current_info.get("resource_cost", 0.0) or 0.0)
            opp.label_source = "environment"
            if self._v2_logger:
                self._v2_logger.append(opp)
            self._v2_pending = None

        # ── Lifecycle update ──
        if not self.frozen:
            if self._use_attribution:
                # Full attribution-aware path (doc §14): classify outcome,
                # avoid penalizing knowledge for navigation/execution failures
                progress_delta = 0.0
                if self._inventory_snapshot and self._current_observation:
                    obs_inv = self._current_observation.get("inventory", {})
                    if isinstance(obs_inv, dict):
                        for item, snap_count in self._inventory_snapshot.items():
                            cur = int(obs_inv.get(item, 0))
                            if cur > snap_count:
                                progress_delta = 1.0
                                break
                self._inventory_snapshot = {}

                attribution = self._attributor.attribute(
                    success=is_success,
                    is_harmful=bool(is_harmful),
                    contract_violated=contract_violated,
                    interaction_conflict=interaction_conflict,
                    used_knowledge=was_used,
                    progress_delta=progress_delta,
                )
                attr_result = apply_attribution_to_lifecycle(
                    kid, attribution, sv, float(is_harmful),
                    self._store, ctx, self._attributor,
                    propensity=(last.propensity_reuse if was_used else last.propensity_base) if last else None)
            else:
                # w/o Attribution Lifecycle: simple use/harm recording,
                # ALL failures penalize knowledge (no navigation/execution distinction)
                self._store.record_episode(kid, ctx, used=was_used, success=sv,
                                           is_harmful=float(is_harmful),
                                           propensity=(last.propensity_reuse if was_used else last.propensity_base) if last else None,
                                           episode_id=f"ep_{waypoint}_{self._current_seed}_{self.method}",
                                           postcondition_satisfied=postcondition_satisfied)
                attribution = None
                attr_result = {"action": "simple_record", "attribution": "none"}
                progress_delta = 0.0
        else:
            attribution = None
            attr_result = {"action": "frozen", "attribution": "none"}
            progress_delta = 0.0

        if not self.frozen and self._use_attribution:
            self._store._observations.append({
                "kid": kid, "context": ctx, "used": bool(was_used),
                "success": float(sv), "is_harmful": float(is_harmful),
                "propensity": (last.propensity_reuse if was_used else last.propensity_base) if last else None,
                "episode_id": f"ep_{waypoint}_{self._current_seed}_{self.method}",
                "postcondition_satisfied": postcondition_satisfied,
            })

        # ── Wire lifecycle_logs ──
        self.lifecycle_logs.append({
            "knowledge_id": kid,
            "old_status": lc,
            "new_status": self._store.get_lifecycle_state(kid),
            "reason": attr_result.get("action", "unknown"),
            "pi_uplift": round(pi, 4),
            "harm_ucb": round(hu, 4),
            "round": getattr(self, '_current_round', 0),
        })

        # Reuse decision log (TABLE 152)
        self.reuse_logs.append({
            "decision_id": f"{waypoint}_{self._current_seed}_{self._reuse_count + self._fallback_count}",
            "opportunity_id": getattr(self._v2_pending, "opportunity_id", "") if self._v2_pending else "",
            "episode_id": f"ep_{waypoint}_{self._current_seed}_{self.method}",
            "branch_parent_id": self.branch_parent_id,
            "branch_mode": self.branch_mode,
            "branch_triggered": bool(self._branch_triggered),
            "waypoint": waypoint, "kid": kid, "ctx": ctx,
            "method": self.method, "frozen": self.frozen,
            "success": is_success, "task_group": task_grp,
            "pi_uplift": round(pi, 4), "uplift_lcb": round(ul, 4),
            "harm_ucb": round(hu, 4), "ess": round(ess, 1),
            "status_before": lc, "decision": last.decision if last else "reuse",
            "budget_before": getattr(last, "certificate_budget_before", None) if last else None,
            "budget_after": getattr(last, "certificate_budget_after", None) if last else None,
            "risk_charge": getattr(last, "certificate_risk_charge", None) if last else None,
            "pre_admit_contract_pass": csr_before,
            "contract_satisfied_before": csr_before,
            "contract_violation_after": contract_violated,
            "harmful_reuse": is_harmful,
            "tau_threshold": last.tau_group_level if last else 0.88,
            "delta_threshold": last.delta_group_level if last else 0.05,
            "harm_threshold": last.harm_threshold_group_level if last else 0.10,
            "hard_boundary_triggered": False,
            "interaction_state": last.interaction_state if last else "safe",
            "supervised": self._last_was_supervised,
            "postcondition_satisfied": postcondition_satisfied,
            "contract_satisfied_after": postcondition_satisfied,
            "resource_conflict": bool(resource_conflict),
            "chain_success": bool(chain_success),
            "progress_delta": progress_delta,
            # Observed environment costs; null means the environment did not
            # expose the field and paired collection must fail closed.
            "steps": self._current_info.get("window_steps", self._current_info.get("steps", self._current_info.get("num_steps"))),
            "resource_cost": self._current_info.get("resource_cost"),
            "snapshot_hash": self._current_info.get("snapshot_hash", ""),
            "world_seed": self._current_info.get("world_seed", self._current_seed),
            "attribution": attribution.value if attribution else "frozen",
            "attr_action": attr_result.get("action", "none"),
            "outcome_success": int(is_success),
            "propensity_reuse": last.propensity_reuse if last else None,
            "propensity_base": last.propensity_base if last else None,
            "propensity": (last.propensity_reuse if was_used else last.propensity_base) if last else None,
        })

        # Episode log
        self.episode_logs.append({
            "episode_id": f"ep_{waypoint}_{self._current_seed}_{self.method}",
            "task_id": waypoint,
            "waypoint": waypoint, "method": self.method, "frozen": self.frozen,
            "task_group": task_grp, "group": task_grp, "difficulty": self._current_difficulty,
            "seed": self._current_seed,
            "hard_task": self._current_difficulty == "hard",
            "success": int(is_success),
            "latency_sec": 0.0,
            "reuse_count": self._reuse_count,
            "fallback_count": self._fallback_count,
            "harmful_reuse_count": self._harmful_count,
            "total_steps": 0, "llm_calls": 0, "tokens": 0,
            "unrecoverable_failure": 0 if is_success else 1,
            "postcondition_pass": postcondition_satisfied,
            "harmful_reuse": int(is_harmful),
            "resource_conflict": bool(resource_conflict),
            "chain_success": bool(chain_success),
        })

        # Knowledge generation
        if not self.frozen and is_success and ess >= 2:
            self.episode_logs[-1]["knowledge_generated"] = f"gen_{waypoint}_{self._generated_cnt:02d}"
            self._generated_cnt += 1

        # Drift tracking + periodic decay
        if not self.frozen and self._use_decay:
            pred_err = abs(sv - self._store.mean(kid, ctx, "use"))
            self._store.adapt_decay(pred_err)
            self._drift_counter += 1
            if self._drift_counter % 20 == 0:
                self._store.decay_all()

        # Reset supervised flag — no longer in supervised context after recording
        self._last_was_supervised = False

        # Context bucket adaptive maintenance (doc §6.3)
        # Only in non-frozen modes (E1/E2/E5 calibration)
        if not self.frozen:
            self._bucket.accumulate(ctx, pi, sv)
            if self._drift_counter % 30 == 0:  # Check every 30 episodes
                self._bucket.maintain(allowed=True)

    def _make_v2_opportunity(self, waypoint, kid, ctx, task_grp, assignment, propensity, random_seed):
        contract = self._store.get_contract(kid) or {}
        obs = self._current_observation or {}
        inv = obs.get("inventory", {}) if isinstance(obs, dict) else {}
        inv_sig = "|".join(sorted(str(k) for k in inv)[:16])
        raw = str(contract.get("raw_text", contract.get("full_text", contract.get("gene", waypoint))))
        import hashlib
        blocked = []
        if bool(self._current_info.get("global_safety_block", False)):
            blocked.append("global_safety_shield")
        if bool(self._current_info.get("high_risk", False)):
            blocked.append("high_risk")
        if bool(self._current_info.get("task_finished", False)):
            blocked.append("task_finished")
        if bool(self._current_info.get("duplicate_action", False)):
            blocked.append("duplicate_action")
        if self._current_info.get("window_can_mask", True) is False:
            blocked.append("window_cannot_mask")
        eligible = not blocked
        return Opportunity(
            episode_id=self._episode_id,
            opportunity_id=f"opp_{waypoint}_{self._current_seed}_{self._reuse_count + self._fallback_count + 1}",
            round=self._current_round, stream_seed=self._current_seed,
            task_id=str(waypoint), world_seed=int(self._current_info.get("world_seed", self._current_seed) or 0),
            knowledge_id=kid, source=str(contract.get("source", "FAM")),
            type=str(contract.get("type", "skill")), retrieval_rank=1,
            retrieval_score=float(contract.get("retrieval_score", 1.0)),
            raw_text_hash=hashlib.sha256(raw.encode("utf-8")).hexdigest(),
            task_group=task_grp, failure_type=str(self._current_info.get("failure_type", "none")),
            risk_tier=self._infer_risk(waypoint), resource_scarcity=str(self._current_info.get("resource_scarcity", "ordinary")),
            boundary_status="blocked" if blocked else "applicable", inventory_signature=inv_sig,
            episode_phase=str(self._current_info.get("episode_phase", "early")),
            prior_admission_bin=str(self._current_info.get("prior_admission_bin", "0")),
            prior_fallback_bin=str(self._current_info.get("prior_fallback_bin", "0")),
            prior_harm_flag=int(bool(self._harmful_count)),
            remaining_critical_resource_ratio=float(self._current_info.get("remaining_critical_resource_ratio", 1.0) or 1.0),
            time_since_last_window=int(self._current_info.get("time_since_last_window", 0) or 0),
            collection_exposure_count=int(self._current_info.get("collection_exposure_count", 0) or 0),
            assignment=int(assignment), propensity_reuse=float(propensity),
            propensity_base=float(1.0 - propensity), randomization_seed=int(random_seed),
            start_step=0, end_step=0, censor_flag=bool(self._current_info.get("censor_flag", False)),
            window_type=str(self._current_info.get("window_type", "fixed")),
            second_intervention_flag=bool(self._current_info.get("second_intervention", False)), eligible=eligible,
            eligibility_reason="eligible" if eligible else blocked[0],
            annotator_status=str(self._current_info.get("annotator_status", "not_applicable")),
            snapshot_hash=str(self._current_info.get("snapshot_hash") or self._snapshot_hash),
        )

    # ── is_succeeded_waypoint (MAIN GATE) ──
    def is_succeeded_waypoint(self, waypoint):
        """C-ACT admission gate for knowledge reuse."""
        ctx = self._bucket.encode(knowledge_type="skill",
                                  subgoal_type=self._infer_subgoal_type(waypoint),
                                  task_tier=self._infer_tier(waypoint),
                                  risk_level=self._infer_risk(waypoint))
        kid = self._waypoint_to_kid.get(waypoint, f"skill:{waypoint}")
        task_grp = self._infer_group(waypoint)
        lc = self._store.get_lifecycle_state(kid)
        ess = self._store.ess(kid, ctx)

        # Build candidate list
        contract = self._store.get_contract(kid) or {}
        if not contract and self.method in ("NoKnowledge", "Base-Only"):
            self._last_decision_result = None
            return False, None
        if not contract:
            if self.frozen:
                # Frozen evaluation cannot create or mutate knowledge.
                self._last_decision_result = None
                return False, None
            kid = self.register_knowledge({"type": "skill", "subgoal": waypoint,
                                           "correction": str(waypoint),
                                           "group": task_grp}) or kid
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

            # Equipment: what tool/weapon is currently equipped.
            # mineRL may return nested dicts (e.g. {"mainhand": {"type": "iron_pickaxe", ...}})
            eq = obs.get("equipped_items", {})
            if isinstance(eq, dict):
                for k, v in eq.items():
                    if isinstance(v, dict):
                        state[k] = dict(v)  # shallow copy nested dict
                    elif isinstance(v, str) and v != "none":
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
            state["near_lava"] = bool(info.get("near_lava", obs.get("near_lava", "lava_bucket" in inv_items)))
            state["in_combat"] = bool(info.get("in_combat", False) or ({"iron_sword", "diamond_sword", "stone_sword",
                "wooden_sword", "shield", "bow"} & inv_items) and eq.get("mainhand", {}).get(
                "type", "") in ("sword", "axe", "bow"))
            life = obs.get("life_stats", {}) if isinstance(obs.get("life_stats", {}), dict) else {}
            state["low_health"] = bool(info.get("low_health", life.get("health", life.get("life", 20)) < 5))
            state["near_cliff"] = bool(info.get("near_cliff", obs.get("near_cliff", False)))
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

        if self.frozen:
            mode = "evaluation"
        elif self.method.startswith("Online-"):
            mode = "online"
        elif self.active_calib_rate > 0:
            mode = "calibration"
        else:
            mode = "accumulation"

        if self._supervision_blocked:
            return False, None

        # Protocol v2 decision: randomized D_fit/D_select logging or frozen
        # calibrated policy. The legacy controller remains a compatibility path
        # only when no v2 protocol artifact is supplied.
        use_v2 = self._protocol_enabled and self._v2_logger is not None and self.method in (
            "C-ACT", "C-ACT-Pointwise", "Online-C-ACT", "Online-C-ACT-Pointwise")
        if use_v2:
            if self.frozen and self._v2_policy is None:
                raise FileNotFoundError(
                    f"frozen {self.method} requires a readable v2 policy artifact: {self.requested_method}")
            applicable = True
            if contract:
                contract_obj = self._controller._dict_to_contract(contract)
                scope_ok = self._checker.check_scope_match(contract_obj, context)
                pre_ok = self._checker.check_preconditions(contract_obj, state)[0]
                boundary_ok = self._checker.check_hard_boundary(contract_obj, state)[0]
                applicable = bool(scope_ok and pre_ok and boundary_ok)
            provisional = self._make_v2_opportunity(waypoint, kid, ctx, task_grp, 1, 0.5, 0)
            if not provisional.eligible or not applicable:
                assignment, propensity, random_seed = 0, 0.5, 0
                gate = {"decision": "FALLBACK", "reason": provisional.eligibility_reason if not provisional.eligible else "inapplicable", "depth": None}
            elif self.branch_mode:
                target = (not self.branch_target_opportunity or
                          provisional.opportunity_id == self.branch_target_opportunity)
                if target and not self._branch_triggered:
                    assignment = int(self.branch_mode == "reuse")
                    self._branch_triggered = True
                    reason = f"paired_branch_{self.branch_mode}"
                else:
                    assignment = self.branch_prefix_trace.get(provisional.opportunity_id, self.branch_prefix_assignment)
                    reason = "paired_prefix"
                propensity, random_seed = 0.5, 0
                gate = {"decision": "ADMIT" if assignment else "FALLBACK",
                        "reason": reason, "depth": None}
            elif self.frozen and self._v2_policy is not None:
                gate = self._v2_policy.decide(provisional, applicable=applicable)
                assignment, propensity, random_seed = int(gate["decision"] == "ADMIT"), 0.5, 0
            else:
                assignment, propensity, random_seed = self._v2_assigner.assign(
                    f"{waypoint}|{self._current_seed}|{self._reuse_count + self._fallback_count}")
                gate = {"decision": "ADMIT" if assignment else "FALLBACK",
                        "reason": "randomized_logging", "depth": None}
            result = __import__("cact.decision_controller", fromlist=["DecisionResult"]).DecisionResult()
            result.decision = "reuse" if assignment else "fallback"
            result.chosen_knowledge_id = kid if assignment else ""
            result.contract_satisfied_before = bool(applicable)
            result.propensity_reuse, result.propensity_base = propensity, 1.0 - propensity
            result.lifecycle_state = lc
            for _name, _value in gate.items():
                setattr(result, f"certificate_{_name}", _value)
            self._v2_pending = self._make_v2_opportunity(
                waypoint, kid, ctx, task_grp, assignment, propensity, random_seed)
        # C-ACT decision
        elif self.method in ("C-ACT", "Online-C-ACT") or self.method.startswith("C-ACT-"):
            self._sync_ablation_flags()
            result = self._controller.decide(
                candidates, state, task, context, mode)
        else:
            result = self._legacy_decide(
                kid, ctx, candidates, state, task, context)

        if mode in ("accumulation", "calibration", "online"):
            if result.propensity_reuse <= 0 and result.propensity_base <= 0:
                result.propensity_base = max(0.0, min(1.0, self.active_calib_rate))
                result.propensity_reuse = 1.0 - result.propensity_base
        else:
            result.propensity_reuse = 1.0 if result.decision in ("reuse", "probe") else 0.0
            result.propensity_base = 1.0 - result.propensity_reuse
        self._last_decision_result = result
        trusted = result.decision in ("reuse", "probe")
        self._last_was_supervised = getattr(result, "supervised", False)
        self._supervision_pending = self._last_was_supervised

        # Snapshot inventory when knowledge is used, for progress_delta computation
        if trusted and obs:
            inv = obs.get("inventory", {})
            self._inventory_snapshot = {}
            if isinstance(inv, dict):
                for k, v in inv.items():
                    try:
                        self._inventory_snapshot[k] = v.item() if hasattr(v, 'item') else int(v)
                    except (TypeError, ValueError):
                        pass
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
                "resource_conflict": True,
                "chain_success": False,
            })

        # Propensity logging
        if result.propensity_base > 0:
            self.base_logs.append({
                "decision_id": f"{waypoint}_step",
                "candidate_knowledge_id": result.chosen_knowledge_id,
                "assigned_action": result.decision,
                "propensity_reuse": result.propensity_reuse,
                "propensity_base": result.propensity_base,
                "context_bucket": ctx,
                "would_reuse_under_policy": True,
                "base_success": 0,
                "base_progress_delta": 0.0,
                "task_group": task_grp,
            })

        self._prev_kid = kid if trusted else None
        return final

    # ── Legacy decision methods (for baselines) ──
    def _legacy_decide(self, kid, ctx, candidates, state, task, context):
        """Decision logic for non-v2 and legacy methods."""
        from .decision_controller import DecisionResult
        result = DecisionResult()
        pi = self._store.uplift_probability(kid, ctx)
        up = self._store.uplift_lcb(kid, ctx)
        hu = self._store.harm_upper_bound(kid, ctx)
        ess = self._store.ess(kid, ctx)
        lc = self._store.get_lifecycle_state(kid)
        grp = task.get("group", "crafting")

        if self.method in ("NoKnowledge", "Base-Only"):
            result.decision = "fallback"
        elif self.method in ("NoGate", "XENON-Original", "Online-NoGate"):
            result.decision = "reuse"
        elif self.method in ("BankCuration", "Online-BankCuration"):
            result.decision = "reuse" if (self._store.mean(kid, ctx, "use") >= 0.70 and
                                            self._store.harm_upper_bound(kid, ctx) <= 0.10 and
                                            self._store.is_reusable(kid)) else "fallback"
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
        elif self.method == "PairwisePreferenceGate":
            row = dict(context)
            row.update({"task_group": grp, "knowledge_id": kid,
                        "source": (candidates[0].get("source", "") if candidates else ""),
                        "type": (candidates[0].get("type", "") if candidates else "")})
            pref = self._preference_model.decide(row)
            result.decision = "reuse" if pref["decision"] == "ADMIT" else "fallback"
            result.preference_probability = pref["preference_probability"]
        elif self.method == "C-ACT" or self.method == "Online-C-ACT":
            result.decision = "reuse"  # Full C-ACT handled by decision_controller
        elif self.method == "OracleGate":
            result.decision = "reuse" if bool(task.get("oracle_reuse", False)) else "fallback"
        elif self.method == "ShuffledKnowledge":
            result.decision = "reuse" if np.random.random() < 0.5 else "fallback"
        else:
            raise ValueError(f"unsupported C-ACT method: {self.method}")

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

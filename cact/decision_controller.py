"""
C-ACT Decision Controller — 7-step admission orchestration.

Implements the main C-ACT decision flow (Table 76 / Fig from C-ACT doc):

  1. Contract filtering → context match + precondition + non_applicable
  2. Lazy decay → apply temporal decay to candidates
  3. Evidence computation → π_uplift, harm_ucb, certified判定
  4. Chain building → build candidate knowledge chain
  5. Interaction check → pairwise冲突检测
  6. Final decision → reuse(certified best) / fallback
  7. Active logging / probing → only in accumulation/calibration/online modes

Returns DecisionResult with full audit trail.
"""

from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field

from .trust_store import TrustStore
from .trust_gate import TrustGate
from .interaction_gate import InteractionGate
from .active_logging import ActiveBaseLogger
from .thompson_probe import SafeThompsonProber
from .contract import ContractChecker, ContractExtractor


@dataclass
class DecisionResult:
    """Complete C-ACT decision output with full audit trail."""
    decision: str = "fallback"        # "reuse" | "fallback" | "probe" | "force_base"
    chosen_knowledge_id: str = ""
    chosen_contract: Dict = field(default_factory=dict)

    # Evidence
    pi_uplift: float = 0.0
    harm_ucb: float = 0.0
    uplift_lcb: float = 0.0

    # Conditions
    contract_satisfied_before: bool = True
    contract_violation_after: bool = False
    interaction_safe: bool = True
    interaction_state: str = "safe"
    synergy_pairs: List = field(default_factory=list)  # Synergistic knowledge combos

    # Gate thresholds (for logging)
    tau_group_level: float = 0.90
    delta_group_level: float = 0.05
    harm_threshold_group_level: float = 0.10

    # Propensity (for base logging / IPW)
    propensity_reuse: float = 0.0
    propensity_base: float = 0.0

    # Audit
    lifecycle_state: str = ""
    supervised: bool = False       # Probation knowledge requires double-verification
    filtered_count: int = 0
    scored_count: int = 0
    chain_length: int = 0


class DecisionController:
    """C-ACT 7-step decision orchestration."""

    def __init__(self, trust_store: TrustStore,
                 trust_gate: TrustGate,
                 interaction_gate: InteractionGate = None,
                 active_logger: ActiveBaseLogger = None,
                 thompson_prober: SafeThompsonProber = None,
                 contract_checker: ContractChecker = None):
        self.store = trust_store
        self.gate = trust_gate
        self.ig = interaction_gate or InteractionGate()
        self.al = active_logger or ActiveBaseLogger()
        self.tp = thompson_prober or SafeThompsonProber()
        self.cc = contract_checker or ContractChecker()

        # Ablation flags (synced from CactMemory via _sync_ablation_flags)
        # Note: adaptive_tau is controlled by TrustGate.abl_adaptive, synced separately
        self.abl_contract = True
        self.abl_interaction = True
        self.abl_active_calib = True
        self.abl_thompson = True

    def decide(self, candidate_knowledge: List[Dict],
               state: Dict, task: Dict, context: Dict,
               mode: str = "evaluation") -> DecisionResult:
        """Main C-ACT decision entry point.

        Args:
            candidate_knowledge: List of knowledge dicts with at least {knowledge_id, type, level, gene, ...}
            state: Current environment state dict
            task: Task dict with {task_id, group, difficulty, ...}
            context: Context dict with {subgoal_type, failure_type, risk_level, ...}
            mode: "accumulation" | "calibration" | "evaluation" | "online"

        Returns:
            DecisionResult with full decision and audit trail.
        """
        result = DecisionResult()

        # ── Step 1: Contract filtering ──
        if self.abl_contract:
            valid = self._contract_filter(candidate_knowledge, state, context)
        else:
            valid = candidate_knowledge
        result.filtered_count = len(valid)

        if not valid:
            result.decision = "fallback"
            return result

        # ── Step 2: Lazy decay ──
        if getattr(self.store, 'abl_decay', True):
            self.store.decay_all()

        # ── Step 3: Evidence computation ──
        scored = []
        for u in valid:
            kid = u["knowledge_id"]
            ctx_key = context.get("bucket", context.get("subgoal_type", "craft"))
            task_group = task.get("group", "crafting")

            pi = self.store.uplift_probability(kid, ctx_key)
            ul = self.store.uplift_lcb(kid, ctx_key)
            hu = self.store.harm_upper_bound(kid, ctx_key)
            lc = self.store.get_lifecycle_state(kid)

            # Contract pre-check (limited: state dict does not contain full game state.
            # Only waypoint/task_group available. Full precondition checking requires
            # deeper integration with XENON's observation pipeline.)
            contract = self.store.get_contract(kid)
            contract_ok = True
            if contract:
                contract_ok, _ = self.cc.check_preconditions(
                    self._dict_to_contract(contract), state)

            # Gate evaluation
            allow, info = self.gate.evaluate(
                pi_uplift=pi, uplift_lcb=ul, harm_ucb_val=hu,
                task_group=task_group, lifecycle_state=lc,
                contract_satisfied=contract_ok, interaction_safe=True,
            )
            # Probation safety guard (TABLE 95): no probation reuse in high-risk context
            if allow and lc == "probation" and context.get("risk_level") == "high":
                allow = False
                info["reason"] = "probation_high_risk_blocked"

            scored.append({
                "knowledge_id": kid,
                "contract": u,
                "pi_uplift": pi,
                "uplift_lcb": ul,
                "harm_ucb": hu,
                "lifecycle": lc,
                "certified": allow,
                "contract_ok": contract_ok,
                "gate_info": info,
            })

        result.scored_count = len(scored)
        if not scored:
            result.decision = "fallback"
            return result

        # ── Step 4: Build candidate chain ──
        # Sort by pi_uplift descending, enrich with stats for interaction check
        chain = sorted(scored, key=lambda x: x.get("synergy_score", x["pi_uplift"]), reverse=True)
        ctx_key = context.get("bucket", context.get("subgoal_type", "craft"))
        for c in chain:
            kid = c["knowledge_id"]
            a1, b1 = self.store.get_stats(kid, ctx_key, "use")
            a2, b2 = self.store.get_stats(kid, ctx_key, "base")
            c["use_alpha"] = a1; c["use_beta"] = b1
            c["base_alpha"] = a2; c["base_beta"] = b2
        result.chain_length = len(chain)

        # ── Step 5: Interaction check (conflict + synergy) ──
        result.interaction_safe = True
        result.interaction_state = "safe"
        result.synergy_pairs = []

        if self.abl_interaction and len(chain) >= 2:
            pair_stats = self.store.get_all_pair_stats(
                [c["knowledge_id"] for c in chain], ctx_key)
            chain_check = self.ig.check_chain(chain, pair_stats, context)
            result.interaction_safe = chain_check["safe"]
            result.interaction_state = chain_check["recommendation"]

            # Promote synergy combos: synergy affects RANKING only, not gate decision.
            # Per TABLE 102: "synergy 不能直接乘 posterior probability"
            for c in chain:
                c["synergy_score"] = c["pi_uplift"]  # Default: no synergy boost
            for combo in chain_check.get("recommended_combos", []):
                ki, kj = combo["pair"]
                result.synergy_pairs.append({
                    "pair": (ki, kj),
                    "pi_synergy": combo["pi_synergy"],
                })
                # Boost synergy_score for ranking, but NEVER modify pi_uplift
                for c in chain:
                    if c["knowledge_id"] in (ki, kj):
                        c["synergy_score"] = min(1.0, c["pi_uplift"] + 0.02)
                        c["synergy_boosted"] = True

        # ── Step 6: Make decision ──
        # Sort by synergy_score (which incorporates synergy boost for ranking),
        # but gate on certified flag which uses unmodified pi_uplift
        best = None
        for c in chain:
            if c["certified"] and c["contract_ok"]:
                if result.interaction_safe:
                    best = c
                    break

        if best is not None:
            result.decision = "reuse"
            result.chosen_knowledge_id = best["knowledge_id"]
            result.chosen_contract = best["contract"]
            result.pi_uplift = best["pi_uplift"]
            result.uplift_lcb = best["uplift_lcb"]
            result.harm_ucb = best["harm_ucb"]
            result.lifecycle_state = best["lifecycle"]
            # Populate thresholds from gate info
            gi = best["gate_info"]
            result.tau_group_level = gi.get("tau", 0.90)
            result.delta_group_level = gi.get("delta", 0.05)
            result.harm_threshold_group_level = gi.get("harm", 0.10)
            result.supervised = gi.get("supervised", False)
        else:
            result.decision = "fallback"
            result.chosen_knowledge_id = ""

        # ── Step 7: Active logging / Thompson probing (skip if ablated) ──
        if (self.abl_active_calib or self.abl_thompson) and mode in ("accumulation", "calibration", "online"):
            result = self._apply_active_ops(result, scored, state, context, mode)

        return result

    # ── Contract filtering ──
    def _contract_filter(self, candidates: List[Dict],
                         state: Dict, context: Dict) -> List[Dict]:
        """Filter candidates by contract context match + preconditions + safety."""
        valid = []
        for u in candidates:
            # Check context match
            claimed = u.get("claimed_context", {})
            if claimed and not self.cc.check_context_match(
                self._dict_to_contract(u), context):
                continue

            # Check non-applicable contexts (hard safety boundaries)
            non_app = u.get("non_applicable_contexts", [])
            if non_app:
                safe, triggered = self.cc.check_non_applicable(
                    self._dict_to_contract(u), state)
                if not safe:
                    continue  # Hard safety violation → never allow

            # Check preconditions
            preconds = u.get("preconditions", [])
            if preconds:
                ok, _ = self.cc.check_preconditions(
                    self._dict_to_contract(u), state)
                if not ok:
                    continue  # Precondition not met → can't reuse

            valid.append(u)
        return valid

    # ── Active logging / probing ──
    def _apply_active_ops(self, result: DecisionResult,
                          scored: List[Dict],
                          state: Dict, context: Dict,
                          mode: str) -> DecisionResult:
        """Apply active base logging and Thompson probing (only in non-eval modes).
        Caller gates on mode — this method assumes mode != 'evaluation'."""
        # Active base logging: possibly override reuse → force_base

        # Active base logging: possibly override reuse → force_base (skip if ablated)
        if self.abl_active_calib and result.decision == "reuse" and result.chosen_knowledge_id:
            kid = result.chosen_knowledge_id
            ctx_key = context.get("bucket", "default")
            uncertainty = ActiveBaseLogger.compute_uncertainty(result.pi_uplift)
            n_use = self.store.total_count(kid, ctx_key, "use")
            n_base = self.store.total_count(kid, ctx_key, "base")
            imbalance = ActiveBaseLogger.compute_imbalance(n_use, n_base)
            danger = ActiveBaseLogger.compute_danger_score(state)

            should_force, q = self.al.should_force_base(
                uncertainty, imbalance, danger, context.get("risk_level", "medium"))

            if should_force:
                result.decision = "force_base"
                result.propensity_base = q
                result.propensity_reuse = 1.0 - q
                return result

        # Thompson probing: possibly allow reuse for uncertain knowledge (skip if ablated)
        if self.abl_thompson and result.decision == "fallback" and scored:
            for c in sorted(scored, key=lambda x: x["pi_uplift"], reverse=True):
                kid = c["knowledge_id"]
                ctx_key = context.get("bucket", "default")
                a1, b1 = self.store.get_stats(kid, ctx_key, "use")
                a2, b2 = self.store.get_stats(kid, ctx_key, "base")
                a3, b3 = self.store.get_stats(kid, ctx_key, "harm")
                ess = self.store.ess(kid, ctx_key)

                n_base = self.store.total_count(kid, ctx_key, "base")
                should_probe, q = self.tp.should_probe(
                    a1, b1, a2, b2, a3, b3, ess,
                    risk_level=context.get("risk_level", "medium"),
                    pi_uplift=c.get("pi_uplift", 0.5),
                    tau_threshold=c.get("gate_info", {}).get("tau", 0.90),
                    n_base=n_base,
                    interaction_state=result.interaction_state)
                if should_probe:
                    result.decision = "probe"
                    result.chosen_knowledge_id = kid
                    result.chosen_contract = c.get("contract", {})
                    result.pi_uplift = c["pi_uplift"]
                    return result

        return result

    # ── Helpers ──
    @staticmethod
    def _dict_to_contract(d: Dict):
        """Build a lightweight contract-like object from dict for ContractChecker."""
        from .contract import KnowledgeContract
        return KnowledgeContract(
            knowledge_id=d.get("knowledge_id", ""),
            type=d.get("type", ""),
            level=d.get("level", ""),
            gene=d.get("gene", ""),
            full_text=d.get("full_text", d.get("gene", "")),
            claimed_context=d.get("claimed_context", {}),
            preconditions=d.get("preconditions", []),
            postconditions=d.get("postconditions", []),
            non_applicable_contexts=d.get("non_applicable_contexts", []),
        )

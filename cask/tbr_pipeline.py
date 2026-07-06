"""
TBR Pipeline: Trust Before Reuse — 完整 4 层可信准入管线

用法:
  pipeline = TBRPipeline(flags={"certify": True, "calibrate": True, "compose": True, "evolve": True})
  pipeline.process_episode(agent_version, knowledge_chain, outcomes)

消融开关:
  certify:    Layer 1 — Counterfactual Certificate
  calibrate:  Layer 2 — Risk-Calibrated Gate
  compose:    Layer 3 — Interaction-Aware Composition
  evolve:     Layer 4 — Safe Evolution

Git: pre-upgrade commit done. Now implementing 4-layer full stack.
"""

import json, os, time
from typing import Dict, List, Tuple, Optional

from .trust_store import TrustStore
from .trust_gate import TrustGate


class TBRPipeline:
    """
    Trust Before Reuse — 4-layer self-evolution certifier.
    """

    def __init__(self,
                 flags: Dict[str, bool] = None,
                 epsilon: float = 0.1,
                 version_check_interval: int = 20):
        self.flags = flags or {
            "certify": True, "calibrate": True,
            "compose": True, "evolve": True
        }
        self.epsilon = epsilon
        self.version_check_interval = version_check_interval

        # Core components
        self.store = TrustStore()
        self.gate = TrustGate(epsilon=epsilon)

        # Layer 3: Pairwise interaction tracking
        self._interaction_cache: Dict[str, Dict] = {}

        # Layer 4: Version state
        self.current_version = 1
        self._version_history: List[Dict] = []
        self._version_promoted = []

    # ═══════════════════════════════════════════════
    # Layer 1: Counterfactual Certificate
    # ═══════════════════════════════════════════════

    def certify_knowledge(self, kid: str, ctx: str) -> Dict:
        """
        Compute counterfactual trust certificate for a single knowledge item.

        Returns:
          { uplift_lcb, mean_uplift, harm_ucb, support_use, support_base,
            score, certified }
        """
        if not self.flags.get("certify", True):
            return {"certified": True, "score": 0, "uplift_lcb": 0, "note": "disabled"}

        uplift = self.store.uplift(kid, ctx)
        h_ucb = self.store.harm_ucb(kid, ctx)
        score = self.gate.trust_score(uplift, h_ucb)

        # Count support
        a_use, b_use = self.store.get_stats(kid, ctx, "use")
        a_base, b_base = self.store.get_stats(kid, ctx, "base")
        n_use = a_use + b_use - 2
        n_base = a_base + b_base - 2

        certified = (self.gate.t_epsilon is not None and
                      self.gate.should_reuse(uplift, h_ucb))

        return {
            "uplift_lcb": uplift,
            "mean_uplift": self.store.mean(kid, ctx, "use") - self.store.mean(kid, ctx, "base"),
            "harm_ucb": h_ucb,
            "score": score,
            "support_use": max(0, n_use),
            "support_base": max(0, n_base),
            "certified": certified
        }

    # ═══════════════════════════════════════════════
    # Layer 2: Calibrated Reuse Gate
    # ═══════════════════════════════════════════════

    def calibrate_gate(self, calib_data: List[Dict]) -> Dict:
        """Run calibration on collected data. Returns calibration stats."""
        if not self.flags.get("calibrate", True):
            self.gate.t_epsilon = 0.0
            return {"t_epsilon": 0.0, "note": "disabled"}
        return self.gate.calibrate(calib_data)

    # ═══════════════════════════════════════════════
    # Layer 3: Interaction-Aware Composition
    # ═══════════════════════════════════════════════

    def _interaction_key(self, kid_i: str, kid_j: str, ctx: str) -> str:
        ids = sorted([kid_i, kid_j])
        return f"{ids[0]}|{ids[1]}|{ctx}|interact"

    def record_pairwise_outcome(self, kid_i: str, kid_j: str, ctx: str,
                                used_both: bool, is_harmful: float):
        """Record outcome when both knowledge items are used together."""
        if not self.flags.get("compose", True):
            return
        ikey = self._interaction_key(kid_i, kid_j, ctx)
        if ikey not in self._interaction_cache:
            self._interaction_cache[ikey] = {"alpha": 1.0, "beta": 1.0}
        e = self._interaction_cache[ikey]
        if used_both:
            e["alpha"] += (1 - is_harmful)
            e["beta"] += is_harmful

    def get_pairwise_conflict_prob(self, kid_i: str, kid_j: str, ctx: str) -> float:
        """Get probability that i and j are harmful when used together."""
        ikey = self._interaction_key(kid_i, kid_j, ctx)
        e = self._interaction_cache.get(ikey, {"alpha": 1.0, "beta": 1.0})
        # beta represents harmful count, alpha = non-harmful
        return e["beta"] / (e["alpha"] + e["beta"])

    def check_plan_chain(self, knowledge_chain: List[str], ctx: str) -> Dict:
        """
        Check if a plan chain is compatible (all pairwise conflicts low).
        Returns {compatible, conflicts, plan_chain_score}
        """
        if not self.flags.get("compose", True):
            return {"compatible": True, "conflicts": [], "plan_chain_score": 0}

        conflicts = []
        chain_uplift = 0.0
        chain_penalty = 0.0
        N = len(knowledge_chain)

        for i, kid_i in enumerate(knowledge_chain):
            cert = self.certify_knowledge(kid_i, ctx)
            chain_uplift += cert["uplift_lcb"]
            for j, kid_j in enumerate(knowledge_chain):
                if j <= i: continue
                conflict_p = self.get_pairwise_conflict_prob(kid_i, kid_j, ctx)
                if conflict_p > 0.5:  # threshold for pairwise conflict
                    conflicts.append((kid_i, kid_j, conflict_p))
                    chain_penalty += conflict_p

        # Plan chain score: sum of individual uplifts minus length penalty minus conflict penalty
        chain_score = chain_uplift - 0.1 * N - 0.5 * chain_penalty
        compatible = len(conflicts) == 0

        return {
            "compatible": compatible,
            "conflicts": conflicts,
            "plan_chain_score": chain_score,
            "chain_uplift": chain_uplift,
            "length_penalty": 0.1 * N,
            "conflict_penalty": 0.5 * chain_penalty
        }

    # ═══════════════════════════════════════════════
    # Layer 4: Safe Evolution Version Check
    # ═══════════════════════════════════════════════

    def record_version_outcome(self, version: int, value: float, is_harmful: float):
        """Record aggregated performance of an agent version."""
        if not self.flags.get("evolve", True):
            return
        # Find or create version entry
        found = False
        for vh in self._version_history:
            if vh["version"] == version:
                vh["value_alpha"] += value
                vh["value_beta"] += (1 - value)
                vh["harm_alpha"] += is_harmful
                vh["harm_beta"] += (1 - is_harmful)
                found = True
                break
        if not found:
            self._version_history.append({
                "version": version,
                "value_alpha": value + 1.0,
                "value_beta": (1 - value) + 1.0,
                "harm_alpha": is_harmful + 1.0,
                "harm_beta": (1 - is_harmful) + 1.0,
            })

    def check_safe_upgrade(self, version_k1: int,
                           version_k: int = None) -> Dict:
        """
        Check if version_{k+1} is safe to promote over version_k.
        Uses LCB[ΔV] ≥ 0 ∧ UCB[ΔH] ≤ η.
        """
        if not self.flags.get("evolve", True):
            return {"allow_promote": True, "note": "disabled"}

        if version_k is None:
            version_k = self.current_version

        vh_new = None
        vh_old = None
        for vh in self._version_history:
            if vh["version"] == version_k1: vh_new = vh
            if vh["version"] == version_k: vh_old = vh

        if not vh_new or not vh_old:
            return {"allow_promote": False, "note": f"missing version data (have {version_k}, checking {version_k1})"}

        import scipy.stats
        beta = scipy.stats.beta

        # Value comparison: LCB[ΔV] = LCB[V_new] - UCB[V_old]
        lcb_new = float(beta.ppf(0.1, vh_new["value_alpha"], vh_new["value_beta"]))
        ucb_old = float(beta.ppf(0.9, vh_old["value_alpha"], vh_old["value_beta"]))
        delta_v_lcb = lcb_new - ucb_old

        # Harm comparison: UCB[ΔH] = UCB[H_new] - LCB[H_old]
        ucb_h_new = float(beta.ppf(0.9, vh_new["harm_alpha"], vh_new["harm_beta"]))
        lcb_h_old = float(beta.ppf(0.1, vh_old["harm_alpha"], vh_old["harm_beta"]))
        delta_h_ucb = ucb_h_new - lcb_h_old

        allow = delta_v_lcb >= 0 and delta_h_ucb <= 0.15  # η = 0.15

        result = {
            "allow_promote": allow,
            "delta_v_lcb": delta_v_lcb,
            "delta_h_ucb": delta_h_ucb,
            "v_new_lcb": lcb_new,
            "v_old_ucb": ucb_old,
            "h_new_ucb": ucb_h_new,
            "h_old_lcb": lcb_h_old,
            "reason": f"ΔV_lcb={delta_v_lcb:+.3f} {'≥0' if delta_v_lcb>=0 else '<0'} "
                      f"ΔH_ucb={delta_h_ucb:+.3f} {'≤0.15' if delta_h_ucb<=0.15 else '>0.15'}"
        }

        return result

    def promote_version(self, version: int):
        """Promote a version to active."""
        if self.flags.get("evolve", True):
            self.current_version = version
            self._version_promoted.append({
                "version": version, "timestamp": time.time()
            })

    def increment_version(self) -> int:
        """Move to next version."""
        self.current_version += 1
        return self.current_version

    # ═══════════════════════════════════════════════
    # Full Pipeline: process one episode
    # ═══════════════════════════════════════════════

    def process_episode(self, knowledge_chain: List[str],
                        outcomes: List[Dict], ctx: str = "craft") -> Dict:
        """
        Process one episode through all 4 layers.

        Args:
          knowledge_chain: [kid_1, kid_2, ...] used in this plan
          outcomes: [{kid, used, success, is_harmful}]
          ctx: context bucket

        Returns:
          { layer1, layer2, layer3, layer4, decisions }
        """
        result = {}

        # Layer 1: Certify each knowledge item
        certifications = {}
        for kid in knowledge_chain:
            certifications[kid] = self.certify_knowledge(kid, ctx)
        result["layer1"] = certifications

        # Layer 2: Gate decision per item
        gate_decisions = {}
        for kid, cert in certifications.items():
            if self.gate.t_epsilon is None:
                # Not calibrated yet — use direct uplift check
                gate_decisions[kid] = cert["uplift_lcb"] > 0
            else:
                gate_decisions[kid] = cert["certified"]
        result["layer2"] = gate_decisions

        # Layer 3: Plan chain compatibility
        chain_check = self.check_plan_chain(knowledge_chain, ctx)
        result["layer3"] = chain_check

        # Layer 4: Record outcomes + version check
        for outcome in outcomes:
            kid = outcome["kid"]
            used = outcome.get("used", True)
            success = outcome.get("success", 0.0)
            is_harmful = outcome.get("is_harmful", 0.0)

            # Record in trust store
            self.store.record_episode(kid, ctx, used=used,
                                      success=success, is_harmful=is_harmful)

            # Layer 3: Record pairwise interactions
            if used and self.flags.get("compose", True):
                for other_kid in knowledge_chain:
                    if other_kid != kid:
                        self.record_pairwise_outcome(kid, other_kid, ctx,
                                                     True, is_harmful)

            # Layer 4: Aggregate episode-level value
            if self.flags.get("evolve", True):
                self.record_version_outcome(self.current_version, success, is_harmful)

        # Version check if enough data
        version_check = None
        if (self.flags.get("evolve", True) and
            len(self._version_history) >= 2 and
            len(knowledge_chain) >= self.version_check_interval):
            prev_v = self._version_promoted[-1]["version"] if self._version_promoted else 1
            version_check = self.check_safe_upgrade(self.current_version, prev_v)
        result["layer4"] = version_check

        # Final decision for each item
        decisions = {}
        for kid in knowledge_chain:
            l1_ok = certifications[kid]["certified"]
            l2_ok = gate_decisions[kid]
            l3_ok = chain_check["compatible"] if self.flags.get("compose", True) else True
            l4_ok = version_check is None or version_check.get("allow_promote", True)

            if self.flags.get("evolve", True) and version_check:
                decisions[kid] = l1_ok and l2_ok and l3_ok and l4_ok
            else:
                decisions[kid] = l1_ok and l2_ok and l3_ok
        result["decisions"] = decisions

        return result

    def get_ablation_config(self) -> Dict:
        """Return current ablation flags for logging."""
        return {
            "certify": self.flags.get("certify", True),
            "calibrate": self.flags.get("calibrate", True),
            "compose": self.flags.get("compose", True),
            "evolve": self.flags.get("evolve", True),
            "epsilon": self.epsilon,
            "t_epsilon": self.gate.t_epsilon,
            "current_version": self.current_version,
        }

    def save_state(self, path: str):
        """Save full pipeline state for checkpointing."""
        state = {
            "ablation": self.get_ablation_config(),
            "interaction_cache": {k: v for k, v in self._interaction_cache.items()},
            "version_history": self._version_history,
            "version_promoted": self._version_promoted,
            "current_version": self.current_version,
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(state, f, indent=2)

    def load_state(self, path: str):
        """Load pipeline state from checkpoint."""
        if not os.path.exists(path):
            return
        with open(path) as f:
            state = json.load(f)
        self._interaction_cache = state.get("interaction_cache", {})
        self._version_history = state.get("version_history", [])
        self._version_promoted = state.get("version_promoted", [])
        self.current_version = state.get("current_version", 1)

"""
Interaction-Aware Reuse Gate — pairwise knowledge interaction detection.

When multiple knowledge items are reused simultaneously, their effects
may not be additive. The interaction gate checks pairwise interactions:

  Δ_int(u_i, u_j) = Δ(u_i ∧ u_j) − Δ(u_i) − Δ(u_j)

Decision table:
  | Pair State              | Action                        |
  |-------------------------|-------------------------------|
  | Certified + Synergy     | allow_pair                    |
  | Certified + Neutral     | allow_pair                    |
  | Certified + Conflict    | block_pair / single_best_only |
  | Unknown + Low-risk      | allow but log (pair_probe)    |
  | Unknown + High-risk     | force_fallback                |
  | Unknown + Resource-crit | force_fallback                |
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.stats import beta as beta_dist


# Interaction state labels
SYNERGY = "synergy"
NEUTRAL = "neutral"
CONFLICT = "conflict"
UNKNOWN = "unknown"

# Default thresholds
THETA_CONF_DEFAULT = 0.80  # P(conflict) threshold for blocking
THETA_SYN_DEFAULT = 0.70   # P(synergy) threshold for positive
MIN_PAIR_SUPPORT = 5       # Minimum joint observations for reliable state

# Interaction effect size threshold
DELTA_INT = 0.03  # Minimum interaction effect to declare non-neutral


class InteractionGate:
    """Pairwise knowledge interaction conflict detection."""

    def __init__(self, theta_conf: float = THETA_CONF_DEFAULT,
                 theta_syn: float = THETA_SYN_DEFAULT,
                 min_pair_support: int = MIN_PAIR_SUPPORT):
        self.theta_conf = theta_conf
        self.theta_syn = theta_syn
        self.min_pair_support = min_pair_support

    # ── Pair-level interaction check ──
    def check_pair(self, stats_i: Dict, stats_j: Dict,
                   joint_stats: Dict,
                   context: Dict = None) -> Dict:
        """Check interaction between two knowledge items.

        Args:
            stats_i, stats_j: Individual knowledge Beta stats
                {use_alpha, use_beta, base_alpha, base_beta}
            joint_stats: Joint observation stats
                {alpha, beta} for joint success Beta

        Returns:
            {delta_int, pi_syn, pi_conf, state, recommendation}
        """
        a_j, b_j = joint_stats.get("alpha", 1.0), joint_stats.get("beta", 1.0)
        n_joint = a_j + b_j - 2.0

        # Individual uplifts
        up_i = self._lcb(stats_i["use_alpha"], stats_i["use_beta"]) - \
               self._ucb(stats_i.get("base_alpha", 1.0), stats_i.get("base_beta", 1.0))
        up_j = self._lcb(stats_j["use_alpha"], stats_j["use_beta"]) - \
               self._ucb(stats_j.get("base_alpha", 1.0), stats_j.get("base_beta", 1.0))

        if n_joint < 2:
            return self._result(0.0, 0.0, UNKNOWN, "pair_probe",
                               "insufficient_joint_data")

        # Joint uplift LCB minus individual uplifts
        lcb_joint = float(beta_dist.ppf(0.05, a_j, b_j))
        ucb_base = self._ucb(stats_i.get("base_alpha", 1.0),
                            stats_i.get("base_beta", 1.0))
        up_ij = lcb_joint - ucb_base
        delta_mean = up_ij - up_i - up_j

        # Confidence bounds on interaction effect
        lcb_half = float(beta_dist.ppf(0.025, a_j, b_j))
        delta_lcb = (lcb_half - ucb_base) - up_i - up_j

        # State classification
        if delta_lcb > DELTA_INT:
            state = SYNERGY
            recommendation = "allow_pair"
        elif delta_lcb < -DELTA_INT * 1.5:
            state = CONFLICT
            recommendation = "block_pair"
        elif n_joint >= self.min_pair_support and abs(delta_lcb) <= DELTA_INT:
            state = NEUTRAL
            recommendation = "allow_pair"
        else:
            state = UNKNOWN
            # Check risk context
            if context and context.get("risk_level") == "high":
                recommendation = "force_fallback"
            elif context and context.get("resource_critical"):
                recommendation = "force_fallback"
            else:
                recommendation = "pair_probe"

        # Compute posterior probabilities
        pi_syn = float(beta_dist.cdf(1.0, a_j, b_j)) if state == SYNERGY else \
                 self._prob_positive(delta_lcb, DELTA_INT)
        pi_conf = float(1.0 - beta_dist.cdf(0.0, a_j, b_j)) if state == CONFLICT else \
                  self._prob_negative(delta_lcb, -DELTA_INT)

        return self._result(delta_mean, delta_lcb, state, recommendation, "")

    # ── Chain-level interaction check ──
    def check_chain(self, chain: List[Dict],
                    pair_stats: Dict[str, Dict],
                    context: Dict = None) -> Dict:
        """Check all pairwise interactions in a knowledge chain.

        Args:
            chain: [{knowledge_id, use_alpha, use_beta, base_alpha, base_beta}, ...]
            pair_stats: {(kid_i, kid_j): {alpha, beta}, ...}
            context: Current context dict with optional risk_level, resource_critical

        Returns:
            {safe, blocked_pairs, conflict_pairs, recommendation}
        """
        blocked_pairs = []
        conflict_pairs = []
        synergy_pairs = []

        for i in range(len(chain)):
            for j in range(i + 1, len(chain)):
                ki, kj = chain[i]["knowledge_id"], chain[j]["knowledge_id"]
                pair_key = (ki, kj)
                joint = pair_stats.get(pair_key, {"alpha": 1.0, "beta": 1.0})

                result = self.check_pair(
                    stats_i=chain[i],
                    stats_j=chain[j],
                    joint_stats=joint,
                    context=context,
                )

                if result["recommendation"] == "block_pair":
                    blocked_pairs.append({"pair": (ki, kj), "result": result})
                elif result["recommendation"] == "force_fallback":
                    blocked_pairs.append({"pair": (ki, kj), "result": result,
                                         "reason": "high_risk_unknown"})
                elif result["state"] == CONFLICT:
                    conflict_pairs.append({"pair": (ki, kj), "result": result})
                elif result["state"] == SYNERGY:
                    synergy_pairs.append({"pair": (ki, kj), "result": result})

        # Build recommended combos from synergy pairs
        recommended_combos = []
        for sp in synergy_pairs:
            ki, kj = sp["pair"]
            pi = sp["result"].get("pi_syn", 0.5)
            if pi >= self.theta_syn:
                recommended_combos.append({
                    "pair": (ki, kj),
                    "pi_synergy": pi,
                    "recommendation": "prefer_joint",
                })

        return {
            "safe": len(blocked_pairs) == 0,
            "blocked_pairs": blocked_pairs,
            "conflict_pairs": conflict_pairs,
            "synergy_pairs": synergy_pairs,
            "recommended_combos": recommended_combos,
            "recommendation": "interaction_safe" if len(blocked_pairs) == 0
                              else ("single_best_only" if len(blocked_pairs) == 1
                                    else "force_fallback"),
        }

    # ── Helpers ──
    @staticmethod
    def _lcb(alpha: float, beta_param: float, delta: float = 0.05) -> float:
        return float(beta_dist.ppf(delta, alpha, beta_param))

    @staticmethod
    def _ucb(alpha: float, beta_param: float, delta: float = 0.05) -> float:
        return float(beta_dist.ppf(1 - delta, alpha, beta_param))

    @staticmethod
    def _prob_positive(value: float, threshold: float) -> float:
        """Approximate P(Δ > threshold) using normal approx."""
        return min(0.99, max(0.01, 1.0 / (1.0 + np.exp(-5 * (value - threshold)))))

    @staticmethod
    def _prob_negative(value: float, threshold: float) -> float:
        """Approximate P(Δ < threshold)."""
        return min(0.99, max(0.01, 1.0 / (1.0 + np.exp(5 * (value - threshold)))))

    @staticmethod
    def _result(delta_mean: float, delta_lcb: float, state: str,
                recommendation: str, detail: str = "") -> Dict:
        return {
            "delta_mean": round(delta_mean, 4),
            "delta_lcb": round(delta_lcb, 4),
            "state": state,
            "pi_syn": 1.0 if state == SYNERGY else (0.0 if state == CONFLICT else 0.5),
            "pi_conf": 1.0 if state == CONFLICT else (0.0 if state == SYNERGY else 0.5),
            "recommendation": recommendation,
            "detail": detail,
        }

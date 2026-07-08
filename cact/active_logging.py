"""
Active Base Logging — randomized forced-fallback for counterfactual signal.

Problem: If we only observe outcomes when the agent chooses to reuse,
we have selection bias — we never see what happens WITHOUT the knowledge.

Solution: Randomly force 5–30% of reuse-eligible decisions to use
base/fallback instead. The randomization probability adapts to:
  - Uncertainty (higher uncertainty → more probing)
  - Sample imbalance (more imbalance → more probing)
  - Danger level (high risk → less / zero probing)

Each forced-base decision records its propensity for later
IPW/DR estimation.
"""

import json, random
from typing import Dict, Tuple, Optional


# Bounds for adaptive base logging rate
Q_BASE_MIN = 0.05
Q_BASE_MAX = 0.30
TARGET_AVERAGE = 0.15

# Danger flags where base logging is strongly suppressed
HIGH_DANGER_FLAGS = [
    "lava_nearby", "low_health", "combat",
    "near_cliff", "irreversible_resource_constraint"
]


class ActiveBaseLogger:
    """Adaptive force-base logger with propensity tracking."""

    def __init__(self, q_min: float = Q_BASE_MIN, q_max: float = Q_BASE_MAX,
                 target_avg: float = TARGET_AVERAGE,
                 log_path: str = None):
        self.q_min = q_min
        self.q_max = q_max
        self.target_avg = target_avg
        self._log_path = log_path
        self._total_decisions = 0
        self._force_base_count = 0

    def base_probability(self, uncertainty: float,
                         sample_imbalance: float = 0.0,
                         danger_score: float = 0.0) -> float:
        """Compute adaptive base logging probability.

        Args:
            uncertainty: U = 4·π·(1−π), max at π=0.5
            sample_imbalance: B = |n_use/(n_use+n_base) − 0.5|
            danger_score: 0–1 where 1 = extreme danger

        Returns:
            q_base ∈ [q_min, q_max]
        """
        # Reward high uncertainty (we want to learn about unknown territory)
        u_term = 0.15 * uncertainty  # max contribution ~0.15

        # Reward sample imbalance (we want balanced n_use and n_base)
        b_term = 0.10 * sample_imbalance  # max contribution ~0.05

        # Penalize danger
        d_term = 0.25 * danger_score  # at danger=1, subtracts 0.25

        q = self.q_min + u_term + b_term - d_term

        # Self-correct towards target average
        if self._total_decisions > 0:
            current_rate = self._force_base_count / self._total_decisions
            if current_rate > self.target_avg + 0.05:
                q *= 0.7  # Reduce if over-sampling
            elif current_rate < self.target_avg - 0.05:
                q *= 1.3  # Increase if under-sampling

        return max(self.q_min, min(self.q_max, q))

    def should_force_base(self, uncertainty: float,
                          sample_imbalance: float = 0.0,
                          danger_score: float = 0.0,
                          risk_level: str = "medium") -> Tuple[bool, float]:
        """Decide whether to force base/fallback for this decision.

        Args:
            uncertainty: U = 4·π·(1−π)
            sample_imbalance: B
            danger_score: 0–1 danger level
            risk_level: "low" | "medium" | "high"

        Returns:
            (should_force, propensity): decision + probability for logging
        """
        # Never force base in high-risk scenarios
        if risk_level == "high" or danger_score > 0.7:
            return False, 0.0

        q = self.base_probability(uncertainty, sample_imbalance, danger_score)
        should_force = random.random() < q

        self._total_decisions += 1
        if should_force:
            self._force_base_count += 1

        return should_force, q

    @staticmethod
    def compute_uncertainty(pi_uplift: float) -> float:
        """U = 4·π·(1−π), the Bernoulli variance scaled to [0,1]."""
        return 4.0 * max(0.0, min(1.0, pi_uplift)) * (1.0 - max(0.0, min(1.0, pi_uplift)))

    @staticmethod
    def compute_imbalance(n_use: float, n_base: float) -> float:
        """B = |n_use/(n_use+n_base) − 0.5|, measures sample imbalance."""
        total = n_use + n_base
        if total < 1:
            return 0.5  # Maximum imbalance when no data
        return abs(n_use / total - 0.5)

    @staticmethod
    def compute_danger_score(state: Dict) -> float:
        """Compute danger score from state flags."""
        flags = sum(1 for f in HIGH_DANGER_FLAGS if state.get(f))
        return min(1.0, flags / len(HIGH_DANGER_FLAGS))

    def log_decision(self, decision_id: str, knowledge_id: str,
                     assigned_action: str, propensity_reuse: float,
                     propensity_base: float, context_bucket: str = ""):
        """Log a forced-base decision with propensity."""
        entry = {
            "decision_id": decision_id,
            "candidate": knowledge_id,
            "assigned": assigned_action,
            "propensity_reuse": propensity_reuse,
            "propensity_base": propensity_base,
            "context_bucket": context_bucket,
        }
        # Write to log file if path configured
        if self._log_path:
            import os
            os.makedirs(os.path.dirname(self._log_path), exist_ok=True)
            with open(self._log_path, "a") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return entry

    def stats(self) -> Dict:
        return {
            "total_decisions": self._total_decisions,
            "force_base_count": self._force_base_count,
            "current_rate": (self._force_base_count / max(self._total_decisions, 1)),
        }

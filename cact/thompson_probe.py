"""
Safe Thompson Probing — exploration for cold-start knowledge.

Cold-start problem: no data → gate won't approve reuse → never get data.

Solution: During accumulation/calibration phases, use Thompson sampling
to occasionally allow low-risk exploration of uncertain knowledge.
The probe rate adapts to:
  - Uncertainty (higher → more probing)
  - Effective sample size (below n_min → more probing)
  - Risk level (high → zero probing)

E3 strict frozen test: NO Thompson probing allowed.
"""

import random
import numpy as np
from typing import Dict, Tuple


# Probe rate bounds
Q_PROBE_MAX = 0.10
Q_PROBE_BASE = 0.15  # Base multiplier before uncertainty scaling

# Minimum effective sample sizes for probe eligibility
N_MIN_EASY = 5
N_MIN_MEDIUM = 8
N_MIN_HARD = 10
N_MIN_STRESS = 10   # interaction stress tasks

# Probe conditions
DELTA_PROBE = 0.03   # Minimum uplift effect for probing
EPS_PROBE = 0.05     # Maximum harm for probing


class SafeThompsonProber:
    """Safe Thompson exploration controller."""

    def __init__(self, q_probe_max: float = Q_PROBE_MAX,
                 probe_budget: int = 100):
        self.q_max = q_probe_max
        self._budget = probe_budget
        self._probe_count = 0

    def probe_probability(self, uncertainty: float,
                          ess: float,
                          risk_level: str = "medium") -> float:
        """Compute adaptive Thompson probe probability.

        q_probe = clip(q₀ · U · 𝟙[ESS < n_min] · 𝟙[Risk ≠ high], 0, q_max)

        Args:
            uncertainty: U = 4·π·(1−π)
            ess: Effective sample size
            risk_level: "low" | "medium" | "high"
        """
        # Hard stop: no probing in high risk
        if risk_level == "high":
            return 0.0

        # Budget check
        if self._probe_count >= self._budget:
            return 0.0

        # ESS gate: only probe if insufficient data
        n_min = {"low": N_MIN_EASY, "medium": N_MIN_MEDIUM,
                 "high": N_MIN_HARD}.get(risk_level, N_MIN_MEDIUM)
        if ess >= n_min:
            return 0.0

        # Uncertainty-scaled probability
        q = Q_PROBE_BASE * uncertainty
        return max(0.0, min(self.q_max, q))

    def should_probe(self, use_alpha: float, use_beta: float,
                     base_alpha: float, base_beta: float,
                     harm_alpha: float, harm_beta: float,
                     ess: float, risk_level: str = "medium",
                     force_allow: bool = False) -> Tuple[bool, float]:
        """Decide whether to Thompson-probe this knowledge.

        Checks 4 conditions:
          1. Thompson sample: p_use > p_base + δ_probe
          2. Thompson sample: p_harm < ε_probe
          3. Context is low/medium risk
          4. Global exploration budget not exceeded

        Returns:
            (should_probe, probe_probability)
        """
        if risk_level == "high":
            return False, 0.0

        q = self.probe_probability(
            uncertainty=4.0 * (use_alpha / max(use_alpha + use_beta, 1e-8)) *
                        (1.0 - use_alpha / max(use_alpha + use_beta, 1e-8)),
            ess=ess,
            risk_level=risk_level,
        )

        if q <= 0.0 and not force_allow:
            return False, 0.0

        # Thompson sample check
        rng = np.random.default_rng()
        p_use = float(rng.beta(use_alpha, use_beta))
        p_base = float(rng.beta(base_alpha, base_beta))
        p_harm = float(rng.beta(harm_alpha, harm_beta))

        conditions_met = (
            p_use > p_base + DELTA_PROBE and
            p_harm < EPS_PROBE and
            self._probe_count < self._budget
        )

        if force_allow or (conditions_met and random.random() < q):
            self._probe_count += 1
            return True, q

        return False, q

    def reset_budget(self, budget: int = None):
        """Reset probe budget (e.g., at start of new experiment phase)."""
        if budget is not None:
            self._budget = budget
        self._probe_count = 0

    def stats(self) -> Dict:
        return {
            "probe_count": self._probe_count,
            "budget_remaining": self._budget - self._probe_count,
        }

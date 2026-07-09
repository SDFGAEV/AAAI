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

    # ── Pre-registered probe triggers (doc §11.2, TABLE 92) ──
    @staticmethod
    def is_triggered(pi_uplift: float, tau_threshold: float,
                     harm_alpha: float, harm_beta: float,
                     n_base: float, min_base: int = 3,
                     ece: float = 0.0, precond_fail_rate: float = 0.0,
                     interaction_state: str = "safe") -> Tuple[bool, str]:
        """Check pre-registered trigger conditions for targeted admission probing.

        Returns (triggered, trigger_reason).
        Only one trigger needs to fire for probing to be considered.
        """
        # Trigger 1: π_u near threshold (within 0.05 of τ)
        if abs(pi_uplift - tau_threshold) < 0.05:
            return True, "pi_near_threshold"

        # Trigger 2: harm_ucb high uncertainty
        # Harm posterior Beta variance = αβ/((α+β)²(α+β+1))
        harm_total = harm_alpha + harm_beta
        if harm_total > 0:
            harm_var = (harm_alpha * harm_beta) / (harm_total * harm_total * (harm_total + 1))
            if harm_var > 0.01:  # High uncertainty threshold
                return True, "harm_uncertainty_high"

        # Trigger 3: base sample too sparse
        if n_base < min_base:
            return True, "base_sparse"

        # Trigger 4: context bucket ECE high
        if ece > 0.15:
            return True, "bucket_ece_high"

        # Trigger 5: contract precondition often fails
        if precond_fail_rate > 0.3:
            return True, "precond_fail_rate_high"

        # Trigger 6: interaction pair unknown
        if interaction_state == "unknown":
            return True, "interaction_unknown"

        return False, "none"

    def should_probe(self, use_alpha: float, use_beta: float,
                     base_alpha: float, base_beta: float,
                     harm_alpha: float, harm_beta: float,
                     ess: float, risk_level: str = "medium",
                     force_allow: bool = False,
                     pi_uplift: float = 0.5, tau_threshold: float = 0.90,
                     n_base: float = 0, ece: float = 0.0,
                     precond_fail_rate: float = 0.0,
                     interaction_state: str = "safe") -> Tuple[bool, float]:
        """Decide whether to Thompson-probe this knowledge.

        Checks:
          1. Pre-registered trigger fired (new — doc §11.2)
          2. Thompson sample: p_use > p_base + δ_probe
          3. Thompson sample: p_harm < ε_probe
          4. Context is low/medium risk
          5. Global exploration budget not exceeded

        Returns:
            (should_probe, probe_probability)
        """
        if risk_level == "high":
            return False, 0.0

        mean_use = use_alpha / max(use_alpha + use_beta, 1e-8)
        uncertainty = 4.0 * mean_use * (1.0 - mean_use)
        q = self.probe_probability(uncertainty=uncertainty, ess=ess,
                                   risk_level=risk_level)

        if q <= 0.0 and not force_allow:
            return False, 0.0

        # Pre-registered trigger check (doc §11.2): at least one trigger must fire
        if not force_allow:
            triggered, reason = self.is_triggered(
                pi_uplift, tau_threshold, harm_alpha, harm_beta,
                n_base, ece=ece, precond_fail_rate=precond_fail_rate,
                interaction_state=interaction_state)
            if not triggered:
                return False, q

        # Thompson sample check
        p_use = float(np.random.beta(use_alpha, use_beta))
        p_base = float(np.random.beta(base_alpha, base_beta))
        p_harm = float(np.random.beta(harm_alpha, harm_beta))

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

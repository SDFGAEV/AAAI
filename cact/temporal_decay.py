"""
Drift-aware lazy temporal decay for C-ACT TrustStore.

Core idea: old evidence should gradually decay towards the prior,
but the decay rate should adapt to whether the environment is
changing (drift) or stable.

Formula:
  α_t = α_0 + ρ^{Δt} · (α_old − α_0)
  β_t = β_0 + ρ^{Δt} · (β_old − β_0)

ρ adapts based on prediction error:
  - High error → lower ρ → faster decay (environment changing)
  - Low error  → higher ρ → slower decay (stable environment)

ρ ∈ [ρ_min, ρ_max], default ρ = 0.95
"""

from typing import Tuple


# Bounds for adaptive decay factor
RHO_MIN = 0.85
RHO_MAX = 0.99
RHO_DEFAULT = 0.95

# Bounds for effective prior strength
EFFECTIVE_STRENGTH_MIN = 0.1
EFFECTIVE_STRENGTH_MAX = 20.0


class TemporalDecay:
    """Drift-aware lazy temporal decay controller."""

    def __init__(self, rho: float = RHO_DEFAULT,
                 rho_min: float = RHO_MIN,
                 rho_max: float = RHO_MAX):
        self.rho = max(rho_min, min(rho_max, rho))
        self.rho_min = rho_min
        self.rho_max = rho_max
        self._recent_errors: list = []
        self._error_window = 10

    def adapt(self, prediction_error: float):
        """Update decay rate based on recent prediction error.

        Larger error → decrease ρ (faster decay).
        """
        self._recent_errors.append(prediction_error)
        if len(self._recent_errors) > self._error_window:
            self._recent_errors.pop(0)

        if len(self._recent_errors) >= 3:
            avg_error = sum(self._recent_errors) / len(self._recent_errors)
            if avg_error > 0.3:
                self.rho = max(self.rho_min, self.rho - 0.03)
            elif avg_error < 0.1:
                self.rho = min(self.rho_max, self.rho + 0.01)

    def decay_params(self, alpha_old: float, beta_old: float,
                     alpha_0: float, beta_0: float,
                     delta_t: int = 1) -> Tuple[float, float]:
        """Apply temporal decay to Beta parameters.

        Args:
            alpha_old, beta_old: Current posterior parameters
            alpha_0, beta_0: Prior parameters (from empirical Bayes)
            delta_t: Number of time units since last decay

        Returns:
            (alpha_new, beta_new): Decayed posterior parameters
        """
        factor = self.rho ** delta_t
        alpha_new = alpha_0 + factor * (alpha_old - alpha_0)
        beta_new = beta_0 + factor * (beta_old - beta_0)
        # Clip to valid range
        alpha_new = max(EFFECTIVE_STRENGTH_MIN, min(EFFECTIVE_STRENGTH_MAX, alpha_new))
        beta_new = max(EFFECTIVE_STRENGTH_MIN, min(EFFECTIVE_STRENGTH_MAX, beta_new))
        return alpha_new, beta_new

    def get_drift_factor(self) -> float:
        """Return 1/ρ as drift multiplier (for backwards compatibility)."""
        return 1.0 / max(self.rho, 0.01)

    def reset(self):
        """Reset to default decay rate."""
        self.rho = RHO_DEFAULT
        self._recent_errors = []

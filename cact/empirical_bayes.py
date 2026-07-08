"""
Empirical Bayes priors for level-aware trust estimation.

For each (knowledge_type, knowledge_level), estimate a Beta prior
from E1/E2 history using method-of-moments with shrinkage towards
a global prior. This avoids hand-tuned priors and allows the model
to adapt to the actual distribution of knowledge effectiveness.

Key formula:
  Prior_{z,l} = w_{z,l} * Prior_{z,l}^{emp} + (1-w_{z,l}) * Prior_{global}
  w_{z,l} = n_{z,l} / (n_{z,l} + k),  k = 20
"""

import numpy as np
from typing import Dict, List, Tuple, Optional


# Default global priors for each stat
DEFAULT_PRIORS = {
    "use":  (0.8, 1.2),
    "base": (0.6, 1.4),
    "harm": (0.1, 2.0),
}

# Shrinkage constant: larger = more conservative, slower to trust type-specific data
K_SHRINK = 20

# Clip bounds for estimated priors (prevents degenerate posteriors)
ALPHA_MIN, ALPHA_MAX = 0.1, 20.0
BETA_MIN, BETA_MAX = 0.1, 20.0


class EmpiricalBayes:
    """Level-aware empirical Bayes prior estimator."""

    def __init__(self, k_shrink: float = K_SHRINK,
                 default_priors: Dict[str, Tuple[float, float]] = None):
        self.k = k_shrink
        self.defaults = default_priors or DEFAULT_PRIORS
        # {type_level_key: {stat: (alpha, beta, n_samples)}}
        self._type_stats: Dict[str, Dict[str, Tuple[float, float, float]]] = {}

    def estimate(self, type_data: Dict[str, List[Dict]]):
        """Estimate per-type priors from historical data.

        Args:
            type_data: {type_level_key: [{stat, success, ...}, ...]}
                where type_level_key is e.g. "action_correction|atomic"
        """
        for key, records in type_data.items():
            by_stat = {"use": [], "base": [], "harm": []}
            for r in records:
                st = r.get("stat", "use")
                if st in by_stat:
                    val = r.get("success", r.get("outcome", 0.5))
                    by_stat[st].append(float(val))

            for st, vals in by_stat.items():
                if len(vals) < 5:
                    continue
                m = float(np.mean(vals))
                v = max(float(np.var(vals)), 1e-6)
                # Method of moments: m = a/(a+b), v = ab/((a+b)^2(a+b+1))
                raw_s = m * (1 - m) / v - 1
                s = max(0.1, min(raw_s, 20.0))
                a0 = max(ALPHA_MIN, min(m * s, ALPHA_MAX))
                b0 = max(BETA_MIN, min((1 - m) * s, BETA_MAX))
                if key not in self._type_stats:
                    self._type_stats[key] = {}
                self._type_stats[key][st] = (a0, b0, float(len(vals)))

    def get_prior(self, knowledge_type: str, knowledge_level: str = "",
                  stat: str = "use") -> Tuple[float, float]:
        """Get level-aware empirical Bayes prior for a (type, level, stat) triplet.

        Uses shrinkage: Prior = w * type_empirical + (1-w) * global_default
        w = n / (n + k) — weight grows with sample count.
        """
        a0, b0 = self.defaults.get(stat, (1.0, 1.0))
        key = f"{knowledge_type}|{knowledge_level}" if knowledge_level else knowledge_type

        if key in self._type_stats and stat in self._type_stats[key]:
            t_a, t_b, t_n = self._type_stats[key][stat]
            w = min(t_n / (t_n + self.k), 0.8) if t_n > 0 else 0.0
            a0 = (1 - w) * a0 + w * t_a
            b0 = (1 - w) * b0 + w * t_b

        return a0, b0

    def get_all_type_info(self) -> Dict:
        """Return all stored type statistics (for serialization)."""
        return {
            key: {st: {"alpha": a, "beta": b, "n": n}
                  for st, (a, b, n) in stats.items()}
            for key, stats in self._type_stats.items()
        }

    def to_dict(self) -> Dict:
        return {"k": self.k, "type_stats": self.get_all_type_info()}

    @classmethod
    def from_dict(cls, d: Dict) -> "EmpiricalBayes":
        eb = cls(k_shrink=d.get("k", K_SHRINK))
        for key, stats in d.get("type_stats", {}).items():
            eb._type_stats[key] = {}
            for st, info in stats.items():
                eb._type_stats[key][st] = (info["alpha"], info["beta"], info["n"])
        return eb

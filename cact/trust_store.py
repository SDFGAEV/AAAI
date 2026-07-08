"""
C-ACT TrustStore: Bayesian 3-Beta posterior store with lifecycle.

Each knowledge (kid, context) maintains three Beta posteriors:
  use  — P(Y=1 | do(u), c)
  base — P(Y=1 | do(∅), c)
  harm — P(H=1 | do(u), c)

Enhancements over CASK TrustStore:
  - Integrated with KnowledgeContract (contract field per entry)
  - Delegates to LifecycleManager for state transitions
  - Delegates to TemporalDecay for adaptive ρ decay
  - Delegates to EmpiricalBayes for level-aware priors
  - Pairwise interaction joint tracking
"""

import json, os, math
from typing import Dict, List, Tuple, Optional
from scipy.stats import beta as beta_dist
from scipy.special import betaln
import numpy as np

from .lifecycle_manager import (LifecycleManager, LifecycleState,
                                 CANDIDATE, PROBATION, CERTIFIED,
                                 DEPRECATED, DISABLED, QUARANTINED)
from .temporal_decay import TemporalDecay
from .empirical_bayes import EmpiricalBayes


class TrustStore:
    """Bayesian counterfactual certificate store with lifecycle governance."""

    def __init__(self, store_path: str = None, prior_strength: float = 1.0):
        self.store_path = store_path or os.path.join(
            os.path.dirname(__file__), "..", "ckpt", "cask_cert"
        )
        os.makedirs(self.store_path, exist_ok=True)
        self._db_file = os.path.join(self.store_path, "cert_db.json")
        self._lifecycle_file = os.path.join(self.store_path, "lifecycle.json")

        self._data: Dict[str, Dict] = {}
        self._contracts: Dict[str, Dict] = {}          # {kid: contract_dict}
        self._prior_strength = prior_strength

        # Sub-modules
        self.lifecycle = LifecycleManager(self.store_path)
        self.decay = TemporalDecay()
        self.empirical_bayes = EmpiricalBayes()

        self._decay_events = 0
        self._load()

    # ── I/O ──
    def _load(self):
        if os.path.exists(self._db_file):
            try: self._data = json.load(open(self._db_file))
            except: self._data = {}
        # Load contracts if stored alongside
        contract_file = os.path.join(self.store_path, "contracts.json")
        if os.path.exists(contract_file):
            try: self._contracts = json.load(open(contract_file))
            except: self._contracts = {}

    def _save(self):
        json.dump(self._data, open(self._db_file, "w"), indent=2)
        # Also save contracts
        contract_file = os.path.join(self.store_path, "contracts.json")
        json.dump(self._contracts, open(contract_file, "w"), indent=2)

    def _make_key(self, kid: str, context: str, stat: str) -> str:
        return f"{kid}|{context}|{stat}"

    # ── Contract management ──
    def register_contract(self, kid: str, contract_dict: Dict):
        """Register a KnowledgeContract for this knowledge item."""
        self._contracts[kid] = contract_dict
        self.lifecycle._states[kid] = CANDIDATE
        self.lifecycle._save()
        self._save()

    def get_contract(self, kid: str) -> Optional[Dict]:
        return self._contracts.get(kid)

    def get_knowledge_type(self, kid: str) -> str:
        c = self._contracts.get(kid, {})
        return c.get("type", "")

    def get_knowledge_level(self, kid: str) -> str:
        c = self._contracts.get(kid, {})
        return c.get("level", "")

    def get_type_level_key(self, kid: str) -> str:
        c = self._contracts.get(kid, {})
        return f"{c.get('type', 'unknown')}|{c.get('level', 'unknown')}"

    # ── Priors ──
    DEFAULT_PRIORS = {
        "use":  (0.8, 1.2), "base": (0.6, 1.4), "harm": (0.1, 2.0),
        "joint": (1.0, 1.0),
    }

    def _effective_prior(self, stat: str, kid: str = None) -> Tuple[float, float]:
        """Combine global prior with level-aware empirical Bayes estimates."""
        a0, b0 = self.DEFAULT_PRIORS.get(stat, (1.0, 1.0))
        s = self._prior_strength

        if kid and kid in self._contracts:
            ktype = self.get_knowledge_type(kid)
            klevel = self.get_knowledge_level(kid)
            a0, b0 = self.empirical_bayes.get_prior(ktype, klevel, stat)

        return 1.0 + s * (a0 - 1.0), 1.0 + s * (b0 - 1.0)

    def _get(self, kid: str, context: str, stat: str) -> Dict:
        k = self._make_key(kid, context, stat)
        if k not in self._data:
            a0, b0 = self._effective_prior(stat, kid)
            self._data[k] = {"alpha": a0, "beta": b0}
        return self._data[k]

    def get_stats(self, kid: str, context: str, stat: str = "use") -> Tuple[float, float]:
        d = self._get(kid, context, stat)
        return d["alpha"], d["beta"]

    # ── Recording ──
    def record_use(self, kid: str, context: str, success: float):
        d = self._get(kid, context, "use")
        d["alpha"] += success
        d["beta"] += (1.0 - success)
        self._save()

    def record_base(self, kid: str, context: str, success: float):
        d = self._get(kid, context, "base")
        d["alpha"] += success
        d["beta"] += (1.0 - success)
        self._save()

    def record_harm(self, kid: str, context: str, harmful: float):
        d = self._get(kid, context, "harm")
        d["alpha"] += harmful
        d["beta"] += (1.0 - harmful)
        self._save()

    def record_episode(self, kid: str, context: str, used: bool = True,
                       success: float = 0.0, is_harmful: float = 0.0):
        """Main recording method called after each knowledge use/fallback decision."""
        if used:
            self.record_use(kid, context, success)
            self.record_harm(kid, context, is_harmful)
        else:
            self.record_base(kid, context, success)

        # Auto lifecycle transitions
        n = self.total_count(kid, context, "use")
        pi = self.uplift_probability(kid, context)
        hu = self.harm_upper_bound(kid, context)
        tau = 0.88  # Default, overridden by config in practice
        h_star = 0.10

        new_state = self.lifecycle.evaluate_auto_transition(
            kid, pi, hu, tau, h_star, int(n))
        if new_state:
            self.lifecycle.transition(kid, new_state, "auto_after_observation")

    # ── Queries ──
    def mean(self, kid: str, context: str, stat: str = "use") -> float:
        a, b = self.get_stats(kid, context, stat)
        total = a + b
        return a / max(total, 1e-8)

    def lcb(self, kid: str, context: str, stat: str = "use",
            delta: float = 0.05) -> float:
        a, b = self.get_stats(kid, context, stat)
        return float(beta_dist.ppf(delta, a, b))

    def ucb(self, kid: str, context: str, stat: str = "use",
            delta: float = 0.05) -> float:
        a, b = self.get_stats(kid, context, stat)
        return float(beta_dist.ppf(1 - delta, a, b))

    def total_count(self, kid: str, context: str, stat: str = "use") -> float:
        a, b = self.get_stats(kid, context, stat)
        a0, b0 = self._effective_prior(stat, kid)
        return a + b - a0 - b0

    def ess(self, kid: str, context: str) -> float:
        return (self.total_count(kid, context, "use") +
                self.total_count(kid, context, "base") +
                self.total_count(kid, context, "harm"))

    # ── Bayesian counterfactual uplift (C-ACT core) ──
    def uplift_probability(self, kid: str, context: str) -> float:
        """π_u(c) = P(p_use > p_base | data) — exact posterior probability."""
        a1, b1 = self.get_stats(kid, context, "use")
        a2, b2 = self.get_stats(kid, context, "base")
        total = 0.0
        for i in range(int(a2)):
            term = (betaln(a1 + i, b1 + b2) - math.log(b2 + i)
                    - betaln(1 + i, b2) - betaln(a1, b1))
            total += math.exp(term)
        return min(total, 1.0 - 1e-8)

    def uplift_lcb(self, kid: str, context: str, delta: float = 0.05) -> float:
        """Conservative uplift: LCB[use] - UCB[base]"""
        return self.lcb(kid, context, "use", delta) - \
               self.ucb(kid, context, "base", delta)

    def harm_upper_bound(self, kid: str, context: str, q: float = 0.95,
                          delta: float = 0.05) -> float:
        """h⁺(c) = Q_{0.95}[p_harm] — 95% posterior upper bound."""
        return self.ucb(kid, context, "harm", 1 - q)

    def prob_harm_safe(self, kid: str, context: str, h_max: float = 0.10) -> float:
        """P(p_harm ≤ h_max | data)"""
        a, b = self.get_stats(kid, context, "harm")
        return float(beta_dist.cdf(h_max, a, b))

    def thompson_sample(self, kid: str, context: str) -> Tuple[float, float, float]:
        """Draw from posterior: (p_use, p_base, p_harm)"""
        a1, b1 = self.get_stats(kid, context, "use")
        a2, b2 = self.get_stats(kid, context, "base")
        a3, b3 = self.get_stats(kid, context, "harm")
        rng = np.random.default_rng()
        return (float(rng.beta(a1, b1)), float(rng.beta(a2, b2)),
                float(rng.beta(a3, b3)))

    # ── Temporal decay (delegates to TemporalDecay) ──
    def decay_all(self):
        """Apply adaptive temporal decay to all posteriors."""
        self._decay_events += 1
        for k in list(self._data.keys()):
            stat = k.rsplit("|", 1)[-1]
            kid = k.split("|", 0)[0] if "|" in k else ""
            a0, b0 = self._effective_prior(stat, kid)
            d = self._data[k]
            d["alpha"], d["beta"] = self.decay.decay_params(
                d["alpha"], d["beta"], a0, b0)
        self._save()

    def adapt_decay(self, prediction_error: float):
        self.decay.adapt(prediction_error)

    # ── Interaction (for interaction_gate) ──
    def record_joint(self, kid_i: str, kid_j: str, context: str,
                     success: float, harmful: float = 0.0):
        pair_key = f"pair:{kid_i}:{kid_j}"
        d = self._get(pair_key, context, "joint")
        d["alpha"] += success
        d["beta"] += (1.0 - success)
        self._save()

    def get_joint_stats(self, kid_i: str, kid_j: str, context: str) -> Dict:
        pair_key = f"pair:{kid_i}:{kid_j}"
        a, b = self.get_stats(pair_key, context, "joint")
        return {"alpha": a, "beta": b}

    def get_all_pair_stats(self, kid_list: List[str], context: str) -> Dict:
        """Get all pairwise joint stats for interaction_gate."""
        result = {}
        for i in range(len(kid_list)):
            for j in range(i + 1, len(kid_list)):
                stats = self.get_joint_stats(kid_list[i], kid_list[j], context)
                result[(kid_list[i], kid_list[j])] = stats
        return result

    # ── Knowledge lifecycle (delegates to LifecycleManager) ──
    def get_lifecycle_state(self, kid: str) -> str:
        return self.lifecycle.get_state(kid).value

    def is_reusable(self, kid: str) -> bool:
        return self.lifecycle.is_reusable(kid)

    def get_active_knowledge(self) -> List[str]:
        return self.lifecycle.active_knowledge_ids()

    def lifecycle_stats(self) -> Dict[str, int]:
        return self.lifecycle.stats()

    # ── Persistence helpers ──
    def export_for_calibration(self) -> Dict:
        """Export data needed for E2 calibration."""
        data = {"_data": self._data, "_contracts": self._contracts}
        return data

    def import_from_calibration(self, data: Dict):
        """Import calibration results (adaptive priors, thresholds)."""
        if "_data" in data:
            self._data.update(data["_data"])
        if "_contracts" in data:
            self._contracts.update(data["_contracts"])
        self._save()

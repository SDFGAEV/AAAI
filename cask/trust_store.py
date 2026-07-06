"""
TrustStore: Bayesian counterfactual certificate store (ACT-RL v2)

Each knowledge (kid, context) maintains three Beta posteriors:
  use  — P(Y=1 | do(u), c)
  base — P(Y=1 | do(∅), c)
  harm — P(H=1 | do(u), c)

Knowledge lifecycle: Candidate → Probation → Certified → Deprecated → Disabled

Features:
  - Type-level empirical Bayes priors (from E1/E2 history)
  - Drift-aware lazy temporal decay
  - Interaction joint outcome tracking
  - Knowledge governance state machine
"""

import json, os, math
from typing import Dict, List, Tuple, Optional
from scipy.stats import beta as beta_dist
from scipy.special import betaln
import numpy as np

# ── Knowledge lifecycle states ──
CANDIDATE  = "candidate"
PROBATION  = "probation"
CERTIFIED  = "certified"
DEPRECATED = "deprecated"
DISABLED   = "disabled"


class TrustStore:
    def __init__(self, store_path: str = None, prior_strength: float = 1.0):
        self.store_path = store_path or os.path.join(
            os.path.dirname(__file__), "..", "ckpt", "cask_cert"
        )
        os.makedirs(self.store_path, exist_ok=True)
        self._db_file = os.path.join(self.store_path, "cert_db.json")
        self._lifecycle_file = os.path.join(self.store_path, "lifecycle.json")
        self._data: Dict[str, Dict] = {}
        self._lifecycle: Dict[str, str] = {}
        self._type_stats: Dict[str, Dict] = {}  # {type: {stat: (α,β)}}
        self._prior_strength = prior_strength
        self._decay_events = 0
        self._load()

    # ── I/O ──
    def _load(self):
        if os.path.exists(self._db_file):
            try: self._data = json.load(open(self._db_file))
            except: self._data = {}
        if os.path.exists(self._lifecycle_file):
            try: self._lifecycle = json.load(open(self._lifecycle_file))
            except: self._lifecycle = {}

    def _save(self):
        json.dump(self._data, open(self._db_file, "w"), indent=2)

    def _save_lifecycle(self):
        json.dump(self._lifecycle, open(self._lifecycle_file, "w"), indent=2)

    def _make_key(self, kid: str, context: str, stat: str) -> str:
        return f"{kid}|{context}|{stat}"

    # ── Priors ──
    DEFAULT_PRIORS = {
        "use":  (0.8, 1.2), "base": (0.6, 1.4), "harm": (0.1, 2.0),
        "joint": (1.0, 1.0),
    }

    def _effective_prior(self, stat: str, knowledge_type: str = None) -> Tuple[float, float]:
        """Combine global prior with type-level empirical Bayes estimates."""
        a0, b0 = self.DEFAULT_PRIORS.get(stat, (1.0, 1.0))
        s = self._prior_strength
        if knowledge_type and knowledge_type in self._type_stats:
            ts = self._type_stats[knowledge_type]
            if stat in ts:
                t_a, t_b, t_n = ts[stat]
                # Weight: type prior influence grows with sample count
                w = min(t_n / (t_n + 5.0), 0.8) if t_n > 0 else 0.0
                a0 = (1 - w) * a0 + w * t_a
                b0 = (1 - w) * b0 + w * t_b
        return 1.0 + s * (a0 - 1.0), 1.0 + s * (b0 - 1.0)

    def _get(self, kid: str, context: str, stat: str,
             knowledge_type: str = None) -> Dict:
        k = self._make_key(kid, context, stat)
        if k not in self._data:
            a0, b0 = self._effective_prior(stat, knowledge_type)
            self._data[k] = {"alpha": a0, "beta": b0}
        return self._data[k]

    def get_stats(self, kid: str, context: str, stat: str = "use") -> Tuple[float, float]:
        d = self._get(kid, context, stat)
        return d["alpha"], d["beta"]

    # ── Recording ──
    def record_use(self, kid: str, context: str, success: float,
                   knowledge_type: str = None):
        d = self._get(kid, context, "use", knowledge_type)
        d["alpha"] += success; d["beta"] += (1.0 - success); self._save()

    def record_base(self, kid: str, context: str, success: float,
                    knowledge_type: str = None):
        d = self._get(kid, context, "base", knowledge_type)
        d["alpha"] += success; d["beta"] += (1.0 - success); self._save()

    def record_harm(self, kid: str, context: str, harmful: float,
                    knowledge_type: str = None):
        d = self._get(kid, context, "harm", knowledge_type)
        d["alpha"] += harmful; d["beta"] += (1.0 - harmful); self._save()

    def record_episode(self, kid: str, context: str, used: bool = True,
                       success: float = 0.0, is_harmful: float = 0.0,
                       knowledge_type: str = None):
        self.record_use(kid, context, success if used else 0.5, knowledge_type)
        if not used:
            self.record_base(kid, context, success, knowledge_type)
        if used:
            self.record_harm(kid, context, is_harmful, knowledge_type)
        # Promote probation → certified when evidence strong
        lc = self._lifecycle.get(kid, CANDIDATE)
        n = self.total_count(kid, context, "use")
        if lc == CANDIDATE and n >= 1:
            self._lifecycle[kid] = PROBATION; self._save_lifecycle()
        if lc == PROBATION and n >= 3 and self.lcb(kid, context, "use") > 0.3:
            self._lifecycle[kid] = CERTIFIED; self._save_lifecycle()

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
        return a + b - self._effective_prior(stat)[0] - self._effective_prior(stat)[1] + 2.0

    def ess(self, kid: str, context: str) -> float:
        """Effective sample size: use + base + harm observations."""
        return (self.total_count(kid, context, "use") +
                self.total_count(kid, context, "base") +
                self.total_count(kid, context, "harm"))

    # ── Bayesian uplift ──
    def prob_use_better(self, kid: str, context: str) -> float:
        """P(p_use > p_base | data) — exact posterior probability."""
        a1, b1 = self.get_stats(kid, context, "use")
        a2, b2 = self.get_stats(kid, context, "base")
        total = 0.0
        for i in range(int(a2)):
            term = (betaln(a1 + i, b1 + b2) - math.log(b2 + i)
                    - betaln(1 + i, b2) - betaln(a1, b1))
            total += math.exp(term)
        return min(total, 1.0 - 1e-8)

    def uplift(self, kid: str, context: str, delta: float = 0.05) -> float:
        """Conservative uplift: LCB[use] - UCB[base]"""
        return self.lcb(kid, context, "use", delta) - self.ucb(kid, context, "base", delta)

    def harm_ucb(self, kid: str, context: str, delta: float = 0.05) -> float:
        return self.ucb(kid, context, "harm", delta)

    def prob_harm_safe(self, kid: str, context: str, h_max: float = 0.10) -> float:
        a, b = self.get_stats(kid, context, "harm")
        return float(beta_dist.cdf(h_max, a, b))

    def thompson_sample(self, kid: str, context: str) -> Tuple[float, float, float]:
        a1, b1 = self.get_stats(kid, context, "use")
        a2, b2 = self.get_stats(kid, context, "base")
        a3, b3 = self.get_stats(kid, context, "harm")
        rng = np.random.default_rng()
        return (float(rng.beta(a1, b1)), float(rng.beta(a2, b2)),
                float(rng.beta(a3, b3)))

    # ── Empirical Bayes priors ──
    def estimate_type_priors(self, type_data: Dict[str, List[Dict]]):
        """Estimate per-type priors from E1/E2 history.
        type_data: {type: [{stat, success, ...}, ...]}
        Uses method-of-moments with clipping.
        """
        for ktype, records in type_data.items():
            by_stat = {"use": [], "base": [], "harm": []}
            for r in records:
                st = r.get("stat", "use")
                if st in by_stat:
                    by_stat[st].append(r.get("success", r.get("outcome", 0.5)))
            for st, vals in by_stat.items():
                if len(vals) < 5:
                    continue
                m = np.mean(vals); v = max(np.var(vals), 1e-6)
                raw_s = m * (1 - m) / v - 1
                s = max(0.1, min(raw_s, 20.0))
                a0 = max(0.1, min(m * s, 20.0))
                b0 = max(0.1, min((1 - m) * s, 20.0))
                if ktype not in self._type_stats:
                    self._type_stats[ktype] = {}
                self._type_stats[ktype][st] = (a0, b0, float(len(vals)))

    def get_type_prior(self, knowledge_type: str, stat: str) -> Tuple[float, float]:
        if knowledge_type in self._type_stats and stat in self._type_stats[knowledge_type]:
            a, b, _ = self._type_stats[knowledge_type][stat]
            return a, b
        return self.DEFAULT_PRIORS.get(stat, (1.0, 1.0))

    # ── Temporal decay (drift-aware) ──
    def decay_all(self, retention: float = 0.95, drift_factor: float = 1.0):
        """Rescale posteriors towards priors. Lower retention = faster forgetting.
        drift_factor > 1.0 means increased drift → shorter half-life."""
        self._decay_events += 1
        rho = max(0.85, min(0.99, retention ** drift_factor))
        for k in list(self._data.keys()):
            stat = k.rsplit("|", 1)[-1]
            # Infer knowledge type from key (best effort)
            ktype = None
            for tk in ["skill", "remedy", "action_correction", "dependency"]:
                if tk in k: ktype = tk; break
            a0, b0 = self._effective_prior(stat, ktype)
            d = self._data[k]
            d["alpha"] = rho * d["alpha"] + (1 - rho) * a0
            d["beta"] = rho * d["beta"] + (1 - rho) * b0
        self._save()

    def detect_drift(self, recent_errors: List[float], healthy_error: float) -> float:
        """Drift factor: >1 if prediction errors increased."""
        if len(recent_errors) < 5: return 1.0
        mu = np.mean(recent_errors); ref = max(healthy_error, 0.001)
        return min(2.0, max(0.5, mu / ref))

    # ── Interaction ──
    def record_joint(self, kid_i: str, kid_j: str, context: str,
                     success: float, harmful: float = 0.0):
        pair_key = f"pair:{kid_i}:{kid_j}"
        d = self._get(pair_key, context, "joint")
        d["alpha"] += success; d["beta"] += (1.0 - success)
        self._save()

    def interaction_uplift(self, kid_i: str, kid_j: str, context: str,
                           delta: float = 0.05) -> tuple:
        """Interaction effect with 4-state classification."""
        a_j, b_j = self.get_stats(f"pair:{kid_i}:{kid_j}", context, "joint")
        n_joint = a_j + b_j - 2.0
        up_i = self.uplift(kid_i, context, delta)
        up_j = self.uplift(kid_j, context, delta)
        if n_joint < 2:
            return (0.0, 0.0, False, False, "unknown")
        lcb_joint = float(beta_dist.ppf(delta, a_j, b_j))
        ucb_base = self.ucb(kid_i, context, "base", delta)
        up_ij = lcb_joint - ucb_base
        delta_mean = up_ij - up_i - up_j
        delta_lcb = (float(beta_dist.ppf(delta/2, a_j, b_j)) - ucb_base) - up_i - up_j
        if delta_lcb > 0.03:
            state = "synergy"
        elif delta_lcb < -0.05:
            state = "conflict"
        elif n_joint >= 5 and abs(delta_lcb) <= 0.03:
            state = "neutral"
        else:
            state = "unknown"
        return (round(delta_mean, 4), round(delta_lcb, 4),
                state == "synergy", state == "conflict", state)

    # ── Knowledge lifecycle ──
    def get_lifecycle(self, kid: str) -> str:
        return self._lifecycle.get(kid, CANDIDATE)

    def set_lifecycle(self, kid: str, state: str):
        valid = {CANDIDATE, PROBATION, CERTIFIED, DEPRECATED, DISABLED}
        if state in valid:
            self._lifecycle[kid] = state; self._save_lifecycle()

    def lifecycle_stats(self) -> Dict[str, int]:
        counts = {s: 0 for s in [CANDIDATE, PROBATION, CERTIFIED, DEPRECATED, DISABLED]}
        for st in self._lifecycle.values():
            if st in counts: counts[st] += 1
        return counts

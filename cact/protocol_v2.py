"""C-ACT v2 protocol core.

This module implements the protocol in the complete research manual. It
separates applicability, randomized opportunity logging, episode-clustered
cross-fitted AIPW estimation, and selective policy calibration.
"""
from __future__ import annotations
import hashlib, json, math, os, random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import numpy as np

SCHEMA_VERSION = "cact.v2"
MAIN_METHODS = ("NoKnowledge", "NoGate", "FixedBayes",
                "PairwisePreferenceGate", "C-ACT-Pointwise", "C-ACT")
ONLINE_MAIN_METHODS = ("Online-NoGate", "Online-FixedBayes",
                       "Online-C-ACT-Pointwise", "Online-C-ACT")
E4_ABLATION_METHODS = ("C-ACT", "C-ACT-IndependentMargins", "C-ACT-HardBackoff", "C-ACT-MyopicReserve")
# Note: Global-Risk Only and w/o Ledger are controller-family/kappa variants
# selected at calibration time, not separate method names.
TRAIN_CALIB_METHODS = ("C-ACT",)
LEGACY_METHODS = ("XENON-Original", "BankCuration", "LifecycleSuccessGate",
                  "SuccessLifecycle", "ACT", "C-ACT-Full", "OracleGate",
                  "ShuffledKnowledge")
METHOD_ALIASES = {
    "C-ACT-Full": "C-ACT",
    "SuccessLifecycle": "LifecycleSuccessGate",
}
DEFAULT_EPS_ABS = 0.10
DEFAULT_EPS_INC = 0.02
DEFAULT_DELTA = 0.05
DEFAULT_BUDGET = 0.05
DEFAULT_KAPPAS = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
# CAP uses one conservatism parameter; kappa remains readable for legacy artifacts.
DEFAULT_LAMBDAS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
DEFAULT_CAP_ALPHA = 0.05
# Matches the preregistered CAP algorithm in aaai-paper/main.tex.
# Keep this frozen across calibration and deployment for reproducible certificates.
DEFAULT_CAP_EPSILON = 1e-6
DEFAULT_FUTURE_OPPORTUNITY_CLIP = 6
EVIDENCE_NUS = (12.0, 24.0, 48.0)
DEFAULT_EVIDENCE_NU = 24.0
MIN_ARM_SUPPORT = 12
MIN_ESS = 24.0
K_COLLECT = 4  # §6.1/§9.7: max randomized exposure per episode

def canonical_method_name(method: str) -> str:
    return METHOD_ALIASES.get(str(method), str(method))

def validate_method_name(method: str, allowed: Sequence[str] = None,
                         allow_legacy: bool = True) -> str:
    """Validate C-ACT method name. When `allowed` is given, only those methods pass."""
    canonical = canonical_method_name(method)
    if allowed is not None:
        allowed_set = {canonical_method_name(x) for x in allowed}
        if canonical not in allowed_set:
            raise ValueError(f"unsupported C-ACT method: {method}")
        return canonical
    allowed_set = set(MAIN_METHODS)
    if allow_legacy:
        allowed_set.update(LEGACY_METHODS)
        allowed_set.update(ONLINE_MAIN_METHODS)
        allowed_set.update(E4_ABLATION_METHODS)
        allowed_set.update(canonical_method_name(x) for x in LEGACY_METHODS)
        if canonical.startswith("C-ACT-") or canonical.startswith("Online-C-ACT-"):
            return canonical
    if canonical not in allowed_set:
        raise ValueError(f"unsupported C-ACT method: {method}")
    return canonical

def _stable_hash(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "big")

def _clip_propensity(value: float) -> float:
    return float(min(max(float(value), 0.2), 0.8))

@dataclass
class ApplicabilitySpec:
    """Applicability only; it is not a value or harm model."""
    knowledge_id: str
    source: str = ""
    type: str = ""
    scope: Dict[str, Any] = field(default_factory=dict)
    preconditions: List[str] = field(default_factory=list)
    hard_non_applicable: List[str] = field(default_factory=list)
    postconditions: List[str] = field(default_factory=list)
    provenance: Dict[str, Any] = field(default_factory=dict)
    raw_text: str = ""

    def evaluate(self, state: Mapping[str, Any], context: Mapping[str, Any]) -> Tuple[bool, str]:
        for key, expected in self.scope.items():
            if expected is not None and context.get(key) != expected:
                return False, "scope_mismatch"
        for condition in self.preconditions:
            if not _eval_condition(condition, state):
                return False, "precondition_failed"
        for boundary in self.hard_non_applicable:
            if _eval_flag(boundary, state, context):
                return False, "hard_boundary"
        return True, "applicable"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ApplicabilitySpec":
        return cls(
            knowledge_id=str(data.get("knowledge_id", data.get("id", ""))),
            source=str(data.get("source", "")),
            type=str(data.get("type", "")),
            scope=dict(data.get("scope", {}) or {}),
            preconditions=list(data.get("preconditions", []) or []),
            hard_non_applicable=list(data.get("hard_non_applicable",
                                             data.get("hard_non_applicable_contexts", [])) or []),
            postconditions=list(data.get("postconditions", []) or []),
            provenance=dict(data.get("provenance", {}) or {}),
            raw_text=str(data.get("raw_text", data.get("full_text", ""))),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

def _eval_condition(condition: str, state: Mapping[str, Any]) -> bool:
    cond = str(condition).strip()
    if " not in " in cond:
        key, vals = cond.split(" not in ", 1)
        return str(state.get(key.strip(), "")) not in {v.strip() for v in vals.strip(" {}").split(",")}
    if " in " in cond:
        key, vals = cond.split(" in ", 1)
        return str(state.get(key.strip(), "")) in {v.strip() for v in vals.strip(" {}").split(",")}
    if "==" in cond:
        key, value = cond.split("==", 1)
        return str(state.get(key.strip(), "")).lower() == value.strip().lower()
    return bool(state.get(cond, False))

def _eval_flag(flag: str, state: Mapping[str, Any], context: Mapping[str, Any]) -> bool:
    aliases = {
        "lava_nearby": ("near_lava", "lava_nearby"),
        "low_health": ("low_health",),
        "active_combat": ("in_combat", "combat_active"),
        "near_cliff": ("near_cliff",),
        "resource_critical": ("resource_critical", "irreversible_resource_constraint"),
    }
    keys = aliases.get(flag, (flag,))
    return any(bool(state.get(k, context.get(k, False))) for k in keys)

@dataclass
class Opportunity:
    episode_id: str
    opportunity_id: str
    round: int
    stream_seed: int
    task_id: str
    world_seed: int
    knowledge_id: str
    source: str
    type: str
    retrieval_rank: int
    retrieval_score: float
    raw_text_hash: str
    task_group: str
    failure_type: str
    risk_tier: str
    resource_scarcity: str
    boundary_status: str
    inventory_signature: str
    episode_phase: str = "early"
    prior_admission_bin: str = "0"
    prior_fallback_bin: str = "0"
    prior_harm_flag: int = 0
    remaining_critical_resource_ratio: float = 1.0
    # Logged estimate of eligible opportunities after this checkpoint.
    remaining_opportunities: int = 0
    time_since_last_window: int = 0
    collection_exposure_count: int = 0
    assignment: int = 0
    propensity_reuse: float = 0.5
    propensity_base: float = 0.5
    randomization_seed: int = 0
    start_step: int = 0
    end_step: int = 0
    window_type: str = "fixed"
    censor_flag: bool = False
    second_intervention_flag: bool = False
    eligible: bool = True
    eligibility_reason: str = "eligible"
    y: Optional[int] = None
    h1: Optional[int] = None
    h2: Optional[int] = None
    h3: Optional[int] = None
    h4: Optional[int] = None
    h5: Optional[int] = None
    h6: Optional[int] = None
    progress_delta: Optional[float] = None
    steps: Optional[int] = None
    resource_cost: Optional[float] = None
    token_cost: Optional[int] = None
    call_cost: Optional[int] = None
    label_source: str = "environment"
    annotator_status: str = "not_applicable"
    exclusion_reason: str = ""
    snapshot_hash: str = ""

    @property
    def harm(self) -> Optional[int]:
        if any(v is None for v in (self.h1, self.h2, self.h3, self.h4, self.h5, self.h6)):
            return None
        return int(bool(self.h1 or self.h2 or self.h3 or self.h4 or self.h5 or self.h6))

    def to_dict(self) -> Dict[str, Any]:
        row = asdict(self)
        row["schema_version"] = SCHEMA_VERSION
        row["harm"] = self.harm
        return row

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Opportunity":
        values = dict(data)
        values.pop("schema_version", None)
        values.pop("harm", None)
        return cls(**{k: values[k] for k in cls.__dataclass_fields__ if k in values})

class OpportunityLogger:
    REQUIRED = {
        "schema_version", "episode_id", "opportunity_id", "task_id", "world_seed",
        "knowledge_id", "assignment", "propensity_reuse", "propensity_base",
        "randomization_seed", "eligible", "eligibility_reason",
        "round", "stream_seed", "source", "type", "retrieval_rank",
        "retrieval_score", "raw_text_hash", "task_group", "failure_type",
        "risk_tier", "resource_scarcity", "episode_phase",
        "prior_admission_bin", "prior_fallback_bin", "prior_harm_flag",
        "remaining_critical_resource_ratio", "remaining_opportunities", "time_since_last_window",
        "collection_exposure_count", "boundary_status", "inventory_signature",
        "start_step", "end_step", "censor_flag", "second_intervention_flag",
        "window_type", "label_source", "annotator_status", "exclusion_reason",
        "snapshot_hash",
    }
    def __init__(self, path: os.PathLike[str] | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
    @classmethod
    def validate_row(cls, row: Mapping[str, Any]) -> None:
        missing = sorted(cls.REQUIRED - set(row))
        if missing:
            raise ValueError(f"{SCHEMA_VERSION} opportunity missing fields: {missing}")
        if row["schema_version"] != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema: {row['schema_version']}")
        if not row["eligible"] and row.get("eligibility_reason") == "eligible":
            raise ValueError("ineligible opportunity cannot use eligibility_reason=eligible")
        if row["eligible"]:
            p = float(row["propensity_reuse"])
            if not (0.2 <= p <= 0.8):
                raise ValueError(f"positivity violation: {p}")
            if abs(p + float(row["propensity_base"]) - 1.0) > 1e-6:
                raise ValueError("propensities must sum to one")
    def append(self, opportunity: Opportunity) -> None:
        row = opportunity.to_dict()
        self.validate_row(row)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    def load(self, eligible_only: bool = False) -> List[Opportunity]:
        if not self.path.exists():
            return []
        rows = []
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip(): continue
                row = json.loads(line)
                self.validate_row(row)
                if not eligible_only or row.get("eligible"):
                    rows.append(Opportunity.from_dict(row))
        return rows

class RandomizedAssignment:
    def __init__(self, probability: float = 0.5, seed: int = 0):
        self.probability = _clip_propensity(probability)
        self.seed = int(seed)
    def assign(self, opportunity_id: str) -> Tuple[int, float, int]:
        local_seed = _stable_hash(f"{self.seed}|{opportunity_id}")
        assignment = int(random.Random(local_seed).random() < self.probability)
        return assignment, self.probability, local_seed

@dataclass
class GroupEstimate:
    level: str
    key: str
    n: int
    n_reuse: int
    n_base: int
    ess: float
    delta_y: float
    se_y: float
    risk_abs: float
    se_abs: float
    risk_inc: float
    se_inc: float
    supported: bool
    backoff_level: Optional[str] = None
    parent_key: Optional[str] = None
    evidence_weight: float = 1.0
    evidence_nu: float = DEFAULT_EVIDENCE_NU
    # Arrays share draw ids, preserving benefit-risk dependence.
    joint_draws: Dict[str, List[float]] = field(default_factory=dict)
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class AIPWEstimator:
    # Protocol §8.2: knowledge + task + state + history features
    FEATURE_FIELDS = ("source", "type", "task_group", "failure_type",
                      "risk_tier", "resource_scarcity", "boundary_status",
                      "inventory_signature",
                      "episode_phase", "prior_admission_bin", "prior_fallback_bin",
                      "prior_harm_flag", "remaining_critical_resource_ratio",
                      "remaining_opportunities")
    def __init__(self, n_folds: int = 5, seed: int = 17, ridge: float = 1.0,
                 bootstrap_draws: int = 2000):
        self.n_folds, self.seed, self.ridge = int(n_folds), int(seed), float(ridge)
        self.bootstrap_draws = max(200, int(bootstrap_draws))
    def _features(self, rows: Sequence[Opportunity]) -> np.ndarray:
        width = 64
        matrix = np.zeros((len(rows), width), dtype=float)
        for i, row in enumerate(rows):
            for field_name in self.FEATURE_FIELDS:
                matrix[i, _stable_hash(f"{field_name}={getattr(row, field_name)}") % width] += 1.0
            matrix[i, _stable_hash(f"score={round(float(row.retrieval_score), 2)}") % width] += 1.0
        return np.column_stack([np.ones(len(rows)), matrix])
    def _fit(self, rows: Sequence[Opportunity], outcome: str, arm: int):
        selected = [r for r in rows if r.assignment == arm and getattr(r, outcome) is not None]
        if len(selected) < 2:
            return None
        x = self._features(selected)
        y = np.asarray([float(getattr(r, outcome)) for r in selected])
        beta = np.zeros(x.shape[1], dtype=float)
        n = len(selected)
        reg = self.ridge * np.eye(x.shape[1]); reg[0, 0] = 0.0
        for _ in range(80):
            z = np.clip(x @ beta, -30.0, 30.0)
            prob = 1.0 / (1.0 + np.exp(-z))
            w = np.maximum(prob * (1.0 - prob), 1e-4)
            hessian = x.T @ (w[:, None] * x) / n + reg
            gradient = x.T @ (prob - y) / n + reg @ beta
            try:
                step = np.linalg.solve(hessian, gradient)
            except np.linalg.LinAlgError:
                break
            beta -= step
            if float(np.max(np.abs(step))) < 1e-6:
                break
        return beta
    @staticmethod
    def _predict(beta, x):
        if beta is None:
            return np.full(len(x), 0.5)
        return 1.0 / (1.0 + np.exp(-np.clip(x @ beta, -30.0, 30.0)))
    def cross_fit(self, rows: Sequence[Opportunity], fit_rows: Sequence[Opportunity] = None) -> List[Dict[str, Any]]:
        eligible = [r for r in rows if r.eligible and not r.censor_flag and
                    not r.second_intervention_flag and r.y is not None and r.harm is not None]
        if not eligible: return []
        episodes = sorted({r.episode_id for r in eligible})
        n_folds = max(1, min(self.n_folds, len(episodes)))
        folds = {ep: _stable_hash(f"{self.seed}|{ep}") % n_folds for ep in episodes}
        pseudo = []
        for fold in sorted(set(folds.values())):
            test = [r for r in eligible if folds[r.episode_id] == fold]
            train = list(fit_rows) if fit_rows is not None else [r for r in eligible if folds[r.episode_id] != fold]
            x_test = self._features(test)
            models = {"y1": self._fit(train, "y", 1), "y0": self._fit(train, "y", 0),
                      "h1": self._fit(train, "harm", 1), "h0": self._fit(train, "harm", 0)}
            pred = {k: self._predict(v, x_test) for k, v in models.items()}
            for i, row in enumerate(test):
                e, a = _clip_propensity(row.propensity_reuse), int(row.assignment)
                y, h = float(row.y), float(row.harm)
                phi_y = pred["y1"][i] - pred["y0"][i] + a*(y-pred["y1"][i])/e - (1-a)*(y-pred["y0"][i])/(1-e)
                phi_h = pred["h1"][i] - pred["h0"][i] + a*(h-pred["h1"][i])/e - (1-a)*(h-pred["h0"][i])/(1-e)
                psi_h1 = pred["h1"][i] + a*(h-pred["h1"][i])/e
                pseudo.append({"episode_id": row.episode_id, "opportunity_id": row.opportunity_id,
                               "task_group": row.task_group, "risk_tier": row.risk_tier,
                               "source": row.source, "type": row.type,
                               "boundary_status": row.boundary_status, "failure_type": row.failure_type,
                               "resource_scarcity": row.resource_scarcity,
                               "episode_phase": row.episode_phase,
                               "prior_admission_bin": row.prior_admission_bin,
                               "prior_fallback_bin": row.prior_fallback_bin,
                               "prior_harm_flag": row.prior_harm_flag,
                               "remaining_critical_resource_ratio": row.remaining_critical_resource_ratio,
                               "assignment": a,
                               "propensity": e, "phi_y": float(phi_y), "phi_h": float(phi_h),
                               "psi_h1": float(psi_h1), "y": y, "harm": h})
        return pseudo
    @staticmethod
    def _cluster_se(rows: Sequence[Mapping[str, Any]], field: str) -> float:
        clusters = {}
        for row in rows:
            clusters.setdefault(row["episode_id"], []).append(float(row[field]))
        means = np.asarray([np.mean(v) for v in clusters.values()], dtype=float)
        return float(np.std(means, ddof=1) / math.sqrt(len(means))) if len(means) >= 2 else float("inf")

    def _joint_bootstrap(self, rows: Sequence[Mapping[str, Any]], key: str) -> Dict[str, List[float]]:
        """Episode-clustered paired draws for the unified CAP.

        All three arrays use the same resampled episode ids, so CAP never
        combines a benefit draw with an unrelated worst-case harm draw.
        """
        clusters: Dict[str, List[Mapping[str, Any]]] = {}
        for row in rows:
            clusters.setdefault(str(row["episode_id"]), []).append(row)
        episodes = sorted(clusters)
        if not episodes:
            return {}
        rng = np.random.default_rng(_stable_hash(f"joint|{self.seed}|{key}") % (2**32))
        b = self.bootstrap_draws
        out = {"delta_y": [], "risk_inc": [], "risk_abs": []}
        # A single cluster has no empirical bootstrap variance.  Keep a
        # deterministic degenerate draw rather than fabricating uncertainty.
        for _ in range(b):
            chosen = rng.integers(0, len(episodes), size=len(episodes))
            sample = [row for idx in chosen for row in clusters[episodes[int(idx)]]]
            out["delta_y"].append(float(np.mean([r["phi_y"] for r in sample])))
            out["risk_inc"].append(float(np.mean([r["phi_h"] for r in sample])))
            out["risk_abs"].append(float(np.mean([r["psi_h1"] for r in sample])))
        return out

    def aggregate(self, pseudo: Sequence[Mapping[str, Any]]) -> List[GroupEstimate]:
        if not pseudo: return []
        # Protocol §7.1: 4-level hierarchy (g0=8 fields, g1=6, g2=4, g3=2)
        groups = {
            "g0": lambda r: f"{r['source']}|{r['type']}|{r['task_group']}|{r.get('failure_type','none')}|{r['risk_tier']}|{r.get('resource_scarcity','ordinary')}|{r.get('episode_phase','early')}|{r.get('prior_admission_bin','0')}",
            "g1": lambda r: f"{r['source']}|{r['type']}|{r['task_group']}|{r.get('failure_type','none')}|{r['risk_tier']}|{r.get('episode_phase','early')}",
            "g2": lambda r: f"{r['source']}|{r['type']}|{r['task_group']}|{r.get('resource_scarcity','ordinary')}",
            "g3": lambda r: f"{r['source']}|{r['type']}",
        }
        def _heterogeneity_se(rows, child_fn, field):
            child_values = []
            for child in sorted({child_fn(r) for r in rows}):
                child_rows = [r for r in rows if child_fn(r) == child]
                if not child_rows:
                    continue
                child_values.append((len(child_rows), float(np.mean([r[field] for r in child_rows])),
                                     self._cluster_se(child_rows, field)))
            if len(child_values) < 2:
                return 0.0
            total = sum(n for n, _, _ in child_values)
            mean = sum(n * value for n, value, _ in child_values) / total
            weighted_var = sum(n * (value - mean) ** 2 for n, value, _ in child_values) / total
            sampling = sum(n * (se ** 2) for n, _, se in child_values) / total
            return math.sqrt(max(0.0, weighted_var - sampling))
        estimates = []
        levels = list(groups.items())
        # First compute raw group estimates and retain the parent relation.
        for level_index, (level, key_fn) in enumerate(levels):
            parent_fn = levels[level_index + 1][1] if level_index + 1 < len(levels) else None
            for key in sorted({key_fn(r) for r in pseudo}):
                rows = [r for r in pseudo if key_fn(r) == key]
                nr = sum(r["assignment"] == 1 for r in rows); nb = len(rows) - nr
                weights = np.asarray([1/r["propensity"] if r["assignment"] else 1/(1-r["propensity"]) for r in rows])
                ess = float(weights.sum()**2 / max((weights**2).sum(), 1e-12))
                se_y = self._cluster_se(rows, "phi_y")
                se_abs = self._cluster_se(rows, "psi_h1")
                se_inc = self._cluster_se(rows, "phi_h")
                if level_index:
                    child_fn = levels[level_index - 1][1]
                    se_y = math.sqrt(se_y ** 2 + _heterogeneity_se(rows, child_fn, "phi_y") ** 2)
                    se_abs = math.sqrt(se_abs ** 2 + _heterogeneity_se(rows, child_fn, "psi_h1") ** 2)
                    se_inc = math.sqrt(se_inc ** 2 + _heterogeneity_se(rows, child_fn, "phi_h") ** 2)
                estimates.append(GroupEstimate(
                    level=level, key=key, n=len(rows), n_reuse=nr, n_base=nb, ess=ess,
                    delta_y=float(np.mean([r["phi_y"] for r in rows])), se_y=se_y,
                    risk_abs=float(np.mean([r["psi_h1"] for r in rows])), se_abs=se_abs,
                    risk_inc=float(np.mean([r["phi_h"] for r in rows])), se_inc=se_inc,
                    supported=nr >= MIN_ARM_SUPPORT and nb >= MIN_ARM_SUPPORT and ess >= MIN_ESS,
                    parent_key=(parent_fn(rows[0]) if parent_fn else None),
                    joint_draws=self._joint_bootstrap(rows, key)))
        # Continuous evidence flow: each child is shrunk toward its parent with
        # omega=ESS/(ESS+nu), preserving aligned joint draws. Root support is
        # still a hard domain-validity requirement; children may borrow smoothly.
        by_key = {e.key: e for e in estimates}
        for e in sorted(estimates, key=lambda x: int(x.level[1:]), reverse=True):
            if not e.parent_key or e.parent_key not in by_key:
                continue
            parent = by_key[e.parent_key]
            nu = DEFAULT_EVIDENCE_NU
            omega = float(np.clip(e.ess / max(e.ess + nu, 1e-12), 0.0, 1.0))
            e.evidence_weight = omega; e.evidence_nu = nu
            e.delta_y = omega * e.delta_y + (1.0 - omega) * parent.delta_y
            e.risk_inc = omega * e.risk_inc + (1.0 - omega) * parent.risk_inc
            e.risk_abs = omega * e.risk_abs + (1.0 - omega) * parent.risk_abs
            e.se_y = math.sqrt((omega * e.se_y) ** 2 + ((1.0 - omega) * parent.se_y) ** 2)
            e.se_inc = math.sqrt((omega * e.se_inc) ** 2 + ((1.0 - omega) * parent.se_inc) ** 2)
            e.se_abs = math.sqrt((omega * e.se_abs) ** 2 + ((1.0 - omega) * parent.se_abs) ** 2)
            e.supported = bool(e.supported or parent.supported)
            child_draws, parent_draws = e.joint_draws, parent.joint_draws
            n = min(len(child_draws.get("delta_y", [])), len(parent_draws.get("delta_y", [])))
            if n >= 32:
                for name in ("delta_y", "risk_inc", "risk_abs"):
                    e.joint_draws[name] = (omega * np.asarray(child_draws[name][:n]) +
                                           (1.0 - omega) * np.asarray(parent_draws[name][:n])).tolist()
        return estimates

@dataclass
class CalibratedPolicy:
    # kappa is retained only to read pre-CAP artifacts.  Online decisions use
    # lambda_value and the single CAP scalar below.
    kappa: float = 0.0
    delta: float = DEFAULT_DELTA
    eps_abs: float = DEFAULT_EPS_ABS
    eps_inc: float = DEFAULT_EPS_INC
    lambda_value: float = 1.0
    alpha: float = DEFAULT_CAP_ALPHA
    cap_epsilon: float = DEFAULT_CAP_EPSILON
    future_opportunity_clip: int = DEFAULT_FUTURE_OPPORTUNITY_CLIP
    estimates: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    coverage: float = 0.0
    supported: bool = False
    audit_passed: bool = False

    @property
    def delta0(self) -> float:
        """Preregistered CAP benefit scale (legacy ``delta`` alias)."""
        return max(float(self.delta), DEFAULT_CAP_EPSILON)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        allowed = set(cls.__dataclass_fields__)
        values = {k: v for k, v in dict(data).items() if k in allowed}
        # Old artifacts have no lambda/CAP fields; loading them remains safe,
        # while every newly selected artifact records the full schema.
        values.setdefault("lambda_value", 1.0)
        values.setdefault("alpha", DEFAULT_CAP_ALPHA)
        values.setdefault("cap_epsilon", DEFAULT_CAP_EPSILON)
        return cls(**values)


def _estimate_draw_arrays(estimate: Mapping[str, Any], seed: int = 0,
                          draws: int = 2000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load paired draws or make a deterministic compatibility envelope."""
    jd = estimate.get("joint_draws") or {}
    try:
        y = np.asarray(jd.get("delta_y", []), dtype=float)
        inc = np.asarray(jd.get("risk_inc", []), dtype=float)
        abs_r = np.asarray(jd.get("risk_abs", []), dtype=float)
        n = min(len(y), len(inc), len(abs_r))
        if n >= 32 and np.all(np.isfinite(np.concatenate((y[:n], inc[:n], abs_r[:n])))):
            return y[:n], inc[:n], abs_r[:n]
    except (TypeError, ValueError):
        pass
    # Compatibility for legacy point estimates: use a shared normal draw for
    # benefit/incremental harm, retaining their dependence rather than taking
    # independent worst cases.
    rng = np.random.default_rng(int(seed) % (2**32))
    z = rng.standard_normal(max(200, int(draws)))
    y0, sy = float(estimate.get("delta_y", 0.0)), max(0.0, float(estimate.get("se_y", 0.0)))
    h0, sh = float(estimate.get("risk_inc", 0.0)), max(0.0, float(estimate.get("se_inc", 0.0)))
    a0, sa = float(estimate.get("risk_abs", 0.0)), max(0.0, float(estimate.get("se_abs", 0.0)))
    return y0 + sy * z, h0 + sh * z, a0 + sa * rng.standard_normal(len(z))


def _cap_draws(estimate: Mapping[str, Any], lambda_value: float,
               q_t: float, delta0: float, epsilon: float,
               seed: int = 0, alpha: float = DEFAULT_CAP_ALPHA) -> Dict[str, Any]:
    y, inc, abs_r = _estimate_draw_arrays(estimate, seed=seed)
    cap_draw = y / max(delta0, epsilon) - float(lambda_value) * np.maximum(inc, 0.0) / max(q_t + epsilon, epsilon)
    return {
        "cap": float(np.quantile(cap_draw, alpha)),
        "benefit_lcb": float(np.quantile(y, alpha)),
        "risk_inc_ucb": float(np.quantile(inc, 1.0 - alpha)),
        "risk_abs_ucb": float(np.quantile(abs_r, 1.0 - alpha)),
        "risk_charge": float(max(0.0, np.quantile(inc, 1.0 - alpha))),
        "draw_count": int(len(cap_draw)),
    }


class PolicyCalibrator:
    def __init__(self, kappas=DEFAULT_KAPPAS, delta=DEFAULT_DELTA,
                 eps_abs=DEFAULT_EPS_ABS, eps_inc=DEFAULT_EPS_INC,
                 lambdas=DEFAULT_LAMBDAS, alpha=DEFAULT_CAP_ALPHA):
        self.kappas = tuple(kappas)  # compatibility metadata only
        self.lambdas = tuple(float(x) for x in lambdas)
        self.delta, self.eps_abs, self.eps_inc = float(delta), float(eps_abs), float(eps_inc)
        self.alpha = float(alpha)

    def _candidate(self, estimates, lambda_value):
        supported = [e for e in estimates if e.supported]
        admitted = []
        for e in supported:
            stats = _cap_draws(e.to_dict(), lambda_value, DEFAULT_BUDGET,
                               self.delta, DEFAULT_CAP_EPSILON,
                               seed=_stable_hash(e.key), alpha=self.alpha)
            if stats["risk_abs_ucb"] <= self.eps_abs and stats["cap"] > 0.0:
                admitted.append(e)
        return supported, admitted

    def select(self, estimates: Sequence[GroupEstimate]) -> CalibratedPolicy:
        best = None
        for lam in self.lambdas:
            supported, admitted = self._candidate(estimates, lam)
            if not supported:
                continue
            candidate = CalibratedPolicy(
                kappa=0.0, delta=self.delta, eps_abs=self.eps_abs,
                eps_inc=self.eps_inc, lambda_value=float(lam), alpha=self.alpha,
                estimates={e.key: e.to_dict() for e in admitted},
                coverage=len(admitted) / len(supported), supported=True)
            if best is None or candidate.coverage > best.coverage or \
                    (candidate.coverage == best.coverage and lam < best.lambda_value):
                best = candidate
        return best or CalibratedPolicy(lambda_value=max(self.lambdas),
                                        delta=self.delta, eps_abs=self.eps_abs,
                                        eps_inc=self.eps_inc, alpha=self.alpha,
                                        supported=bool(estimates))

    def audit(self, policy: CalibratedPolicy, estimates: Sequence[GroupEstimate]) -> CalibratedPolicy:
        if not policy.supported:
            policy.audit_passed = False
            return policy
        by_key = {e.key: e for e in estimates if e.supported}
        policy.audit_passed = bool(policy.estimates) and all(
            key in by_key and
            _cap_draws(by_key[key].to_dict(), policy.lambda_value, DEFAULT_BUDGET,
                       policy.delta0, policy.cap_epsilon,
                       seed=_stable_hash(key), alpha=policy.alpha)["risk_abs_ucb"] <= policy.eps_abs and
            _cap_draws(by_key[key].to_dict(), policy.lambda_value, DEFAULT_BUDGET,
                       policy.delta0, policy.cap_epsilon,
                       seed=_stable_hash(key), alpha=policy.alpha)["cap"] > 0.0
            for key in policy.estimates)
        return policy


class AdmissionPolicyV2:
    """Online unified CAP policy with deterministic no-credit budget updates."""
    def __init__(self, policy: CalibratedPolicy, use_ledger: bool = True,
                 initial_budget: float = DEFAULT_BUDGET,
                 future_opportunity_lookup: Optional[Mapping[str, Any]] = None,
                 variant: str = "full"):
        self.policy, self._est = policy, policy.estimates
        self.variant = str(variant or "full").lower().replace("-", "_")
        self.use_ledger = bool(use_ledger)
        self.initial_budget = max(0.0, float(initial_budget))
        self.future_opportunity_lookup = dict(future_opportunity_lookup or {})
        self._budget_by_episode: Dict[str, float] = {}

    @classmethod
    def load(cls, path, use_ledger: bool = True,
             initial_budget: float = DEFAULT_BUDGET, family: str = "full",
             future_opportunity_lookup: Optional[Mapping[str, Any]] = None,
             variant: str = "full"):
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        families = data.get("families")
        if isinstance(families, dict):
            if family not in families:
                raise ValueError(f"policy artifact missing family '{family}'; available: {sorted(families)}")
            selected = families[family]
            # A lookup may be frozen alongside the policy artifact.
            if future_opportunity_lookup is None:
                future_opportunity_lookup = data.get("future_opportunity_lookup", {})
        elif family != "full":
            raise ValueError(f"policy artifact missing 'families' key; cannot load family '{family}'")
        else:
            selected = data
        draw_path = data.get("joint_evidence_draws_path")
        if draw_path:
            candidate = Path(draw_path)
            if not candidate.is_absolute():
                candidate = Path(path).resolve().parent / candidate
            if candidate.exists():
                from .joint_evidence import JointEvidenceDrawStore
                external = JointEvidenceDrawStore.read(candidate)
                selected = dict(selected)
                estimates = {k: dict(v) for k, v in (selected.get("estimates", {}) or {}).items()}
                for key, draws in external.items():
                    if key in estimates:
                        estimates[key]["joint_draws"] = draws
                selected["estimates"] = estimates
        return cls(CalibratedPolicy.from_dict(selected), use_ledger=use_ledger,
                   initial_budget=initial_budget,
                   future_opportunity_lookup=future_opportunity_lookup, variant=variant)

    def _budget_before(self, episode_id: str) -> float:
        return self._budget_by_episode.setdefault(str(episode_id), self.initial_budget)

    def _remaining_opportunities(self, opportunity: Opportunity) -> int:
        value = getattr(opportunity, "remaining_opportunities", 0)
        keys = (opportunity.opportunity_id, opportunity.episode_id, opportunity.task_id)
        for key in keys:
            if key in self.future_opportunity_lookup:
                value = self.future_opportunity_lookup[key]
                break
        try:
            return max(0, min(int(value), int(self.policy.future_opportunity_clip)))
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _base_certificate(opportunity, budget_before, q_t, n_remaining, applicable):
        return {"episode_id": opportunity.episode_id,
                "opportunity_id": opportunity.opportunity_id,
                "candidate_id": opportunity.knowledge_id,
                "applicable": bool(applicable),
                "remaining_opportunities": n_remaining,
                "q_t": q_t, "lambda": None, "cap": float("-inf"),
                "draw_count": 0,
                "benefit_lcb": 0.0, "risk_abs_ucb": 0.0,
                "risk_inc_ucb": 0.0, "risk_charge": 0.0,
                "budget_before": budget_before, "budget_after": budget_before}

    def decide(self, opportunity: Opportunity, applicable: bool = True) -> Dict[str, Any]:
        budget_before = self._budget_before(opportunity.episode_id)
        n_remaining = self._remaining_opportunities(opportunity)
        if self.variant == "myopic_reserve":
            n_remaining = 0
        q_t = budget_before / (1.0 + n_remaining)
        base = self._base_certificate(opportunity, budget_before, q_t, n_remaining, applicable)
        base["lambda"] = float(self.policy.lambda_value)
        if not opportunity.eligible or not applicable:
            return {**base, "decision": "FALLBACK", "reason": "ineligible_or_inapplicable",
                    "depth": None, "support_level": None}
        candidates = [
            ("g0", f"{opportunity.source}|{opportunity.type}|{opportunity.task_group}|{opportunity.failure_type}|{opportunity.risk_tier}|{opportunity.resource_scarcity}|{opportunity.episode_phase}|{opportunity.prior_admission_bin}"),
            ("g1", f"{opportunity.source}|{opportunity.type}|{opportunity.task_group}|{opportunity.failure_type}|{opportunity.risk_tier}|{opportunity.episode_phase}"),
            ("g2", f"{opportunity.source}|{opportunity.type}|{opportunity.task_group}|{opportunity.resource_scarcity}"),
            ("g3", f"{opportunity.source}|{opportunity.type}")]
        for depth, key in candidates:
            row = self._est.get(key)
            if not row or not row.get("supported", False):
                continue
            stats = _cap_draws(row, self.policy.lambda_value, q_t,
                               self.policy.delta0, self.policy.cap_epsilon,
                               seed=_stable_hash(key), alpha=self.policy.alpha)
            if self.variant == "hard_backoff" and depth != "g0":
                continue
            if self.variant == "independent_margins":
                benefit_lcb = float(row.get("delta_y", 0.0)) - self.policy.lambda_value * float(row.get("se_y", 0.0))
                abs_ucb = float(row.get("risk_abs", 0.0)) + self.policy.lambda_value * float(row.get("se_abs", 0.0))
                inc_ucb = float(row.get("risk_inc", 0.0)) + self.policy.lambda_value * float(row.get("se_inc", 0.0))
                stats.update({"cap": benefit_lcb / self.policy.delta0 - self.policy.lambda_value * max(inc_ucb, 0.0) / max(q_t + self.policy.cap_epsilon, self.policy.cap_epsilon),
                              "benefit_lcb": benefit_lcb, "risk_abs_ucb": abs_ucb,
                              "risk_inc_ucb": inc_ucb, "risk_charge": max(0.0, inc_ucb)})
            certificate = {**base, **stats, "depth": depth,
                           "support_level": depth, "key": key}
            if stats["risk_abs_ucb"] > self.policy.eps_abs:
                return {**certificate, "decision": "FALLBACK", "reason": "absolute_risk"}
            if self.variant == "independent_margins" and stats["benefit_lcb"] < self.policy.delta0:
                return {**certificate, "decision": "FALLBACK", "reason": "benefit_too_low"}
            if self.variant == "independent_margins" and stats["risk_inc_ucb"] > self.policy.eps_inc:
                return {**certificate, "decision": "FALLBACK", "reason": "incremental_risk"}
            if stats["risk_charge"] > budget_before + self.policy.cap_epsilon:
                return {**certificate, "decision": "FALLBACK", "reason": "budget_exhausted"}
            if stats["cap"] <= 0.0:
                return {**certificate, "decision": "FALLBACK", "reason": "cap_nonpositive"}
            budget_after = budget_before - stats["risk_charge"] if self.use_ledger else budget_before
            if self.use_ledger:
                self._budget_by_episode[opportunity.episode_id] = max(0.0, budget_after)
            return {**certificate, "decision": "ADMIT", "reason": "admit",
                    "budget_after": budget_after}
        return {**base, "decision": "FALLBACK", "reason": "unsupported",
                "depth": None, "support_level": None}

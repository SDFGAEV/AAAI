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
LEGACY_METHODS = ("XENON-Original", "BankCuration", "LifecycleSuccessGate",
                  "SuccessLifecycle", "ACT", "C-ACT-Full", "OracleGate",
                  "ShuffledKnowledge", "Online-NoGate", "Online-FixedBayes",
                  "Online-C-ACT-Pointwise", "Online-C-ACT")
METHOD_ALIASES = {
    "C-ACT-Full": "C-ACT",
    "SuccessLifecycle": "LifecycleSuccessGate",
}
DEFAULT_EPS_ABS = 0.10
DEFAULT_EPS_INC = 0.02
DEFAULT_DELTA = 0.05
DEFAULT_BUDGET = 0.05
DEFAULT_KAPPAS = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
MIN_ARM_SUPPORT = 12
MIN_ESS = 24.0

def canonical_method_name(method: str) -> str:
    return METHOD_ALIASES.get(str(method), str(method))

def validate_method_name(method: str, allow_legacy: bool = True) -> str:
    canonical = canonical_method_name(method)
    allowed = set(MAIN_METHODS)
    if allow_legacy:
        allowed.update(LEGACY_METHODS)
        allowed.update(METHOD_ALIASES)
        if canonical.startswith("C-ACT-") or canonical.startswith("Online-C-ACT-"):
            return canonical
    if canonical not in allowed and method not in allowed:
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
        if any(v is None for v in (self.h1, self.h2, self.h3, self.h4)):
            return None
        return int(bool(self.h1 or self.h2 or self.h3 or self.h4))

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
        "remaining_critical_resource_ratio", "time_since_last_window",
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
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class AIPWEstimator:
    FEATURE_FIELDS = ("source", "type", "task_group", "failure_type",
                      "risk_tier", "resource_scarcity", "boundary_status",
                      "inventory_signature")
    def __init__(self, n_folds: int = 5, seed: int = 17, ridge: float = 1.0):
        self.n_folds, self.seed, self.ridge = int(n_folds), int(seed), float(ridge)
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
        reg = self.ridge * np.eye(x.shape[1]); reg[0, 0] = 0.0
        for _ in range(80):
            z = np.clip(x @ beta, -30.0, 30.0)
            prob = 1.0 / (1.0 + np.exp(-z))
            w = np.maximum(prob * (1.0 - prob), 1e-4)
            hessian = x.T @ (w[:, None] * x) + reg
            gradient = x.T @ (prob - y) + reg @ beta
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
                               "resource_scarcity": row.resource_scarcity, "assignment": a,
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
    def aggregate(self, pseudo: Sequence[Mapping[str, Any]]) -> List[GroupEstimate]:
        if not pseudo: return []
        groups = {
            "g0": lambda r: f"{r['source']}|{r['type']}|{r['task_group']}|{r.get('failure_type', 'none')}|{r['risk_tier']}|{r.get('resource_scarcity', 'ordinary')}|{r['boundary_status']}",
            "g1": lambda r: f"{r['source']}|{r['type']}|{r['task_group']}|{r.get('failure_type', 'none')}|{r['risk_tier']}",
            "g2": lambda r: f"{r['source']}|{r['type']}|{r['task_group']}",
            "g3": lambda r: f"{r['source']}|{r['type']}",
        }
        estimates = []
        for level, key_fn in groups.items():
            for key in sorted({key_fn(r) for r in pseudo}):
                rows = [r for r in pseudo if key_fn(r) == key]
                nr = sum(r["assignment"] == 1 for r in rows); nb = len(rows) - nr
                weights = np.asarray([1/r["propensity"] if r["assignment"] else 1/(1-r["propensity"]) for r in rows])
                ess = float(weights.sum()**2 / max((weights**2).sum(), 1e-12))
                estimates.append(GroupEstimate(
                    level=level, key=key, n=len(rows), n_reuse=nr, n_base=nb, ess=ess,
                    delta_y=float(np.mean([r["phi_y"] for r in rows])),
                    se_y=self._cluster_se(rows, "phi_y"),
                    risk_abs=float(np.mean([r["psi_h1"] for r in rows])),
                    se_abs=self._cluster_se(rows, "psi_h1"),
                    risk_inc=float(np.mean([r["phi_h"] for r in rows])),
                    se_inc=self._cluster_se(rows, "phi_h"),
                    supported=nr >= MIN_ARM_SUPPORT and nb >= MIN_ARM_SUPPORT and ess >= MIN_ESS))
        return estimates

@dataclass
class CalibratedPolicy:
    kappa: float
    delta: float = DEFAULT_DELTA
    eps_abs: float = DEFAULT_EPS_ABS
    eps_inc: float = DEFAULT_EPS_INC
    estimates: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    coverage: float = 0.0
    supported: bool = False
    audit_passed: bool = False
    def to_dict(self): return asdict(self)
    @classmethod
    def from_dict(cls, data):
        allowed = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in dict(data).items() if k in allowed})

class PolicyCalibrator:
    def __init__(self, kappas=DEFAULT_KAPPAS, delta=DEFAULT_DELTA,
                 eps_abs=DEFAULT_EPS_ABS, eps_inc=DEFAULT_EPS_INC):
        self.kappas, self.delta, self.eps_abs, self.eps_inc = tuple(kappas), delta, eps_abs, eps_inc
    def select(self, estimates: Sequence[GroupEstimate]) -> CalibratedPolicy:
        supported = [e for e in estimates if e.supported]
        best = None
        for kappa in self.kappas:
            admitted = []
            for e in supported:
                if e.delta_y-kappa*e.se_y >= self.delta and e.risk_abs+kappa*e.se_abs <= self.eps_abs and e.risk_inc+kappa*e.se_inc <= self.eps_inc:
                    admitted.append(e)
            if supported:
                candidate = CalibratedPolicy(kappa=float(kappa), estimates={e.key:e.to_dict() for e in admitted},
                                             coverage=len(admitted)/len(supported), supported=True)
                if best is None or candidate.coverage > best.coverage or (candidate.coverage == best.coverage and kappa < best.kappa):
                    best = candidate
        return best or CalibratedPolicy(kappa=max(self.kappas), supported=bool(supported))
    def audit(self, policy: CalibratedPolicy, estimates: Sequence[GroupEstimate]) -> CalibratedPolicy:
        if not policy.supported: policy.audit_passed = False; return policy
        by_key = {e.key:e for e in estimates if e.supported}
        policy.audit_passed = bool(policy.estimates) and all(
            key in by_key
            and by_key[key].risk_abs + policy.kappa * by_key[key].se_abs <= policy.eps_abs
            and by_key[key].risk_inc + policy.kappa * by_key[key].se_inc <= policy.eps_inc
            for key in policy.estimates)
        return policy

class AdmissionPolicyV2:
    def __init__(self, policy: CalibratedPolicy,
                 use_ledger: bool = True,
                 initial_budget: float = DEFAULT_BUDGET):
        self.policy, self._est = policy, policy.estimates
        self.use_ledger = bool(use_ledger)
        self.initial_budget = float(initial_budget)
        self._budget_by_episode: Dict[str, float] = {}
    @classmethod
    def load(cls, path, use_ledger: bool = True,
             initial_budget: float = DEFAULT_BUDGET, family: str = "full"):
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data.get("families"), dict):
            data = data["families"].get(family, data)
        return cls(CalibratedPolicy.from_dict(data),
                   use_ledger=use_ledger, initial_budget=initial_budget)
    def _budget_before(self, episode_id: str) -> float:
        return self._budget_by_episode.setdefault(episode_id, self.initial_budget)
    def decide(self, opportunity: Opportunity, applicable: bool = True) -> Dict[str, Any]:
        budget_before = self._budget_before(opportunity.episode_id)
        if not opportunity.eligible or not applicable:
            return {"decision":"FALLBACK", "reason":"ineligible_or_inapplicable", "depth":None,
                    "budget_before": budget_before, "budget_after": budget_before,
                    "risk_charge": 0.0}
        candidates = [
            ("g0", f"{opportunity.source}|{opportunity.type}|{opportunity.task_group}|{opportunity.failure_type}|{opportunity.risk_tier}|{opportunity.resource_scarcity}|{opportunity.boundary_status}"),
            ("g1", f"{opportunity.source}|{opportunity.type}|{opportunity.task_group}|{opportunity.failure_type}|{opportunity.risk_tier}"),
            ("g2", f"{opportunity.source}|{opportunity.type}|{opportunity.task_group}"),
            ("g3", f"{opportunity.source}|{opportunity.type}")]
        for depth, key in candidates:
            row = self._est.get(key)
            # Support is defined by the preregistered arm/ESS rules in the
            # estimator (12 per arm and ESS >= 24); do not add an
            # undocumented n>=40 gate at deployment time.
            if not row or not row.get("supported", False): continue
            benefit = float(row["delta_y"]) - self.policy.kappa*float(row["se_y"])
            abs_risk = float(row["risk_abs"]) + self.policy.kappa*float(row["se_abs"])
            inc_risk = float(row["risk_inc"]) + self.policy.kappa*float(row["se_inc"])
            charge = max(0.0, inc_risk)
            if benefit < self.policy.delta:
                reason = "benefit_too_low"
            elif abs_risk > self.policy.eps_abs:
                reason = "absolute_risk"
            elif inc_risk > self.policy.eps_inc:
                reason = "incremental_risk"
            elif self.use_ledger and charge > budget_before:
                reason = "budget_exhausted"
            else:
                budget_after = budget_before - charge if self.use_ledger else budget_before
                self._budget_by_episode[opportunity.episode_id] = budget_after
                return {"decision":"ADMIT", "reason":"admit", "depth":depth, "key":key,
                        "benefit_lcb":benefit, "risk_abs_ucb":abs_risk,
                        "risk_inc_ucb":inc_risk, "risk_charge": charge,
                        "budget_before": budget_before, "budget_after": budget_after}
            if benefit >= self.policy.delta and abs_risk <= self.policy.eps_abs and inc_risk <= self.policy.eps_inc:
                return {"decision":"FALLBACK", "reason":reason, "depth":depth, "key":key,
                        "benefit_lcb":benefit, "risk_abs_ucb":abs_risk,
                        "risk_inc_ucb":inc_risk, "risk_charge": charge,
                        "budget_before": budget_before, "budget_after": budget_before}
        return {"decision":"FALLBACK", "reason":"unsupported", "depth":None,
                "budget_before": budget_before, "budget_after": budget_before,
                "risk_charge": 0.0}


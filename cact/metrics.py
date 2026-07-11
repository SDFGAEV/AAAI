"""Schema-aware metrics for the C-ACT evaluation protocol."""
import math
from typing import Any, Dict, List, Tuple
import numpy as np

def _first(row: Dict, *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in row:
            return row[key]
    return default

def validate_schema(records: List[Dict], required: Tuple[str, ...] = None) -> None:
    required = required or ("decision", "success", "task_group", "harmful_reuse",
                            "pre_admit_contract_pass",
                            "postcondition_satisfied", "resource_conflict",
                            "chain_success")
    aliases = {
        "success": ("success", "outcome_success"),
        "task_group": ("task_group", "group"),
        "harmful_reuse": ("harmful_reuse", "is_harmful"),
        "pre_admit_contract_pass": ("pre_admit_contract_pass", "contract_satisfied_before"),
        "postcondition_satisfied": ("postcondition_satisfied", "postcondition_pass",
                                     "contract_satisfied_after"),
        "resource_conflict": ("resource_conflict",),
        "chain_success": ("chain_success",),
        "decision": ("decision",),
    }
    errors = []
    for i, row in enumerate(records):
        for field in required:
            if not any(k in row for k in aliases.get(field, (field,))):
                errors.append(f"row {i}: missing {field}")
    if errors:
        raise ValueError("Metric schema validation failed: " + "; ".join(errors[:8]))

def compute_sr(data: List[Dict]) -> float:
    return sum(bool(_first(x, "success", "outcome_success", default=False)) for x in data) / len(data) if data else 0.0

def compute_kus(data: List[Dict]) -> float:
    rows = [x for x in data if x.get("decision") in ("reuse", "probe")]
    return sum(bool(_first(x, "outcome_success", "success", default=False)) for x in rows) / len(rows) if rows else 0.0

def compute_hrr(data: List[Dict]) -> float:
    rows = [x for x in data if x.get("decision") in ("reuse", "probe")]
    return sum(bool(_first(x, "harmful_reuse", "is_harmful", default=False)) for x in rows) / len(rows) if rows else 0.0

def compute_irr(data: List[Dict]) -> float:
    rows = [x for x in data if x.get("type") == "remedy"]
    return sum(not bool(x.get("failure_resolved", False)) for x in rows) / len(rows) if rows else 0.0

def compute_coverage(data: List[Dict]) -> float:
    return sum(x.get("decision") in ("reuse", "probe") for x in data) / len(data) if data else 0.0

def compute_ece(data: List[Dict], n_bins: int = 5) -> float:
    pairs = []
    for x in data:
        score = _first(x, "pi_uplift", "trust_score")
        target = _first(x, "uplift_beneficial", "counterfactual_beneficial")
        if score is not None and target is not None:
            pairs.append((float(score), float(bool(target))))
    if len(pairs) < n_bins:
        return math.nan
    pairs.sort(key=lambda z: z[0])
    n = len(pairs)
    error = 0.0
    for i in range(n_bins):
        lo, hi = i * n // n_bins, (i + 1) * n // n_bins
        if hi > lo:
            error += abs(np.mean([p[0] for p in pairs[lo:hi]]) -
                         np.mean([p[1] for p in pairs[lo:hi]])) * (hi - lo) / n
    return float(error)

def compute_cov_risk(data: List[Dict], eps: float = 0.10):
    from scipy.stats import beta as beta_dist
    score_key = "pi_uplift" if any("pi_uplift" in x for x in data) else "trust_score"
    rows = sorted([x for x in data if x.get(score_key) is not None],
                  key=lambda x: x[score_key], reverse=True)
    if not rows:
        return 0.0, [], []
    n = len(rows); best = 0.0; covs = []; risks = []
    for row in rows:
        threshold = row[score_key]
        accepted = [x for x in rows if x[score_key] >= threshold]
        k = sum(bool(_first(x, "harmful_reuse", "is_harmful", default=False)) for x in accepted)
        a = len(accepted)
        ub = float(beta_dist.ppf(0.95, k + 1, a - k + 1)) if a else 1.0
        covs.append(a / n); risks.append(k / a if a else 0.0)
        if ub <= eps:
            best = max(best, a / n)
    return round(best, 3), covs, risks

def compute_hardsr(rows: List[Dict]) -> float:
    return compute_sr([x for x in rows if x.get("difficulty") == "hard"])

def compute_failuresr(rows: List[Dict]) -> float:
    return compute_sr([x for x in rows if _first(x, "task_group", "group") == "failure_recovery"])

def compute_interactionsr(rows: List[Dict]) -> float:
    return compute_sr([x for x in rows if _first(x, "task_group", "group") == "interaction_stress"])

def compute_rcr(rows: List[Dict]) -> float:
    return sum(bool(x.get("resource_conflict")) for x in rows) / len(rows) if rows else 0.0

def compute_cfr(rows: List[Dict]) -> float:
    return sum(not bool(x.get("chain_success")) for x in rows) / len(rows) if rows else 0.0

def compute_kpr(lifecycle_logs: List[Dict]) -> float:
    certified, deprecated = set(), set()
    for event in lifecycle_logs:
        kid = event.get("knowledge_id", "")
        if event.get("new_status") == "certified":
            certified.add(kid)
        if event.get("old_status") == "certified" and event.get("new_status") in ("deprecated", "disabled"):
            deprecated.add(kid)
    return len(deprecated) / len(certified) if certified else 0.0

def compute_csr(rows: List[Dict]) -> float:
    rows = [x for x in rows if x.get("decision") in ("reuse", "probe")]
    if not rows:
        return 0.0
    ok = sum(bool(_first(x, "pre_admit_contract_pass", "contract_satisfied_before", default=False)) and
             bool(_first(x, "postcondition_satisfied", "postcondition_pass",
                         "contract_satisfied_after", default=False)) for x in rows)
    return ok / len(rows)

def compute_cvr(rows: List[Dict]) -> float:
    rows = [x for x in rows if x.get("decision") in ("reuse", "probe")]
    if not rows:
        return 0.0
    return sum(not bool(_first(x, "postcondition_satisfied", "postcondition_pass",
                               "contract_satisfied_after", "contract_violation_after",
                               default=False)) for x in rows) / len(rows)


def _decision_rows(rows):
    return [r for r in rows if r.get("eligible", True) and not r.get("censor_flag", False)]

def _admitted(r):
    return r.get("decision") in ("reuse", "probe", "ADMIT", "admit")

def compute_episode_admission_coverage(rows):
    episodes = {}
    for r in rows:
        episodes.setdefault(r.get("episode_id", r.get("run_id", id(r))), []).append(r)
    return sum(any(_admitted(r) for r in rs) for rs in episodes.values()) / len(episodes) if episodes else 0.0

def compute_echr(rows):
    episodes = {}
    for r in _decision_rows(rows):
        eid = r.get("episode_id", r.get("run_id", id(r))); episodes.setdefault(eid, 0)
        if _admitted(r) and bool(_first(r, "harmful_reuse", "is_harmful", default=False)): episodes[eid] += 1
    return sum(episodes.values()) / len(episodes) if episodes else 0.0

def compute_eahr(rows):
    episodes = {}
    for r in _decision_rows(rows):
        eid = r.get("episode_id", r.get("run_id", id(r))); episodes.setdefault(eid, False)
        episodes[eid] = episodes[eid] or (_admitted(r) and bool(_first(r, "harmful_reuse", "is_harmful", default=False)))
    return sum(episodes.values()) / len(episodes) if episodes else 0.0

def compute_boundary_rates(rows):
    nonapp = [r for r in rows if str(r.get("boundary_status", "")).lower() in {"non_applicable", "blocked", "inapplicable"}]
    app = [r for r in rows if str(r.get("boundary_status", "")).lower() in {"applicable", "eligible"}]
    return {"far": sum(_admitted(r) for r in nonapp) / len(nonapp) if nonapp else 0.0,
            "tar": sum(_admitted(r) for r in app) / len(app) if app else 0.0}

def compute_budget_metrics(rows, initial_budget=0.05):
    admitted = [r for r in rows if _admitted(r)]
    charges = [max(0.0, float(r.get("risk_charge", r.get("certificate_risk_charge", 0.0)) or 0.0)) for r in admitted]
    exhausted = sum(str(r.get("reason", r.get("certificate_reason", ""))) == "budget_exhausted" for r in rows)
    return {"budget_utilization": min(1.0, sum(charges) / max(initial_budget, 1e-12)),
            "budget_triggered_fallback": exhausted / len(rows) if rows else 0.0,
            "total_risk_charge": sum(charges)}

def compute_late_harm_rate(rows, late_phases=("late", "final")):
    late = [r for r in rows if r.get("episode_phase") in late_phases and _admitted(r)]
    return sum(bool(_first(r, "harmful_reuse", "is_harmful", default=False)) for r in late) / len(late) if late else 0.0

def compute_backoff_metrics(rows):
    fallback = [r for r in rows if str(r.get("decision", "")).lower() in {"fallback", "reject"}]
    unsupported = [r for r in fallback if str(r.get("reason", r.get("certificate_reason", ""))).startswith("unsupported")]
    depths = [float(r["depth"]) for r in rows if r.get("depth") is not None]
    return {"unsupported_fallback_rate": len(unsupported) / len(rows) if rows else 0.0,
            "mean_backoff_depth": sum(depths) / len(depths) if depths else float("nan")}

def compute_aulc(values):
    vals = [float(v) for v in values if v is not None]
    if not vals: return 0.0
    if len(vals) == 1: return vals[0]
    area = np.trapezoid(vals, dx=1.0) if hasattr(np, "trapezoid") else np.trapz(vals, dx=1.0)
    return float(area / (len(vals) - 1))

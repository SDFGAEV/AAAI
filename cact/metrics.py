"""
Metrics for C-ACT trust gate evaluation.

Primary metrics (E3 main table):
  SR        — Success Rate
  HardSR    — Hard task Success Rate
  FailureSR — Failure Recovery Success Rate
  InteractionSR — Interaction Stress Success Rate
  KUS       — Knowledge Usage Success
  HRR       — Harmful Reuse Rate
  IRR       — Invalid Repair Rate
  Coverage@Risk<=10% — Risk-calibrated coverage

Diagnostic metrics:
  CSR       — Contract Satisfaction Rate
  CVR       — Contract Violation Rate

Online evolution metrics (E5):
  KPR       — Knowledge Pollution Rate
"""

import numpy as np
from typing import Dict, List, Tuple


def compute_sr(data: List[Dict]) -> float:
    """Success Rate."""
    if not data: return 0.0
    return sum(1 for x in data if x.get("success") or x.get("outcome_success")) / len(data)


def compute_kus(data: List[Dict]) -> float:
    """Knowledge Usage Success — P(success | reused knowledge)."""
    reused = [x for x in data if x.get("decision") == "reuse"]
    if not reused: return 0.0
    return sum(1 for x in reused if x.get("outcome_success")) / len(reused)


def compute_hrr(data: List[Dict]) -> float:
    """Harmful Reuse Rate — P(harmful | reused).

    Harmful defined as: (Δprogress ≤ 0 ∧ Cost > B) ∨ unrecoverable failure.
    """
    reused = [x for x in data if x.get("decision") == "reuse"]
    if not reused: return 0.0
    return sum(1 for x in reused if x.get("is_harmful")) / len(reused)


def compute_irr(data: List[Dict]) -> float:
    """Invalid Repair Rate — P(f_after=f_before ∨ Δprogress≤0 ∨ new_severe_failure | remedy used)."""
    remedies = [x for x in data if x.get("type") == "remedy"]
    if not remedies: return 0.0
    return sum(1 for x in remedies if not x.get("failure_resolved", False)) / len(remedies)


def compute_coverage(data: List[Dict]) -> float:
    """Coverage — fraction of reuse decisions where knowledge was allowed."""
    if not data: return 0.0
    return sum(1 for x in data if x.get("decision") == "reuse") / len(data)


def compute_ece(data: List[Dict], n_bins: int = 5) -> float:
    """Expected Calibration Error."""
    pairs = [(x.get("pi_uplift", x.get("trust_score", 0.5)),
              1.0 if x.get("outcome_success") else 0.0)
             for x in data if "pi_uplift" in x or "trust_score" in x]
    if len(pairs) < n_bins: return 0.0
    scores = np.array([p[0] for p in pairs])
    outcomes = np.array([p[1] for p in pairs])
    idx = np.argsort(scores)
    scores, outcomes = scores[idx], outcomes[idx]
    n = len(scores)
    e = 0.0
    for i in range(n_bins):
        lo, hi = i * n // n_bins, (i + 1) * n // n_bins
        if hi > lo:
            e += abs(np.mean(scores[lo:hi]) - np.mean(outcomes[lo:hi])) * (hi - lo) / n
    return float(e)


def compute_cov_risk(data: List[Dict], eps: float = 0.10) -> Tuple[float, List[float], List[float]]:
    """Coverage at risk <= eps, plus full (coverage, risk) curves."""
    from scipy.stats import beta as beta_dist
    score_key = "pi_uplift" if any("pi_uplift" in x for x in data) else "trust_score"
    sc = sorted([x for x in data if score_key in x and x[score_key] is not None],
                key=lambda x: x[score_key], reverse=True)
    if not sc: return 0.0, [], []
    N = len(sc)
    best_cov, covs, risks = 0.0, [], []
    for x in sc:
        t = x[score_key]
        acc = sum(1 for s in sc if s[score_key] >= t)
        cov = acc / N
        harm = sum(1 for s in sc if s[score_key] >= t and s.get("is_harmful"))
        risk_ub = float(beta_dist.ppf(0.95, harm + 1, acc - harm + 1)) if acc else 1.0
        covs.append(cov)
        risks.append(harm / acc if acc else 0)
        if risk_ub <= eps and cov > best_cov:
            best_cov = cov
    return round(best_cov, 3), covs, risks


def compute_hardsr(task_results: List[Dict]) -> float:
    """Success rate on tasks labeled difficulty='hard'."""
    hard = [t for t in task_results if t.get("difficulty") == "hard"]
    if not hard: return 0.0
    return sum(1 for t in hard if t.get("success")) / len(hard)


def compute_failuresr(task_results: List[Dict]) -> float:
    """Success rate on failure recovery tasks."""
    fr = [t for t in task_results if t.get("group") == "failure_recovery"]
    if not fr: return 0.0
    return sum(1 for t in fr if t.get("success")) / len(fr)


def compute_interactionsr(task_results: List[Dict]) -> float:
    """Success rate on interaction stress tasks."""
    ist = [t for t in task_results if t.get("group") == "interaction_stress"]
    if not ist: return 0.0
    return sum(1 for t in ist if t.get("success")) / len(ist)


def compute_rcr(interaction_logs: List[Dict]) -> float:
    """Resource Conflict Rate."""
    if not interaction_logs: return 0.0
    return sum(1 for x in interaction_logs if x.get("resource_conflict")) / len(interaction_logs)


def compute_cfr(interaction_logs: List[Dict]) -> float:
    """Chain Failure Rate."""
    if not interaction_logs: return 0.0
    return sum(1 for x in interaction_logs if not x.get("chain_success", True)) / len(interaction_logs)


def compute_kpr(lifecycle_logs: List[Dict]) -> float:
    """Knowledge Pollution Rate — certified knowledge later deprecated for harm."""
    ever_certified = set()
    deprecated_from_certified = 0
    for event in lifecycle_logs:
        kid = event.get("knowledge_id", "")
        old_s = event.get("old_status", "")
        new_s = event.get("new_status", "")
        if new_s == "certified":
            ever_certified.add(kid)
        if old_s == "certified" and new_s in ("deprecated", "disabled"):
            deprecated_from_certified += 1
    return deprecated_from_certified / max(len(ever_certified), 1)


def compute_csr(reuse_logs: List[Dict]) -> float:
    """Contract Satisfaction Rate — fraction of reuse decisions satisfying contract."""
    if not reuse_logs: return 0.0
    satisfied = sum(1 for x in reuse_logs if x.get("contract_satisfied_before", True)
                    and not x.get("contract_violation_after", False))
    return satisfied / len(reuse_logs)


def compute_cvr(reuse_logs: List[Dict]) -> float:
    """Contract Violation Rate — fraction of reuse decisions violating contract."""
    if not reuse_logs: return 0.0
    return sum(1 for x in reuse_logs if x.get("contract_violation_after", False)) / len(reuse_logs)

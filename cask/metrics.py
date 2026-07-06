"""
Metrics for CASK trust gate evaluation.

Log schema (produced by CaskMemory):
  decision:      "reuse" | "fallback" | "no_knowledge"
  outcome_success: bool
  is_harmful:    bool
  type:          "skill" | "remedy"
  failure_resolved: bool
  trust_score:   float | None

KUS  = Knowledge Usage Success   — P(success | reused knowledge)
HRR  = Harmful Reuse Rate        — P(harmful | reused)
IRR  = Invalid Repair Rate       — P(remedy fails to fix)
ECE  = Expected Calibration Error
"""

import numpy as np
from typing import Dict, List, Tuple


def compute_kus(data: List[Dict]) -> float:
    reused = [x for x in data if x.get("decision") == "reuse"]
    if not reused:
        return 0.0
    return sum(1 for x in reused if x.get("outcome_success")) / len(reused)


def compute_hrr(data: List[Dict]) -> float:
    reused = [x for x in data if x.get("decision") == "reuse"]
    if not reused:
        return 0.0
    return sum(1 for x in reused if x.get("is_harmful")) / len(reused)


def compute_irr(data: List[Dict]) -> float:
    remedies = [x for x in data if x.get("type") == "remedy"]
    if not remedies:
        return 0.0
    return sum(1 for x in remedies if not x.get("failure_resolved", False)) / len(remedies)


def compute_coverage(data: List[Dict]) -> float:
    if not data:
        return 0.0
    return sum(1 for x in data if x.get("decision") == "reuse") / len(data)


def compute_ece(data: List[Dict], n_bins: int = 5) -> float:
    pairs = [(x["trust_score"], 1.0 if x.get("outcome_success") else 0.0)
             for x in data if "trust_score" in x and x["trust_score"] is not None]
    if len(pairs) < n_bins:
        return 0.0
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


def compute_cov_risk(data: List[Dict], eps: float = 0.10
                     ) -> Tuple[float, List[float], List[float]]:
    """Coverage at risk <= eps, plus full (coverage, risk) curves."""
    from scipy.stats import beta as beta_dist
    sc = sorted([x for x in data if "trust_score" in x],
                key=lambda x: x["trust_score"], reverse=True)
    if not sc:
        return 0.0, [], []
    N = len(sc)
    best_cov, covs, risks = 0.0, [], []
    for x in sc:
        t = x["trust_score"]
        acc = sum(1 for s in sc if s["trust_score"] >= t)
        cov = acc / N
        harm = sum(1 for s in sc if s["trust_score"] >= t and s.get("is_harmful"))
        risk_ub = float(beta_dist.ppf(0.95, harm + 1, acc - harm + 1)) if acc else 1.0
        covs.append(cov)
        risks.append(harm / acc if acc else 0)
        if risk_ub <= eps and cov > best_cov:
            best_cov = cov
    return round(best_cov, 3), covs, risks


# ── Experiment-level metrics (task granularity) ──

def compute_hardsr(task_results: List[Dict]) -> float:
    """Success rate on hard tasks (tech-tree, long-horizon, failure-recovery)."""
    hard = [t for t in task_results if t.get("difficulty") == "hard" or t.get("group") in (
        "tech_tree", "failure_recovery", "interaction_stress")]
    if not hard:
        return 0.0
    return sum(1 for t in hard if t.get("success")) / len(hard)


def compute_rcr(interaction_logs: List[Dict]) -> float:
    """Resource Conflict Rate — fraction of episodes with resource conflict."""
    if not interaction_logs:
        return 0.0
    return sum(1 for x in interaction_logs if x.get("resource_conflict")) / len(interaction_logs)


def compute_cfr(interaction_logs: List[Dict]) -> float:
    """Chain Failure Rate — fraction of knowledge chains that fail."""
    if not interaction_logs:
        return 0.0
    return sum(1 for x in interaction_logs if not x.get("chain_success", True)) / len(interaction_logs)

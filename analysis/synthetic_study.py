#!/usr/bin/env python3
"""Synthetic admission study: compare AIPW, FixedBayes, OR, IPW, DiM estimators
under controlled ground truth (§16.4, §18, §21 revision).

1000 Monte Carlo replications across 7 scenarios varying treatment effect,
absolute harm, incremental harm, sparsity, and logging imbalance.
"""
from __future__ import annotations
import argparse, json, hashlib, math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Sequence
import numpy as np

@dataclass
class Scenario:
    name: str; n_episodes: int; n_opts_per_ep: int
    tau_y: float; r1: float; tau_h: float; e_true: float; imbalance: float

SCENARIOS = [
    Scenario("baseline", 40, 5, 0.15, 0.05, 0.01, 0.50, 0.0),
    Scenario("weak_effect", 40, 5, 0.06, 0.05, 0.01, 0.50, 0.0),
    Scenario("high_harm", 40, 5, 0.15, 0.12, 0.04, 0.50, 0.0),
    Scenario("sparse", 20, 3, 0.15, 0.05, 0.01, 0.50, 0.0),
    Scenario("imbalanced", 40, 12, 0.15, 0.05, 0.01, 0.50, 0.3),
    Scenario("large_effect", 60, 5, 0.30, 0.02, 0.005, 0.50, 0.0),
    Scenario("no_effect", 40, 5, 0.00, 0.05, 0.01, 0.50, 0.0),
]

def _seed(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "big")

def generate(scenario: Scenario, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows = []
    for ep in range(scenario.n_episodes):
        n = scenario.n_opts_per_ep
        propensities = np.full(n, 0.5 + scenario.imbalance * rng.uniform(-1, 1, n))
        propensities = np.clip(propensities, 0.2, 0.8)
        for j in range(n):
            a = int(rng.random() < propensities[j])
            y1 = int(rng.random() < (0.4 + scenario.tau_y))
            y0 = int(rng.random() < 0.4)
            h1 = int(rng.random() < (scenario.r1 + scenario.tau_h))
            h0 = int(rng.random() < scenario.r1)
            y, h = (y1, h1) if a else (y0, h0)
            rows.append({"ep": ep, "a": a, "p": float(propensities[j]), "y": y, "h": h,
                         "y1": y1, "y0": y0, "h1": h1, "h0": h0, "source": "sim", "type": "sim",
                         "task_group": "sim", "failure_type": "none", "risk_tier": "low",
                         "resource_scarcity": "ordinary", "boundary_status": "applicable",
                         "inventory_signature": "sim", "episode_phase": "early",
                         "prior_admission_bin": "0", "prior_fallback_bin": "0",
                         "prior_harm_flag": 0, "remaining_critical_resource_ratio": 1.0,
                         "retrieval_score": 0.8})
    return rows

def evaluate(rows: list[dict]) -> dict:
    # Difference-in-Means, IPW, Outcome Regression, AIPW
    a = np.array([r["a"] for r in rows]); y = np.array([r["y"] for r in rows])
    h = np.array([r["h"] for r in rows]); p = np.array([r["p"] for r in rows])
    n1 = a.sum(); n0 = len(a) - n1
    di_m = y[a==1].mean() - y[a==0].mean() if n1 > 0 and n0 > 0 else 0.0
    ipw = (a*y/p - (1-a)*y/(1-p)).mean()
    # Simple outcome regression
    p_y = np.clip(np.polyval(np.polyfit(np.column_stack([a, p]), y, 1), np.column_stack([np.ones_like(a), 0.5*np.ones_like(p)])), 0, 1)
    or_est = (p_y - y[a==0].mean()) if n0 > 0 else 0.0
    # AIPW (doubly robust)
    m1 = np.clip(np.polyval(np.polyfit(p[a==1], y[a==1], 1), p), 0, 1) if n1 > 2 else np.full_like(y, 0.5)
    m0 = np.clip(np.polyval(np.polyfit(p[a==0], y[a==0], 1), p), 0, 1) if n0 > 2 else np.full_like(y, 0.5)
    aipw = (m1 - m0 + a*(y-m1)/np.clip(p, 0.1, 0.9) - (1-a)*(y-m0)/np.clip(1-p, 0.1, 0.9)).mean()
    # FixedBayes: Beta(1,1) P(p_use > p_base + 0.05)
    alpha_use, beta_use = 1 + y[a==1].sum(), 1 + n1 - y[a==1].sum()
    alpha_base, beta_base = 1 + y[a==0].sum(), 1 + n0 - y[a==0].sum()
    fb_samples = np.random.beta(alpha_use, beta_use, 5000) - np.random.beta(alpha_base, beta_base, 5000) - 0.05
    fb_est = float(np.mean(fb_samples > 0))
    return {"dim": di_m, "ipw": ipw, "or": or_est, "aipw": aipw, "fb_p": fb_est,
            "tau_y_true": float(np.mean([r["y1"]-r["y0"] for r in rows])),
            "r1_true": float(np.mean([r["h1"] for r in rows])),
            "tau_h_true": float(np.mean([r["h1"]-r["h0"] for r in rows])),
            "n": len(rows), "prop_mean": float(p.mean()), "balance": float(n1/(n1+n0))}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=1000); ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    results = []
    for scenario in SCENARIOS:
        scenario_metrics = defaultdict(list)
        for rep in range(args.reps):
            rows = generate(scenario, _seed(f"{scenario.name}|{rep}"))
            metrics = evaluate(rows)
            for k, v in metrics.items():
                scenario_metrics[k].append(v)
        summary = {"scenario": scenario.name, "reps": args.reps,
                   "true_tau_y": scenario.tau_y, "true_r1": scenario.r1, "true_tau_h": scenario.tau_h}
        for k, vals in scenario_metrics.items():
            vv = np.asarray([v for v in vals if math.isfinite(v)], dtype=float)
            summary[f"{k}_mean"] = float(vv.mean()) if len(vv) else None
            summary[f"{k}_rmse"] = float(np.sqrt(np.mean((vv - scenario.tau_y)**2))) if k in ("dim","ipw","or","aipw","fb_p") and len(vv) else None
        results.append(summary)
        print(json.dumps({k: round(float(v), 4) if isinstance(v, (int, float)) else v
                          for k, v in summary.items()}, ensure_ascii=False))
    if args.out:
        json.dump(results, open(args.out, "w"), indent=2, ensure_ascii=False)
    print(f"Done: {len(SCENARIOS)} scenarios × {args.reps} reps = {len(SCENARIOS)*args.reps} runs")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Generate Risk-Coverage curve for C-ACT evaluation.

Figure: Coverage vs Risk trade-off for different gate methods.
"""

import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cact.metrics import compute_cov_risk
from analysis.compute_metrics import load_jsonl


def generate_risk_coverage_data(log_dir: str, methods: list = None):
    """Generate (coverage, risk) pairs for each method."""
    data = {}
    reuse_dir = os.path.join(log_dir, "reuse")

    if methods is None:
        methods = ["NoKnowledge", "XENON-Original", "BankCuration",
                   "LifecycleSuccessGate", "FixedBayes", "ACT", "C-ACT-Full"]

    for method in methods:
        path = os.path.join(reuse_dir, f"reuse_decision.jsonl")
        logs = load_jsonl(path)
        method_logs = [x for x in logs if x.get("method") == method]

        if method_logs:
            _, covs, risks = compute_cov_risk(method_logs)
            data[method] = {"coverage": covs, "risk": risks, "n": len(method_logs)}

    return data


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", required=True)
    parser.add_argument("--output", default="risk_coverage.json")
    args = parser.parse_args()

    data = generate_risk_coverage_data(args.log_dir)
    with open(args.output, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Risk-coverage data written to {args.output}")

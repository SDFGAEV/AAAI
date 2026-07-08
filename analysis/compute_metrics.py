#!/usr/bin/env python3
"""
Compute C-ACT evaluation metrics from experiment logs.

Reads C-ACT log files and computes all primary/secondary metrics
for paper tables and figures.

Usage:
  python analysis/compute_metrics.py --log_dir exp_results/cact_logs/ --output results.json
"""

import json, os, argparse, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cact.metrics import (compute_sr, compute_hardsr, compute_kus,
    compute_hrr, compute_irr, compute_coverage, compute_ece,
    compute_cov_risk, compute_failuresr, compute_interactionsr,
    compute_rcr, compute_cfr, compute_kpr, compute_csr, compute_cvr)


def load_jsonl(path):
    data = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    return data


def compute_all(log_dir: str, task_results_path: str = None):
    """Compute all C-ACT metrics from log directory."""
    episode = load_jsonl(os.path.join(log_dir, "episode", "episode.jsonl"))
    reuse = load_jsonl(os.path.join(log_dir, "reuse", "reuse_decision.jsonl"))
    base = load_jsonl(os.path.join(log_dir, "base", "base_logging.jsonl"))
    interaction = load_jsonl(os.path.join(log_dir, "interaction", "interaction.jsonl"))
    lifecycle = load_jsonl(os.path.join(log_dir, "lifecycle", "lifecycle.jsonl"))
    contracts = load_jsonl(os.path.join(log_dir, "contracts", "contracts.jsonl"))

    task_results = load_jsonl(task_results_path) if task_results_path else []

    results = {
        "primary": {},
        "secondary": {},
        "diagnostic": {},
        "counts": {},
    }

    # Primary
    results["primary"]["SR"] = round(compute_sr(episode), 4)
    results["primary"]["HardSR"] = round(compute_hardsr(task_results or episode), 4)
    results["primary"]["FailureSR"] = round(compute_failuresr(task_results or episode), 4)
    results["primary"]["InteractionSR"] = round(compute_interactionsr(task_results or episode), 4)
    results["primary"]["KUS"] = round(compute_kus(reuse), 4)
    results["primary"]["HRR"] = round(compute_hrr(reuse), 4)
    cov, _, _ = compute_cov_risk(reuse)
    results["primary"]["Coverage@Risk<=10%"] = round(cov, 4)

    # Secondary
    results["secondary"]["IRR"] = round(compute_irr(reuse), 4)
    results["secondary"]["Coverage"] = round(compute_coverage(reuse), 4)
    results["secondary"]["ECE"] = round(compute_ece(reuse), 4)
    results["secondary"]["RCR"] = round(compute_rcr(interaction), 4)
    results["secondary"]["CFR"] = round(compute_cfr(interaction), 4)
    results["secondary"]["KPR"] = round(compute_kpr(lifecycle), 4)

    # Diagnostic
    results["diagnostic"]["CSR"] = round(compute_csr(reuse), 4)
    results["diagnostic"]["CVR"] = round(compute_cvr(reuse), 4)

    # Counts
    results["counts"]["episodes"] = len(episode)
    results["counts"]["reuse_decisions"] = len(reuse)
    results["counts"]["base_logs"] = len(base)
    results["counts"]["interactions"] = len(interaction)
    results["counts"]["contracts"] = len(contracts)

    return results


def main():
    parser = argparse.ArgumentParser(description="Compute C-ACT metrics")
    parser.add_argument("--log_dir", required=True, help="Path to cact_logs directory")
    parser.add_argument("--task_results", default=None, help="Path to task-level results")
    parser.add_argument("--output", default="results.json", help="Output JSON path")
    args = parser.parse_args()

    results = compute_all(args.log_dir, args.task_results)

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

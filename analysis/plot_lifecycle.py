#!/usr/bin/env python3
"""
Generate Lifecycle evolution plot data for E5 online evaluation.

Shows knowledge state distribution across online evolution rounds.
"""

import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from analysis.compute_metrics import load_jsonl


def lifecycle_by_round(log_dir: str):
    """Extract lifecycle state counts per round."""
    path = os.path.join(log_dir, "lifecycle", "lifecycle.jsonl")
    logs = load_jsonl(path)

    rounds = {}
    for entry in logs:
        round_id = entry.get("round", entry.get("version_id", "0"))
        if round_id not in rounds:
            rounds[round_id] = {"candidate": 0, "quarantined": 0,
                                "probation": 0, "certified": 0,
                                "deprecated": 0, "disabled": 0}
        lc = entry.get("lifecycle", {})
        for state, count in lc.items():
            if state in rounds[round_id]:
                rounds[round_id][state] += count

    return rounds


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", required=True)
    parser.add_argument("--output", default="lifecycle_evolution.json")
    args = parser.parse_args()

    data = lifecycle_by_round(args.log_dir)
    with open(args.output, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Lifecycle data written to {args.output}")

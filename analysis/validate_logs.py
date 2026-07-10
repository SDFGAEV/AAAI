#!/usr/bin/env python3
"""Validate C-ACT JSONL artifacts before metric aggregation."""
import argparse
import json
from pathlib import Path

REQUIRED = {
    "episode": {"schema_version", "run_id", "episode_id", "task_id",
                "task_group", "seed", "success", "postcondition_pass",
                "harmful_reuse", "resource_conflict", "chain_success"},
    "reuse": {"schema_version", "run_id", "decision_id", "episode_id",
              "decision", "propensity", "harmful_reuse",
              "pre_admit_contract_pass", "postcondition_satisfied"},
}

def validate(path, kind):
    required = REQUIRED[kind]
    errors = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"{line_no}: invalid JSON ({exc})")
                continue
            missing = sorted(required - row.keys())
            if missing:
                errors.append(f"{line_no}: missing {','.join(missing)}")
    if errors:
        raise SystemExit("\n".join(errors[:20]))
    print(f"OK {path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=Path)
    ap.add_argument("--kind", choices=sorted(REQUIRED), default="episode")
    args = ap.parse_args()
    validate(args.path, args.kind)


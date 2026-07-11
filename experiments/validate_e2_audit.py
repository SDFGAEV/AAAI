#!/usr/bin/env python3
"""Validate the sealed D_audit direct-rollout and paired-branch artifacts."""
from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path

METHODS = {"Base", "NoGate", "Full", "Pointwise"}

def read(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]

def validate_rollouts(rows):
    cells = defaultdict(dict)
    required = {"task_id", "world_seed", "episode_id", "method", "snapshot_hash",
                "success", "harmful_reuse", "coverage", "hrr", "eahr"}
    for row in rows:
        missing = required - set(row)
        if missing: raise ValueError(f"audit rollout missing fields: {sorted(missing)}")
        if row["method"] not in METHODS: raise ValueError(f"unexpected audit method: {row['method']}")
        key = (str(row["task_id"]), str(row["world_seed"]), str(row["episode_id"]))
        if row["method"] in cells[key]: raise ValueError(f"duplicate audit cell: {key} {row['method']}")
        cells[key][row["method"]] = row
    incomplete = [key for key, values in cells.items() if set(values) != METHODS]
    if incomplete: raise ValueError(f"incomplete audit cells: {len(incomplete)}")
    for key, values in cells.items():
        if len({str(x["snapshot_hash"]) for x in values.values()}) != 1:
            raise ValueError(f"snapshot mismatch in audit cell: {key}")
    return len(cells)

def validate_pairs(rows):
    if len(rows) < 200: raise ValueError(f"sealed paired audit needs >=200 rows, got {len(rows)}")
    seen = set()
    for row in rows:
        if not {"pair_id", "parent_episode", "snapshot_hash", "reuse", "base"}.issubset(row):
            raise ValueError("paired audit row missing pair_id/parent_episode/snapshot_hash/reuse/base")
        if row["pair_id"] in seen: raise ValueError(f"duplicate paired audit id: {row['pair_id']}")
        seen.add(row["pair_id"])
        if row["reuse"].get("branch_parent_id") != row["parent_episode"] or row["base"].get("branch_parent_id") != row["parent_episode"]:
            raise ValueError(f"branch parent mismatch: {row['pair_id']}")
        if row["reuse"].get("branch_mode") != "reuse" or row["base"].get("branch_mode") != "base":
            raise ValueError(f"branch mode mismatch: {row['pair_id']}")
        if row.get("prefix_match") is not True: raise ValueError(f"unverified shared prefix: {row['pair_id']}")
    return len(rows)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--rollouts", required=True); ap.add_argument("--pairs", required=True); ap.add_argument("--out", required=True)
    args = ap.parse_args()
    result = {"schema_version": "cact.e2.audit.v1", "rollout_cells": validate_rollouts(read(args.rollouts)), "paired_rows": validate_pairs(read(args.pairs)), "passed": True}
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result)); return 0

if __name__ == "__main__":
    try: raise SystemExit(main())
    except (OSError, ValueError) as exc: print(f"[FAIL] {exc}"); raise SystemExit(2)

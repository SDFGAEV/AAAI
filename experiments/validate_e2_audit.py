#!/usr/bin/env python3
"""Validate the sealed D_audit direct-rollout and paired-branch artifacts."""
from __future__ import annotations
import argparse, json, math, sys
from collections import defaultdict
from pathlib import Path

_PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJ))
from experiments.world_identity import derive_snapshot_hash

METHODS = {"Base", "NoGate", "Full", "Pointwise"}

def read(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]

def validate_rollouts(rows):
    cells = defaultdict(dict)
    required = {"task_id", "world_seed", "episode_id", "matched_cell_id", "method",
                "success", "harmful_reuse", "coverage", "hrr", "eahr", "returncode", "run_id"}
    for row in rows:
        missing = required - set(row)
        if missing: raise ValueError(f"audit rollout missing fields: {sorted(missing)}")
        if row["method"] not in METHODS: raise ValueError(f"unexpected audit method: {row['method']}")
        row["snapshot_hash"] = str(row.get("snapshot_hash") or derive_snapshot_hash(row["task_id"], row["world_seed"]))
        key = (str(row["task_id"]), str(row["world_seed"]), str(row["matched_cell_id"]))
        if not str(row["matched_cell_id"]) or not str(row["snapshot_hash"]):
            raise ValueError(f"empty matched-cell or snapshot hash: {key}")
        if int(row["returncode"]) != 0:
            raise ValueError(f"failed audit rollout: {row.get('run_id', key)}")
        for name in ("success", "harmful_reuse", "coverage", "hrr", "eahr"):
            value = float(row[name])
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"audit metric {name} outside [0,1]: {row[name]}")
        if row["method"] in cells[key]: raise ValueError(f"duplicate audit cell: {key} {row['method']}")
        cells[key][row["method"]] = row
    incomplete = [key for key, values in cells.items() if set(values) != METHODS]
    if incomplete: raise ValueError(f"incomplete audit cells: {len(incomplete)}")
    for key, values in cells.items():
        if len({str(x["snapshot_hash"]) for x in values.values()}) != 1:
            raise ValueError(f"snapshot mismatch in audit cell: {key}")
        if len({str(x["episode_id"]) for x in values.values()}) != 1:
            raise ValueError(f"episode mismatch in audit cell: {key}")
    return len(cells)

def validate_pairs(rows):
    if len(rows) < 200: raise ValueError(f"sealed paired audit needs >=200 rows, got {len(rows)}")
    seen = set()
    for row in rows:
        reuse = row.get("reuse") if isinstance(row.get("reuse"), dict) else {}
        task = row.get("task_id", reuse.get("task_id")); seed = row.get("world_seed", reuse.get("world_seed"))
        if task is None or seed is None:
            raise ValueError(f"paired audit row lacks task_id/world_seed: {row.get('pair_id', 'unknown')}")
        row["snapshot_hash"] = str(row.get("snapshot_hash") or derive_snapshot_hash(task, seed))
        for branch in (row.get("reuse"), row.get("base")):
            if isinstance(branch, dict): branch["snapshot_hash"] = str(branch.get("snapshot_hash") or row["snapshot_hash"])
        if not {"pair_id", "parent_episode", "snapshot_hash", "reuse", "base"}.issubset(row):
            raise ValueError("paired audit row missing pair_id/parent_episode/snapshot_hash/reuse/base")
        if not str(row["pair_id"]) or not str(row["parent_episode"]) or not str(row["snapshot_hash"]):
            raise ValueError("paired audit identifiers and snapshot_hash must be non-empty")
        if not isinstance(row["reuse"], dict) or not isinstance(row["base"], dict):
            raise ValueError(f"paired audit branches must be objects: {row['pair_id']}")
        if row["pair_id"] in seen: raise ValueError(f"duplicate paired audit id: {row['pair_id']}")
        seen.add(row["pair_id"])
        if row["reuse"].get("branch_parent_id") != row["parent_episode"] or row["base"].get("branch_parent_id") != row["parent_episode"]:
            raise ValueError(f"branch parent mismatch: {row['pair_id']}")
        if row["reuse"].get("branch_mode") != "reuse" or row["base"].get("branch_mode") != "base":
            raise ValueError(f"branch mode mismatch: {row['pair_id']}")
        if row["reuse"].get("snapshot_hash") != row["base"].get("snapshot_hash") or row["reuse"].get("snapshot_hash") != row.get("snapshot_hash"):
            raise ValueError(f"snapshot mismatch in paired audit: {row['pair_id']}")
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

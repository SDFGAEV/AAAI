#!/usr/bin/env python3
"""Select E2 policies from real matched-risk rollouts.

The selector consumes a sealed JSONL export produced by running the same
episode/world for Base, seven Full and seven Pointwise candidates.  It never
creates counterfactual labels and fails closed when a matched cell is
incomplete.  Selection is performed on D_select only; the separate audit set
must be checked by the caller before deployment.
"""
from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path

KAPPAS = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
REQUIRED = {"Base"} | {f"Full:{k:g}" for k in KAPPAS} | {f"Pointwise:{k:g}" for k in KAPPAS}

def _cell(row):
    return (str(row.get("task_id", "")), str(row.get("world_seed", row.get("seed", ""))),
            str(row.get("episode_id", "")))

def load(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                missing = {"task_id", "world_seed", "episode_id", "method",
                           "success", "harmful_reuse"} - set(row)
                if missing:
                    raise ValueError(f"row missing required fields: {sorted(missing)}")
                rows.append(row)
    if not rows:
        raise ValueError("D_select direct rollout file is empty")
    return rows

def select(rows, eps_inc=0.02):
    cells = defaultdict(dict)
    for row in rows:
        key = _cell(row); method = str(row["method"])
        if method not in REQUIRED:
            raise ValueError(f"unexpected E2 method: {method}")
        if method in cells[key]:
            raise ValueError(f"duplicate matched cell: {key} {method}")
        cells[key][method] = row
    incomplete = [key for key, methods in cells.items() if set(methods) != REQUIRED]
    if incomplete:
        raise ValueError(f"incomplete matched-risk cells: {len(incomplete)}")
    chosen = {}
    for family in ("Full", "Pointwise"):
        candidates = []
        for kappa in KAPPAS:
            method = f"{family}:{kappa:g}"
            deltas, risks = [], []
            for methods in cells.values():
                base, cand = methods["Base"], methods[method]
                deltas.append(float(cand["success"]) - float(base["success"]))
                risks.append(float(cand["harmful_reuse"]) - float(base["harmful_reuse"]))
            mean_delta = sum(deltas) / len(deltas)
            mean_risk = sum(risks) / len(risks)
            if mean_risk <= eps_inc:
                candidates.append((mean_delta, -kappa, kappa, mean_risk))
        if not candidates:
            raise ValueError(f"no {family} candidate satisfies incremental risk bound")
        best = max(candidates)
        chosen[family.lower()] = {"kappa": best[2], "mean_delta_success": best[0],
                                  "mean_incremental_harm": best[3], "cells": len(cells)}
    return chosen

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--eps-inc", type=float, default=0.02)
    args = ap.parse_args()
    result = {"schema_version": "cact.e2.direct_select.v1", "source": str(Path(args.input)),
              "selection": select(load(args.input), args.eps_inc)}
    Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))

if __name__ == "__main__": main()

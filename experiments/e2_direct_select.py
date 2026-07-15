#!/usr/bin/env python3
"""Select E2 policies from real matched-risk rollouts.

The selector consumes a sealed JSONL export produced by running the same
episode/world for Base, seven Full and seven Pointwise candidates.  It never
creates counterfactual labels and fails closed when a matched cell is
incomplete.  Selection is performed on D_select only; the separate audit set
must be checked by the caller before deployment.
"""
from __future__ import annotations
import argparse, json, math, sys
from collections import Counter, defaultdict
from pathlib import Path

_PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJ))
from experiments.world_identity import derive_snapshot_hash

KAPPAS = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
REQUIRED = {"Base"} | {f"Full:{k:g}" for k in KAPPAS} | {f"Pointwise:{k:g}" for k in KAPPAS}

def _cell(row):
    return (str(row.get("task_id", "")), str(row.get("world_seed", row.get("seed", ""))),
            str(row.get("matched_cell_id", "")))

def load(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                missing = {"task_id", "world_seed", "episode_id", "matched_cell_id",
                           "method", "success", "harmful_reuse",
                           "store_hash", "run_id", "returncode", "coverage", "hrr", "eahr"} - set(row)
                if missing:
                    raise ValueError(f"row missing required fields: {sorted(missing)}")
                if not row.get("snapshot_hash"):
                    row["snapshot_hash"] = derive_snapshot_hash(row["task_id"], row["world_seed"])
                if not str(row["matched_cell_id"]) or not str(row["snapshot_hash"]):
                    raise ValueError("E2 rows require non-empty matched_cell_id and snapshot_hash")
                if not str(row["run_id"]) or not str(row["store_hash"]):
                    raise ValueError("E2 rows require non-empty run_id and store_hash")
                if int(row["returncode"]) != 0:
                    raise ValueError(f"E2 rollout failed: {row['run_id']}")
                for name in ("success", "harmful_reuse", "coverage", "hrr", "eahr"):
                    value = float(row[name])
                    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                        raise ValueError(f"E2 metric {name} outside [0,1]: {row[name]}")
                rows.append(row)
    if not rows:
        raise ValueError("D_select direct rollout file is empty")
    return rows

def select(rows, eps_inc=0.02, eps_abs=0.10):
    cells = defaultdict(dict)
    for row in rows:
        if row.get("returncode", 0) != 0:
            raise ValueError(f"failed rollout present in E2 table: {row.get('run_id', row)}")
        key = _cell(row); method = str(row["method"])
        if method not in REQUIRED:
            raise ValueError(f"unexpected E2 method: {method}")
        if method in cells[key]:
            raise ValueError(f"duplicate matched cell: {key} {method}")
        cells[key][method] = row
    for key, methods in cells.items():
        hashes = {str(row["snapshot_hash"]) for row in methods.values()}
        if len(hashes) != 1:
            raise ValueError(f"matched cell has inconsistent snapshot hashes: {key}")
        episode_ids = {str(row["episode_id"]) for row in methods.values()}
        if len(episode_ids) != 1:
            raise ValueError(f"matched cell has inconsistent episode IDs: {key}")
    cell_count = len(cells)
    if cell_count not in {48, 64}:
        raise ValueError(f"D_select must contain 48 cells (6 seeds) or 64 cells (8-seed expansion), got {cell_count}")
    seeds_per_template = Counter(key[0] for key in cells)
    expected_seeds = 6 if cell_count == 48 else 8
    if len(seeds_per_template) != 8 or set(seeds_per_template.values()) != {expected_seeds}:
        raise ValueError("D_select must be balanced across exactly 8 preregistered templates")
    incomplete = [key for key, methods in cells.items() if set(methods) != REQUIRED]
    if incomplete:
        raise ValueError(f"incomplete matched-risk cells: {len(incomplete)}")
    chosen = {}
    for family in ("Full", "Pointwise"):
        candidates = []
        for kappa in KAPPAS:
            method = f"{family}:{kappa:g}"
            deltas, inc_eahr, hrrs, coverages = [], [], [], []
            for methods in cells.values():
                base, cand = methods["Base"], methods[method]
                deltas.append(float(cand["success"]) - float(base["success"]))
                inc_eahr.append(float(cand["eahr"]) - float(base["eahr"]))
                hrrs.append(float(cand["hrr"]))
                coverages.append(float(cand["coverage"]))
            mean_delta = sum(deltas) / len(deltas)
            mean_inc_eahr = sum(inc_eahr) / len(inc_eahr)
            mean_hrr = sum(hrrs) / len(hrrs)
            mean_coverage = sum(coverages) / len(coverages)
            if mean_inc_eahr <= eps_inc and mean_hrr <= eps_abs:
                candidates.append((mean_coverage, mean_delta, -kappa, kappa,
                                   mean_inc_eahr, mean_hrr))
        if not candidates:
            raise ValueError(f"no {family} candidate satisfies incremental risk bound")
        best = max(candidates)
        chosen[family.lower()] = {"kappa": best[3], "mean_coverage": best[0],
                                  "mean_delta_success": best[1],
                                  "mean_incremental_eahr": best[4], "mean_hrr": best[5],
                                  "cells": len(cells)}
    # Global-Risk Only (§18): same 7 Full rollouts, overall constraints only.
    gr_candidates = []
    for kappa in KAPPAS:
        method = f"Full:{kappa:g}"
        all_hrr, all_eahr, all_delta, all_cov = [], [], [], []
        for methods in cells.values():
            base_row = methods["Base"]
            cand_row = methods[method]
            all_hrr.append(float(cand_row["hrr"]))
            all_eahr.append(float(cand_row["eahr"]) - float(base_row["eahr"]))
            all_delta.append(float(cand_row["success"]) - float(base_row["success"]))
            all_cov.append(float(cand_row["coverage"]))
        m_hrr = sum(all_hrr) / len(all_hrr)
        m_eahr = sum(all_eahr) / len(all_eahr)
        if m_hrr <= eps_abs and m_eahr <= eps_inc:
            gr_candidates.append((sum(all_cov)/len(all_cov), sum(all_delta)/len(all_delta),
                                  -kappa, kappa, m_eahr, m_hrr))
    if gr_candidates:
        best_gr = max(gr_candidates)
        chosen["global_only"] = {"kappa": best_gr[3], "mean_coverage": best_gr[0],
                                 "mean_delta_success": best_gr[1],
                                 "mean_incremental_eahr": best_gr[4], "mean_hrr": best_gr[5],
                                 "cells": len(cells)}
    return chosen

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--eps-inc", type=float, default=0.02)
    ap.add_argument("--eps-abs", type=float, default=0.10)
    args = ap.parse_args()
    rows = load(args.input)
    result = {"schema_version": "cact.e2.direct_select.v1", "source": str(Path(args.input)),
              "design": {"allowed_cells": [48, 64], "observed_cells": len({_cell(r) for r in rows})},
              "selection": select(rows, args.eps_inc, args.eps_abs)}
    Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))

if __name__ == "__main__": main()

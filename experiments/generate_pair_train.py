#!/usr/bin/env python3
"""Generate the sealed E1c paired-preference artifact from real rollouts.

This is deliberately a two-stage collector:
1. collect eligible opportunities with a deterministic pilot;
2. rerun each selected opportunity from the same store/snapshot in a reuse
   and a base branch, forcing only at that opportunity while keeping the
   prefix assignment fixed.

The command fails closed on missing snapshot hashes, branch reachability,
outcome labels, or fewer than 320 pairs.  It never synthesizes labels.
"""
from __future__ import annotations
import argparse, hashlib, json, os, shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.parallel_runner import ExperimentConfig, ParallelRunner
from experiments.world_identity import derive_snapshot_hash

def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists(): return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip(): out.append(json.loads(line))
    return out

def _pilot_candidates(log_root: Path, runs: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for run in runs:
        path = log_root / str(run["run_id"]) / "opportunities.jsonl"
        for row in _read_jsonl(path):
            if not row.get("eligible") or row.get("censor_flag"):
                continue
            if str(row.get("boundary_status", "")).lower() not in {"applicable", "eligible"}:
                continue
            if str(row.get("risk_tier", "")).lower() not in {"low", "medium"}:
                continue
            row["pilot_run_id"] = run["run_id"]
            row["task_idx"] = run["task_idx"]
            row["seed"] = run["seed"]
            row["snapshot_hash"] = str(row.get("snapshot_hash") or derive_snapshot_hash(row["task_idx"], row["seed"]))
            rows.append(row)
    # One opportunity per parent episode/target; stable hash ordering avoids
    # outcome-dependent selection.
    rows.sort(key=lambda r: hashlib.sha256(
        f"{r['task_idx']}|{r['seed']}|{r['opportunity_id']}".encode()).hexdigest())
    seen = set(); selected = []
    for row in rows:
        key = (row["task_idx"], row["seed"], row["episode_id"], row["opportunity_id"])
        if key in seen: continue
        seen.add(key); selected.append(row)
    return selected

def _cfg(benchmark: str, task_idx: int, seed: int, method: str,
         run_id: str, store: Path, snapshot: Path, parent: str,
         branch_mode: str = "", target: str = "", snapshot_hash: str = "",
         prefix_trace: str = "") -> ExperimentConfig:
    cfg = ExperimentConfig(task=str(task_idx), task_idx=task_idx, seed=seed,
                           method=method, benchmark=benchmark, vlm_port=12345,
                           mc_port=0, store_path=str(store), run_id=run_id,
                           snapshot_path=str(snapshot) if snapshot else "", protocol_path="collect",
                           branch_mode=branch_mode,
                           branch_target_opportunity=target,
                           branch_parent_id=parent,
                           branch_prefix_assignment=0, branch_prefix_trace=prefix_trace,
                           snapshot_hash=snapshot_hash)
    cfg.active_calib_rate = 0.5
    return cfg

def _label(row: Dict[str, Any], name: str) -> Dict[str, Any]:
    required = ("outcome_success", "harmful_reuse", "progress_delta",
                "steps", "resource_cost", "opportunity_id",
                "branch_parent_id", "branch_mode")
    missing = [k for k in required if row.get(k) is None]
    if missing:
        raise RuntimeError(f"{name} branch missing outcome fields: {missing}")
    return row

def _preferred(reuse: Dict[str, Any], base: Dict[str, Any]) -> int | None:
    # Manual tie-break order: Y, H (lower is better), progress, steps,
    # resource cost.  A complete tie is excluded, never labeled arbitrarily.
    a = (int(reuse["outcome_success"]), -int(bool(reuse["harmful_reuse"])),
         float(reuse["progress_delta"]), -float(reuse["steps"]), -float(reuse["resource_cost"]))
    b = (int(base["outcome_success"]), -int(bool(base["harmful_reuse"])),
         float(base["progress_delta"]), -float(base["steps"]), -float(base["resource_cost"]))
    if a == b: return None
    return 1 if a > b else 0

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="cact_train")
    ap.add_argument("--pilot-task-indices", required=True)
    ap.add_argument("--pilot-seeds", required=True)
    ap.add_argument("--target", type=int, default=320)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", default=str(ROOT / "exp_results/cact_pair_train/paired/pairs.jsonl"))
    ap.add_argument("--world-snapshot-manifest", default="", help="optional filesystem/procedural snapshot manifest")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    task_indices = [int(x) for x in args.pilot_task_indices.split(",") if x]
    if "-" in args.pilot_seeds:
        lo, hi = args.pilot_seeds.split("-"); seeds = list(range(int(lo), int(hi) + 1))
    else: seeds = [int(x) for x in args.pilot_seeds.split(",") if x]
    if args.target != 320:
        raise SystemExit("The preregistered E1c target is exactly 320 pairs")
    if args.dry_run:
        print(json.dumps({"pilot_episodes": len(task_indices) * len(seeds),
                          "target_pairs": args.target, "branch_episodes": args.target * 2}, ensure_ascii=False)); return
    if args.world_snapshot_manifest:
        manifest = json.loads(Path(args.world_snapshot_manifest).read_text(encoding="utf-8"))
        hashes = manifest.get("hashes", manifest)
        if not isinstance(hashes, dict): raise SystemExit("world snapshot manifest must be a JSON mapping")
    else:
        hashes = {f"{task}|{seed}": derive_snapshot_hash(task, seed) for task in task_indices for seed in seeds}
    root = ROOT / "exp_results" / "cact_pair_train"
    pilot_root = root / "pilot_stores"; log_root = ROOT / "exp_results" / "cact_logs"
    runner = ParallelRunner(workers=max(1, args.workers), vlm_port=12345)
    pilot_cfgs, pilot_meta = [], []
    for task_idx in task_indices:
        for seed in seeds:
            run_id = f"pair_pilot_seed{seed}_task{task_idx}"
            # Pilot creates the source store; snapshot and destination must
            # never be the same path because the runner clears the destination.
            pilot_cfgs.append(_cfg(args.benchmark, task_idx, seed, "C-ACT", run_id,
                                   pilot_root / run_id, None, run_id,
                                   snapshot_hash=str(hashes.get(f"{task_idx}|{seed}", ""))))
            pilot_meta.append({"run_id": run_id, "task_idx": task_idx, "seed": seed})
    runner.run(args.benchmark, seeds=seeds, methods=["C-ACT"], grid=pilot_cfgs)
    candidates = _pilot_candidates(log_root, pilot_meta)
    if len(candidates) < args.target:
        raise SystemExit(f"pilot produced {len(candidates)} eligible opportunities; need {args.target}")
    selected = candidates[:args.target]
    branch_cfgs, branch_meta = [], []
    for i, row in enumerate(selected):
        parent = f"pair_{i:04d}_{row['task_idx']}_{row['seed']}_{row['opportunity_id']}"
        snapshot = pilot_root / str(row["pilot_run_id"])
        if not snapshot.is_dir():
            raise RuntimeError(f"pilot snapshot missing: {snapshot}")
        pilot_opps = _read_jsonl(log_root / str(row["pilot_run_id"]) / "opportunities.jsonl")
        target_index = next((j for j, x in enumerate(pilot_opps)
                             if x.get("opportunity_id") == row["opportunity_id"]), None)
        if target_index is None: raise RuntimeError(f"pilot target not found: {row['opportunity_id']}")
        prefix_trace = json.dumps({str(x["opportunity_id"]): int(x["assignment"])
                                   for x in pilot_opps[:target_index]}, separators=(",", ":"))
        for mode in ("reuse", "base"):
            run_id = f"pair_{mode}_{i:04d}_seed{row['seed']}_task{row['task_idx']}"
            branch_cfgs.append(_cfg(args.benchmark, int(row["task_idx"]), int(row["seed"]),
                                    "C-ACT", run_id, root / "branch_stores" / run_id,
                                    snapshot, parent, mode, str(row["opportunity_id"]),
                                    snapshot_hash=str(row.get("snapshot_hash") or derive_snapshot_hash(row["task_idx"], row["seed"])),
                                    prefix_trace=prefix_trace))
            branch_meta.append((run_id, parent, mode, row))
    runner = ParallelRunner(workers=1, vlm_port=12345)
    runner.run(args.benchmark, seeds=seeds, methods=["C-ACT"], grid=branch_cfgs)
    pairs = []
    for i, row in enumerate(selected):
        got = {}
        parent = f"pair_{i:04d}_{row['task_idx']}_{row['seed']}_{row['opportunity_id']}"
        prefix_signatures = {}
        for run_id, p, mode, _ in branch_meta:
            if p != parent: continue
            opportunities = _read_jsonl(log_root / run_id / "opportunities.jsonl")
            target_index = next((j for j, x in enumerate(opportunities)
                                 if x.get("opportunity_id") == row["opportunity_id"]), None)
            if target_index is None:
                raise RuntimeError(f"{run_id}: target opportunity absent from opportunity log")
            keys = ("opportunity_id", "task_id", "world_seed", "raw_text_hash",
                    "task_group", "failure_type", "risk_tier", "resource_scarcity",
                    "boundary_status", "inventory_signature", "episode_phase",
                    "prior_admission_bin", "prior_fallback_bin")
            prefix_signatures[mode] = [tuple(x.get(k) for k in keys)
                                       for x in opportunities[:target_index]]
            logs = _read_jsonl(log_root / run_id / "reuse" / "reuse_decision.jsonl")
            target_rows = [x for x in logs if x.get("opportunity_id") == row["opportunity_id"]]
            if len(target_rows) != 1: raise RuntimeError(f"{run_id}: target opportunity not reached exactly once")
            branch_row = _label(target_rows[0], mode)
            if branch_row.get("branch_parent_id") != parent or branch_row.get("branch_mode") != mode:
                raise RuntimeError(f"{run_id}: branch identity mismatch")
            got[mode] = branch_row
        if set(got) != {"reuse", "base"}: raise RuntimeError(f"missing paired branch for {parent}")
        if prefix_signatures.get("reuse") != prefix_signatures.get("base"):
            raise RuntimeError(f"shared-prefix mismatch for {parent}")
        pref = _preferred(got["reuse"], got["base"])
        if pref is None: continue
        pairs.append({"schema_version": "cact.pair.v1", "pair_id": parent,
                      "parent_episode": parent, "preferred": pref,
                      "snapshot_hash": row["snapshot_hash"], "world_seed": row["world_seed"],
                      "task_id": row["task_id"], "source": row["source"], "type": row["type"],
                      "task_group": row["task_group"], "failure_type": row.get("failure_type", "none"),
                      "risk_tier": row["risk_tier"], "resource_scarcity": row.get("resource_scarcity", "ordinary"),
                      "boundary_status": row["boundary_status"], "episode_phase": row.get("episode_phase", "early"),
                      "prior_admission_bin": row.get("prior_admission_bin", "0"),
                      "prior_fallback_bin": row.get("prior_fallback_bin", "0"),
                      "reuse_outcome_success": got["reuse"]["outcome_success"],
                      "base_outcome_success": got["base"]["outcome_success"],
                      "reuse_harmful": got["reuse"]["harmful_reuse"],
                      "base_harmful": got["base"]["harmful_reuse"]})
    if len(pairs) != args.target:
        raise SystemExit(f"complete paired labels: {len(pairs)}; expected {args.target} (ties are excluded)")
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text("\n".join(json.dumps(x, ensure_ascii=False, sort_keys=True) for x in pairs) + "\n", encoding="utf-8")
    os.replace(tmp, out)
    print(json.dumps({"pairs": len(pairs), "out": str(out)}, ensure_ascii=False))

if __name__ == "__main__": main()

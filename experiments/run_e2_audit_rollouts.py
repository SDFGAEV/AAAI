#!/usr/bin/env python3
"""Collect real D_audit direct and paired artifacts.

Direct audit cells contain Base, NoGate, and the selected Full/Pointwise
policies.  Paired audit data is collected by the sealed branch generator and
then enriched from its raw branch logs; no labels are synthesized.
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.run_e2_select_rollouts import _ensure_vlm, _run_one
from experiments.world_identity import derive_snapshot_hash

def _seeds(value):
    if "-" in value:
        lo, hi = [int(x) for x in value.split("-", 1)]
        return list(range(lo, hi + 1))
    return [int(x) for x in value.split(",") if x.strip()]

def _selected_kappas(policy_path: Path):
    data = json.loads(policy_path.read_text(encoding="utf-8"))
    families = data.get("families", data)
    return float(families["full"]["kappa"]), float(families["pointwise"]["kappa"])

def _collect_direct(args, tasks, seeds, hashes, full_kappa, point_kappa):
    methods = [("Base", "Base"), ("NoGate", "NoGate"),
               ("Full", f"Full:{full_kappa:g}"),
               ("Pointwise", f"Pointwise:{point_kappa:g}")]
    cfg = {"vlm_port": args.vlm_port, "benchmark": args.benchmark,
           "snapshot_path": args.snapshot_path, "store_path": args.snapshot_path,
           "world_snapshot_hashes": hashes, "protocol_path": args.protocol_path,
           "plan_model": args.plan_model, "timeout": args.timeout}
    jobs = [(task, seed, label, internal) for task in tasks for seed in seeds for label, internal in methods]
    owned = _ensure_vlm(args.vlm_port, args.plan_model)
    rows = []
    try:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = {pool.submit(_run_one, t, s, internal, cfg): (t, s, label) for t, s, label, internal in jobs}
            for future in as_completed(futures):
                row = future.result(); _, _, label = futures[future]
                row["method"] = label
                rows.append(row)
    finally:
        if owned is not None:
            owned.terminate(); owned.wait(timeout=10)
    expected = len(tasks) * len(seeds) * len(methods)
    if len(rows) != expected:
        raise RuntimeError(f"D_audit direct collection incomplete: {len(rows)}/{expected}")
    return rows

def _branch_row(parent: str, mode: str):
    candidates = []
    for path in (ROOT / "exp_results" / "cact_logs").glob("pair_*/reuse/reuse_decision.jsonl"):
        try:
            rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
        except (OSError, json.JSONDecodeError):
            continue
        hit = [r for r in rows if r.get("branch_parent_id") == parent and r.get("reason") == f"paired_branch_{mode}"]
        if hit:
            candidates.append((path.parent.parent.name, hit[0], rows))
    if len(candidates) != 1:
        raise RuntimeError(f"expected one raw {mode} branch for {parent}, found {len(candidates)}")
    run_id, row, reuse_rows = candidates[0]
    base_path = ROOT / "exp_results" / "cact_logs" / run_id.replace("pair_reuse_", "pair_base_") / "reuse" / "reuse_decision.jsonl"
    if not base_path.exists():
        raise RuntimeError(f"base branch log missing for {parent}")
    base_rows = [json.loads(x) for x in base_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    return run_id, row, reuse_rows, base_rows

def _enrich_pairs(flat_path: Path, out_path: Path):
    flat = [json.loads(x) for x in flat_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    out = []
    for row in flat:
        parent = str(row["parent_episode"])
        reuse_run, reuse_row, reuse_rows, base_rows = _branch_row(parent, "reuse")
        base_candidates = [r for r in base_rows if r.get("branch_parent_id") == parent and r.get("reason") == "paired_branch_base"]
        if len(base_candidates) != 1:
            raise RuntimeError(f"expected one base target row for {parent}")
        base_row = base_candidates[0]
        if reuse_row.get("snapshot_hash") != base_row.get("snapshot_hash") or reuse_row.get("snapshot_hash") != row.get("snapshot_hash"):
            raise RuntimeError(f"snapshot mismatch for audit pair {parent}")
        out.append({"schema_version": "cact.e2.audit_pair.v1", "pair_id": row["pair_id"],
                    "parent_episode": parent, "snapshot_hash": row["snapshot_hash"],
                    "reuse": reuse_row, "base": base_row, "prefix_match": True,
                    "source_run_id": reuse_run})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(json.dumps(x, ensure_ascii=False, sort_keys=True) for x in out) + "\n", encoding="utf-8")
    return len(out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="cact_p3")
    ap.add_argument("--task-indices", required=True)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--snapshot-path", required=True)
    ap.add_argument("--world-snapshot-manifest", default="", help="optional filesystem/procedural snapshot manifest")
    ap.add_argument("--protocol-path", required=True)
    ap.add_argument("--policy-path", required=True)
    ap.add_argument("--out-rollouts", required=True)
    ap.add_argument("--out-pairs", required=True)
    ap.add_argument("--pair-target", type=int, default=200)
    ap.add_argument("--pair-task-indices", default="")
    ap.add_argument("--pair-seeds", default="")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--vlm-port", type=int, default=12345)
    ap.add_argument("--timeout", type=int, default=240)
    ap.add_argument("--plan-model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    args = ap.parse_args()
    tasks = [int(x) for x in args.task_indices.split(",") if x.strip()]
    seeds = _seeds(args.seeds)
    if args.world_snapshot_manifest:
        manifest = json.loads(Path(args.world_snapshot_manifest).read_text(encoding="utf-8"))
        hashes = {str(k): str(v) for k, v in manifest.get("hashes", manifest).items()}
        expected = {f"{task}|{seed}" for task in tasks for seed in seeds}
        missing = sorted(expected - {k for k, v in hashes.items() if v})
        if missing: raise SystemExit(f"world snapshot manifest missing {len(missing)} required cells")
    else:
        hashes = {f"{task}|{seed}": derive_snapshot_hash(task, seed) for task in tasks for seed in seeds}
    full_kappa, point_kappa = _selected_kappas(Path(args.policy_path))
    rows = _collect_direct(args, tasks, seeds, hashes, full_kappa, point_kappa)
    out_rollouts = Path(args.out_rollouts); out_rollouts.parent.mkdir(parents=True, exist_ok=True)
    out_rollouts.write_text("\n".join(json.dumps(x, ensure_ascii=False, sort_keys=True) for x in rows) + "\n", encoding="utf-8")
    pair_tasks = args.pair_task_indices or args.task_indices
    pair_seeds = args.pair_seeds or args.seeds
    flat = ROOT / "exp_results" / "e2_audit_pairs_flat.jsonl"
    cmd = [sys.executable, str(ROOT / "experiments/generate_pair_train.py"),
           "--benchmark", args.benchmark, "--pilot-task-indices", pair_tasks,
           "--pilot-seeds", pair_seeds, "--target", str(args.pair_target), "--workers", "1",
           "--out", str(flat)]
    if args.world_snapshot_manifest:
        cmd[cmd.index("--out"):cmd.index("--out")] = ["--world-snapshot-manifest", args.world_snapshot_manifest]
    subprocess.run(cmd, check=True, cwd=ROOT)
    pairs = _enrich_pairs(flat, Path(args.out_pairs))
    print(json.dumps({"rollouts": len(rows), "paired_audit": pairs, "out_rollouts": args.out_rollouts, "out_pairs": args.out_pairs}))

if __name__ == "__main__":
    main()

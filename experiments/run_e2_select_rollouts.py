#!/usr/bin/env python3
"""Run E2 matched-risk direct rollouts: 15 policies on D_select task-seed pairs.

Each task-seed is run 15 times (Base + 7 Full κ + 7 Pointwise κ) with the
same world seed, store snapshot, and environment budget.  Results go into a
single JSONL consumed by e2_direct_select.py.

Usage:
  python experiments/run_e2_select_rollouts.py \
    --benchmark cact_calib --task-indices 0,1,2,3,4,5,6,7 \
    --seeds 3001-3008 --workers 2 --vlm-port 12345 \
    --store-path cact_ckpt/trust_store --protocol-path protocol_release/policy.json \
    --out exp_results/e2_select_rollouts.jsonl
"""
from __future__ import annotations
import argparse, hashlib, json, os, shutil, subprocess, sys, time, urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

_PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJ))
from experiments.world_identity import derive_snapshot_hash

KAPPAS = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0)

def _ensure_vlm(port: int, model: str):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2):
            return None
    except Exception:
        pass
    log = _PROJ / "exp_results" / f"e2_vlm_{port}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    handle = log.open("a", encoding="utf-8")
    proc = subprocess.Popen([sys.executable, str(_PROJ / "app.py"), "--port", str(port), "--plan_model", model],
                            stdout=handle, stderr=handle, cwd=str(_PROJ),
                            env={**os.environ, "PYTHONUNBUFFERED": "1"})
    deadline = time.time() + float(os.environ.get("CACT_VLM_STARTUP_WAIT", "120"))
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2):
                handle.close(); return proc
        except Exception:
            time.sleep(1)
    proc.terminate(); handle.close()
    raise RuntimeError(f"VLM server on port {port} did not become healthy")


def _hash_tree(path: Path) -> str:
    h = hashlib.sha256()
    for fp in sorted(p for p in path.rglob("*") if p.is_file()):
        h.update(str(fp.relative_to(path)).encode())
        h.update(fp.read_bytes())
    return h.hexdigest()

def _clone_store(source: Path, destination: Path) -> str:
    if not source.is_dir():
        raise FileNotFoundError(f"E2 frozen snapshot not found: {source}")
    if destination.exists(): shutil.rmtree(destination)
    if os.environ.get("CACT_FROZEN_HARDLINK") == "1" and os.environ.get("CACT_ALLOW_UNSAFE_HARDLINK") == "1":
        destination.mkdir(parents=True)
        for root, _, files in os.walk(source):
            rel = Path(root).relative_to(source); target = destination / rel
            target.mkdir(parents=True, exist_ok=True)
            for name in files: os.link(Path(root) / name, target / name)
    else: shutil.copytree(source, destination)
    return _hash_tree(source)

def _run_one(task_idx: int, seed: int, method: str, cfg: dict) -> dict:
    """Run one frozen cell with an isolated store and common episode ID."""
    actual_method = ("NoKnowledge" if method == "Base" else
                     "NoGate" if method == "NoGate" else
                     "C-ACT-Pointwise" if method.startswith("Pointwise:") else "C-ACT")
    cell_id = f"e2_cell_seed{seed}_task{task_idx}"
    run_id = f"e2_{method.replace(':', '_')}_seed{seed}_task{task_idx}"
    source = Path(cfg.get("snapshot_path") or cfg.get("store_path", ""))
    run_store = _PROJ / "exp_results" / "e2_stores" / run_id
    store_hash = _clone_store(source, run_store)
    cell_key = f"{task_idx}|{seed}"
    world_hash = cfg.get("world_snapshot_hashes", {}).get(cell_key)
    if not world_hash: raise RuntimeError(f"missing world identity for E2 cell {cell_key}")
    cmd = [sys.executable, "-m", "optimus1.main_planning",
           f"server.port={cfg['vlm_port']}", "server.url=http://127.0.0.1",
           f"benchmark={cfg['benchmark']}", f"+evaluate=[{task_idx}]",
           "env.times=1", f"seed={seed}", f"world_seed={seed}", "prefix=cact_e2",
           f"+cact_method={actual_method}", f"+cact_store_path={run_store}",
           f"+cact_run_id={run_id}", "+cact_frozen=true",
           f"+cact_snapshot_hash={world_hash}",
           f"plan_model={cfg.get('plan_model', 'Qwen/Qwen2.5-VL-7B-Instruct')}" ]
    if cfg.get("protocol_path") and actual_method.startswith("C-ACT"):
        cmd.append(f"+cact_protocol_path={cfg['protocol_path']}")
        cmd.append(f"+cact_kappa={method.split(':', 1)[1]}")
    t0 = time.perf_counter(); log_dir = _PROJ / "exp_results" / "runner_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path, stderr_path = log_dir / f"{run_id}.stdout.log", log_dir / f"{run_id}.stderr.log"
    try:
        with stdout_path.open("w", encoding="utf-8") as out, stderr_path.open("w", encoding="utf-8") as err:
            result = subprocess.run(cmd, stdout=out, stderr=err, text=True,
                                    timeout=cfg.get("timeout", 240), cwd=str(_PROJ),
                                    env={**os.environ, "PYTHONUNBUFFERED": "1",
                                         "PYTHONPATH": os.pathsep.join([str(_PROJ), str(_PROJ / "src"), str(_PROJ / "minerl")])})
        rc = result.returncode
    except subprocess.TimeoutExpired: rc = 124
    episode_file = _PROJ / "exp_results" / "cact_logs" / run_id / "episode" / "episode.jsonl"
    reuse_file = _PROJ / "exp_results" / "cact_logs" / run_id / "reuse" / "reuse_decision.jsonl"
    out_success = out_harm = None; coverage = hrr = eahr = None
    if episode_file.exists():
        lines = [json.loads(x) for x in episode_file.read_text(encoding="utf-8").splitlines() if x.strip()]
        if lines: out_success = int(bool(lines[-1].get("success", False)))
    decision_rows = []
    if reuse_file.exists():
        decision_rows = [json.loads(x) for x in reuse_file.read_text(encoding="utf-8").splitlines() if x.strip()]
    admitted = [x for x in decision_rows if x.get("decision") in ("reuse", "probe")]
    harmful = [x for x in admitted if bool(x.get("harmful_reuse", x.get("is_harmful", False)))]
    coverage = len(admitted) / len(decision_rows) if decision_rows else 0.0
    hrr = len(harmful) / len(admitted) if admitted else 0.0
    eahr = int(bool(harmful)); out_harm = eahr
    if rc != 0 or out_success is None or out_harm is None:
        raise RuntimeError(f"E2 rollout failed/incomplete: {run_id} rc={rc}")
    return {"task_id": str(task_idx), "world_seed": seed, "episode_id": cell_id,
            "matched_cell_id": cell_id, "method": method, "success": out_success,
            "harmful_reuse": out_harm, "snapshot_hash": world_hash,
            "store_hash": store_hash, "coverage": coverage, "hrr": hrr, "eahr": eahr,
            "run_id": run_id, "returncode": rc, "elapsed_sec": round(time.perf_counter()-t0, 1)}


def _methods():
    """Generate the 15 method names: Base + 7 Full κ + 7 Pointwise κ."""
    yield "Base"
    for k in KAPPAS:
        yield f"Full:{k:g}"
    for k in KAPPAS:
        yield f"Pointwise:{k:g}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="cact_calib")
    ap.add_argument("--task-indices", required=True)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--vlm-port", type=int, default=12345)
    ap.add_argument("--store-path", default="", help="legacy alias for frozen snapshot path")
    ap.add_argument("--snapshot-path", default="", help="immutable frozen store snapshot")
    ap.add_argument("--world-snapshot-manifest", default="",
                    help="optional filesystem/procedural snapshot manifest")
    ap.add_argument("--protocol-path", default="")
    ap.add_argument("--calibration-path", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--plan-model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    args = ap.parse_args()

    task_indices = [int(x) for x in args.task_indices.split(",") if x]
    if "-" in args.seeds:
        lo, hi = args.seeds.split("-")
        seeds = list(range(int(lo), int(hi) + 1))
    else:
        seeds = [int(x) for x in args.seeds.split(",") if x]

    total = len(task_indices) * len(seeds) * 15
    print(json.dumps({"benchmark": args.benchmark, "tasks": len(task_indices),
                      "seeds": len(seeds), "methods": 15, "episodes": total}))

    if args.world_snapshot_manifest:
        manifest = json.loads(Path(args.world_snapshot_manifest).read_text(encoding="utf-8"))
        hashes = manifest.get("hashes", manifest)
        if not isinstance(hashes, dict): raise SystemExit("world snapshot manifest must be a JSON mapping")
        expected = {f"{task}|{seed}" for task in task_indices for seed in seeds}
        missing = sorted(expected - {str(k) for k, v in hashes.items() if str(v)})
        if missing: raise SystemExit(f"world snapshot manifest missing {len(missing)} required cells")
    else:
        hashes = {f"{task}|{seed}": derive_snapshot_hash(task, seed) for task in task_indices for seed in seeds}
    cfg = {"vlm_port": args.vlm_port, "benchmark": args.benchmark,
           "store_path": args.store_path, "snapshot_path": args.snapshot_path,
           "world_snapshot_hashes": {str(k): str(v) for k, v in hashes.items()},
           "protocol_path": args.protocol_path,
           "calibration_path": args.calibration_path, "plan_model": args.plan_model}

    from concurrent.futures import ThreadPoolExecutor, as_completed
    owned_vlm = _ensure_vlm(args.vlm_port, args.plan_model)
    all_results = []
    try:
        with ThreadPoolExecutor(max_workers=min(args.workers, os.cpu_count() or 2)) as pool:
            futures = {}
            for ti in task_indices:
                for s in seeds:
                    for method in _methods():
                        futures[pool.submit(_run_one, ti, s, method, cfg)] = (ti, s, method)
            for f in as_completed(futures):
                result = f.result()
                all_results.append(result)
                ti, s, method = futures[f]
                print(f"  [{method}] task={ti} seed={s} success={result['success']} "
                      f"harm={result['harmful_reuse']} rc={result['returncode']} "
                      f"({result['elapsed_sec']:.0f}s)")
    finally:
        if owned_vlm is not None:
            owned_vlm.terminate()
            try: owned_vlm.wait(timeout=10)
            except subprocess.TimeoutExpired: owned_vlm.kill()

    # Validate completeness
    cells = defaultdict(dict)
    for r in all_results:
        cells[(r["task_id"], r["world_seed"])][r["method"]] = r
    expected_methods = set(_methods())
    incomplete = [(k, sorted(expected_methods - set(v.keys())))
                  for k, v in cells.items() if set(v.keys()) != expected_methods]
    if incomplete:
        raise SystemExit(f"incomplete matched-risk cells: {len(incomplete)}")
    print(f"[OK] all {len(cells)} task-seed cells complete with 15 methods each")
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in all_results) + "\n", encoding="utf-8")
    os.replace(tmp, out)
    print(json.dumps({"out": str(out), "rows": len(all_results)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

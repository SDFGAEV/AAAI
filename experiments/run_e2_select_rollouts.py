#!/usr/bin/env python3
"""Run E2 matched-risk direct rollouts: 15 policies on D_select task-seed pairs.

Each task-seed is run 15 times (Base + 7 Full κ + 7 Pointwise κ) with the
same world seed, store snapshot, and environment budget.  Results go into a
single JSONL consumed by e2_direct_select.py.

Usage:
  python experiments/run_e2_select_rollouts.py \
    --benchmark cact_calib --task-indices 0,1,2,3,4,5 \
    --seeds 3001-3008 --workers 2 --vlm-port 12345 \
    --store-path cact_ckpt/trust_store --protocol-path protocol_release/policy.json \
    --out exp_results/e2_select_rollouts.jsonl
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

_PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJ))

KAPPAS = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0)


def _run_one(task_idx: int, seed: int, method: str, cfg: dict) -> dict:
    """Run a single episode and parse its success/harm from logs."""
    cmd = [
        sys.executable, "-m", "optimus1.main_planning",
        f"server.port={cfg['vlm_port']}",
        f"server.url=http://127.0.0.1",
        f"benchmark={cfg['benchmark']}",
        f"+evaluate=[{task_idx}]",
        f"env.times=1",
        f"seed={seed}",
        f"prefix=cact_e2",
        f"+cact_method=C-ACT",
        f"plan_model={cfg.get('plan_model', 'Qwen/Qwen2.5-VL-7B-Instruct')}",
        f"+cact_run_id=e2_{method}_seed{seed}_task{task_idx}",
    ]
    if cfg.get("store_path"):
        cp = Path(cfg["store_path"])
        cmd.append(f"+cact_store_path={cp}")
    if cfg.get("protocol_path"):
        cmd.append(f"+cact_protocol_path={cfg['protocol_path']}")
    if cfg.get("calibration_path"):
        cmd.append(f"+cact_calibration_path={cfg['calibration_path']}")
    if method.startswith("Full"):
        cmd.append(f"+cact_kappa={method.split(':')[1]}")
    elif method.startswith("Pointwise"):
        cmd.append(f"+cact_kappa={method.split(':')[1]}")
    if method != "Base":
        cmd.append("+cact_frozen=true")

    t0 = time.perf_counter()
    result = subprocess.run(
        cmd, capture_output=True, text=True,
        timeout=cfg.get("timeout", 240),  # aligned with parallel_runner 180+60
        cwd=str(_PROJ),
        env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONPATH": os.pathsep.join(
            [str(_PROJ), str(_PROJ / "src"), str(_PROJ / "minerl")])},
    )
    elapsed = time.perf_counter() - t0
    success = result.returncode == 0

    # Parse success/harm from the episode log
    out_success = 0
    out_harm = 0
    log_dir = _PROJ / "exp_results" / "cact_logs" / f"e2_{method}_seed{seed}_task{task_idx}"
    episode_file = log_dir / "episode" / "episode.jsonl"
    if episode_file.exists():
        try:
            lines = [json.loads(l) for l in episode_file.read_text(encoding="utf-8").splitlines() if l.strip()]
            if lines:
                last = lines[-1]
                out_success = int(bool(last.get("success", False)))
                out_harm = int(bool(last.get("harmful_reuse", last.get("is_harmful", False))))
        except (OSError, json.JSONDecodeError):
            pass

    return {
        "task_id": str(task_idx), "world_seed": seed,
        "episode_id": f"e2_{method}_s{seed}_t{task_idx}",
        "method": method, "success": out_success,
        "harmful_reuse": out_harm, "returncode": result.returncode,
        "elapsed_sec": round(elapsed, 1),
    }


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
    ap.add_argument("--store-path", default="")
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

    cfg = {"vlm_port": args.vlm_port, "benchmark": args.benchmark,
           "store_path": args.store_path, "protocol_path": args.protocol_path,
           "calibration_path": args.calibration_path, "plan_model": args.plan_model}

    from concurrent.futures import ThreadPoolExecutor, as_completed
    all_results = []
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

    # Validate completeness
    cells = defaultdict(dict)
    for r in all_results:
        cells[(r["task_id"], r["world_seed"])][r["method"]] = r
    expected_methods = set(_methods())
    incomplete = [(k, sorted(expected_methods - set(v.keys())))
                  for k, v in cells.items() if set(v.keys()) != expected_methods]
    if incomplete:
        print(f"[FAIL] incomplete cells: {len(incomplete)}")
    else:
        print(f"[OK] all {len(cells)} task-seed cells complete with 15 methods each")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in all_results) + "\n",
                   encoding="utf-8")
    print(json.dumps({"out": str(out), "rows": len(all_results)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

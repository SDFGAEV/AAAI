#!/usr/bin/env python3
"""
C-ACT Parallel Experiment Runner.

Turns the serial nested-loop experiment into a multi-process parallel pipeline:

  1. MC Instance Pool — N Minecraft servers on different ports
  2. Shared VLM Server  — single Qwen-VL server, GPU-batched inference
  3. Worker Pool         — N workers, each paired with one MC instance
  4. Checkpoint/Resume   — auto-saves progress, skip completed (task,seed) pairs
  5. Result Aggregation  — collects all logs into unified exp_results/

Usage:
  # E3 main evaluation (36 tasks × 8 seeds × 7 methods = 2016 episodes)
  python experiments/parallel_runner.py \\
    --benchmark cact_p3 --seeds 4001-4008 --methods \\
    NoKnowledge XENON-Original BankCuration LifecycleSuccessGate FixedBayes ACT C-ACT-Full \\
    --workers 4 --vlm_port 12345 --mc_base_port 15000

  # Resume interrupted run
  python experiments/parallel_runner.py --resume --checkpoint_dir exp_results/ckpt/

Speed: ~4× with 4 workers. E3: 17h → 4-5h (single MC per worker).
       ~8× with 8 workers + MC instance sharing.
"""

import subprocess, sys, os, json, time, signal, shutil, argparse, socket
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from pathlib import Path

_PROJ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# ── Defaults ──
DEFAULT_METHODS = ["NoKnowledge", "XENON-Original", "BankCuration",
                   "LifecycleSuccessGate", "FixedBayes", "ACT", "C-ACT-Full"]
DEFAULT_SEEDS = [4001, 4002, 4003, 4004, 4005, 4006, 4007, 4008]


@dataclass
class ExperimentConfig:
    """One (task, seed, method) combination."""
    task: str
    task_idx: int
    seed: int
    method: str
    benchmark: str
    vlm_port: int
    mc_port: int
    plan_model: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    timeout: int = 300
    env_times: int = 1
    prefix: str = "cact"


class ParallelRunner:
    """Orchestrate parallel experiment execution."""

    def __init__(self, workers: int = 4, vlm_port: int = 12345,
                 mc_base_port: int = 15000, checkpoint_dir: str = None,
                 batch_proxy_port: int = 12346):
        self.workers = workers
        self.vlm_port = vlm_port
        self.mc_base_port = mc_base_port
        self.batch_proxy_port = batch_proxy_port
        self.checkpoint_dir = checkpoint_dir or os.path.join(
            _PROJ, "exp_results", "ckpt")
        self._completed: set = set()
        self._results: List[Dict] = []
        self._server_procs: Dict[int, subprocess.Popen] = {}
        self._proxy_thread = None

        os.makedirs(self.checkpoint_dir, exist_ok=True)

    # ── Port management ──
    @staticmethod
    def _port_free(port: int) -> bool:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", port))
                return True
        except OSError:
            return False

    @staticmethod
    def _find_free_port(start: int = 15000, max_attempts: int = 100) -> int:
        for p in range(start, start + max_attempts):
            if ParallelRunner._port_free(p):
                return p
        raise RuntimeError(f"No free ports in range {start}-{start+max_attempts}")

    # ── Checkpoint / resume ──
    def _load_checkpoint(self):
        """Load previously completed (task, seed, method) pairs."""
        ckpt_file = os.path.join(self.checkpoint_dir, "completed.json")
        if os.path.exists(ckpt_file):
            try:
                with open(ckpt_file) as f:
                    data = json.load(f)
                self._completed = {tuple(e) for e in data.get("completed", [])}
                self._results = data.get("results", [])
                print(f"[Resume] Loaded {len(self._completed)} completed runs, "
                      f"{len(self._results)} results")
            except Exception as e:
                print(f"[Resume] Failed to load checkpoint: {e}")

    def _save_checkpoint(self):
        """Save progress to checkpoint file."""
        ckpt_file = os.path.join(self.checkpoint_dir, "completed.json")
        with open(ckpt_file, "w") as f:
            json.dump({
                "completed": [list(c) for c in self._completed],
                "results": self._results,
                "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }, f, indent=2)

    # ── Server lifecycle ──
    def _start_vlm_server(self, plan_model: str):
        """Start the VLM planning server (shared across all workers)."""
        if not self._port_free(self.vlm_port):
            print(f"[Warning] VLM port {self.vlm_port} in use — reusing existing server")
            return

        cmd = [
            sys.executable, os.path.join(_PROJ, "app.py"),
            "--port", str(self.vlm_port),
            "--plan_model", plan_model,
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL)
        self._server_procs[self.vlm_port] = proc
        print(f"[VLM] Started planning server on port {self.vlm_port} (PID {proc.pid})")
        time.sleep(8)  # Wait for model to load into GPU

    def _stop_vlm_server(self):
        """Stop the VLM planning server."""
        if self.vlm_port in self._server_procs:
            proc = self._server_procs[self.vlm_port]
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
            print(f"[VLM] Stopped planning server (PID {proc.pid})")

    def _start_batch_proxy(self):
        """Start the batch VLM proxy for 2-4x GPU throughput."""
        from experiments.batch_proxy import start_proxy_in_thread
        self._proxy_thread = start_proxy_in_thread(
            self.batch_proxy_port,
            f"http://127.0.0.1:{self.vlm_port}",
        )
        print(f"[BatchProxy] Started on port {self.batch_proxy_port}")

    def _stop_batch_proxy(self):
        """The proxy thread is daemon — stops automatically on exit."""

    # ── Single episode execution ──
    def _run_one(self, cfg: ExperimentConfig) -> Dict:
        """Run a single (task, seed, method) episode via subprocess."""
        key = (cfg.task, cfg.seed, cfg.method)
        if key in self._completed:
            return {"key": key, "status": "skipped", "reason": "already_completed"}

        cmd = [
            sys.executable, "-m", "optimus1.main_planning",
            f"server.port={cfg.vlm_port}",
            f"server.url=http://127.0.0.1",
            f"benchmark={cfg.benchmark}",
            f"evaluate=[{cfg.task_idx}]",
            f"env.times={cfg.env_times}",
            f"seed={cfg.seed}",
            f"prefix={cfg.prefix}",
            f"+cact_method={cfg.method}",
            f"plan_model={cfg.plan_model}",
        ]

        t0 = time.perf_counter()
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=cfg.timeout * cfg.env_times + 120,
                cwd=_PROJ,
                env={**os.environ, "PYTHONUNBUFFERED": "1",
                     "VLM_BATCH_PROXY": f"http://127.0.0.1:{self.batch_proxy_port}"},
            )
            elapsed = time.perf_counter() - t0
            success = result.returncode == 0

            return {
                "key": key,
                "task": cfg.task,
                "seed": cfg.seed,
                "method": cfg.method,
                "status": "success" if success else "failed",
                "returncode": result.returncode,
                "elapsed_sec": round(elapsed, 1),
                "stderr_tail": result.stderr[-500:] if result.stderr else "",
            }
        except subprocess.TimeoutExpired:
            return {
                "key": key, "task": cfg.task, "seed": cfg.seed,
                "method": cfg.method, "status": "timeout",
                "elapsed_sec": cfg.timeout + 120,
            }
        except Exception as e:
            return {
                "key": key, "task": cfg.task, "seed": cfg.seed,
                "method": cfg.method, "status": "error",
                "error": str(e),
            }

    # ── Build experiment grid ──
    def _build_grid(self, benchmark: str, seeds: List[int],
                    methods: List[str]) -> List[ExperimentConfig]:
        """Build the full experiment grid from benchmark YAML config."""
        import yaml
        bench_path = os.path.join(_PROJ, "src", "optimus1", "conf",
                                  "benchmark", f"{benchmark}.yaml")
        with open(bench_path) as f:
            bench_cfg = yaml.safe_load(f)

        tasks = bench_cfg.get("all_task", [])
        grid = []
        for idx, task in enumerate(tasks):
            task_name = task if isinstance(task, str) else task.get("name", str(idx))
            for seed in seeds:
                for method in methods:
                    grid.append(ExperimentConfig(
                        task=task_name, task_idx=idx, seed=seed,
                        method=method, benchmark=benchmark,
                        vlm_port=self.vlm_port, mc_port=0,
                    ))
        return grid

    # ── Main entry ──
    def run(self, benchmark: str, seeds: List[int] = None,
            methods: List[str] = None, plan_model: str = None,
            resume: bool = False):
        """Execute the full experiment grid in parallel."""
        seeds = seeds or DEFAULT_SEEDS
        methods = methods or DEFAULT_METHODS
        plan_model = plan_model or "Qwen/Qwen2.5-VL-7B-Instruct"

        if resume:
            self._load_checkpoint()

        # Build grid
        grid = self._build_grid(benchmark, seeds, methods)
        print(f"\n{'='*60}")
        print(f"  C-ACT Parallel Runner")
        print(f"  Benchmark: {benchmark}")
        print(f"  Methods: {len(methods)} ({', '.join(methods)})")
        print(f"  Seeds: {len(seeds)} ({seeds[0]}–{seeds[-1]})")
        print(f"  Total episodes: {len(grid)}")
        print(f"  Workers: {self.workers}")
        print(f"  Already completed: {len(self._completed)}")
        print(f"  To run: {len(grid) - len(self._completed)}")
        print(f"{'='*60}\n")

        if len(grid) - len(self._completed) == 0:
            print("[Done] All episodes already completed. Nothing to run.")
            return self._results

        # Start VLM server + batch proxy
        self._start_vlm_server(plan_model)
        self._start_batch_proxy()

        # Assign ports to workers
        for cfg in grid:
            cfg.vlm_port = self.vlm_port
            cfg.plan_model = plan_model

        try:
            # Run episodes in parallel
            completed = 0
            to_run = len(grid) - len(self._completed)

            with ProcessPoolExecutor(max_workers=self.workers) as pool:
                futures = {pool.submit(self._run_one, cfg): cfg for cfg in grid}
                for future in as_completed(futures):
                    result = future.result()
                    key = result["key"]
                    self._completed.add(key)
                    self._results.append(result)

                    completed += 1
                    elapsed = result.get("elapsed_sec", 0)

                    # Progress report
                    if completed % max(1, to_run // 20) == 0 or completed == to_run:
                        rate = completed / max(time.perf_counter() - self._t_start, 1) * 3600 \
                               if hasattr(self, '_t_start') else 0
                        print(f"[{completed}/{to_run}] {result['status']:>8} | "
                              f"task={result.get('task','?')} seed={result.get('seed','?')} "
                              f"method={result.get('method','?')} "
                              f"({elapsed:.0f}s)" +
                              (f" | ~{rate:.0f}/hr" if rate > 0 else ""))

                    # Save checkpoint every 10 completions
                    if completed % 10 == 0:
                        self._save_checkpoint()

        finally:
            self._stop_vlm_server()
            self._save_checkpoint()

        return self._results

    # ── Summary ──
    def print_summary(self):
        results = self._results
        if not results:
            print("[No results]")
            return

        by_method = {}
        for r in results:
            m = r.get("method", "?")
            if m not in by_method:
                by_method[m] = {"success": 0, "failed": 0, "timeout": 0, "skipped": 0, "total": 0, "time": 0}
            by_method[m]["total"] += 1
            by_method[m][r.get("status", "error")] += 1
            by_method[m]["time"] += r.get("elapsed_sec", 0)

        print(f"\n{'='*70}")
        print(f"  Experiment Summary")
        print(f"  {'Method':<25} {'Total':>6} {'Success':>8} {'Failed':>7} {'TO':>5} {'Skip':>5} {'Time':>8}")
        print(f"  {'-'*25} {'-'*6} {'-'*8} {'-'*7} {'-'*5} {'-'*5} {'-'*8}")
        for m, s in sorted(by_method.items()):
            h = s["time"] / 3600
            print(f"  {m:<25} {s['total']:>6} {s['success']:>8} {s['failed']:>7} "
                  f"{s['timeout']:>5} {s['skipped']:>5} {h:>7.1f}h")
        print(f"{'='*70}")

        total = sum(s["total"] for s in by_method.values())
        total_time = sum(s["time"] for s in by_method.values())
        wall_time = total_time / max(self.workers, 1)
        print(f"  Total episodes: {total}")
        print(f"  Total CPU time: {total_time/3600:.1f}h")
        print(f"  Estimated wall time ({self.workers} workers): {wall_time/3600:.1f}h")
        print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description="C-ACT Parallel Experiment Runner")
    parser.add_argument("--benchmark", default="cact_p3",
                       help="Benchmark YAML name (cact_calib, cact_p3, cact_train)")
    parser.add_argument("--seeds", default="4001-4008",
                       help="Seed range, e.g. '4001-4008' or '4001,4002,4003'")
    parser.add_argument("--methods", nargs="*", default=None,
                       help="Methods to run (default: all 7)")
    parser.add_argument("--workers", type=int, default=4,
                       help="Number of parallel workers (default: 4)")
    parser.add_argument("--vlm_port", type=int, default=12345,
                       help="VLM server port (default: 12345)")
    parser.add_argument("--plan_model", default="Qwen/Qwen2.5-VL-7B-Instruct",
                       help="VLM model name")
    parser.add_argument("--resume", action="store_true",
                       help="Resume from checkpoint")
    parser.add_argument("--checkpoint_dir", default=None,
                       help="Checkpoint directory")
    parser.add_argument("--print_grid", action="store_true",
                       help="Print experiment grid without running")

    args = parser.parse_args()

    # Parse seeds
    if "-" in args.seeds:
        lo, hi = args.seeds.split("-")
        seeds = list(range(int(lo), int(hi) + 1))
    else:
        seeds = [int(s) for s in args.seeds.split(",")]

    runner = ParallelRunner(
        workers=args.workers,
        vlm_port=args.vlm_port,
        checkpoint_dir=args.checkpoint_dir,
    )

    if args.print_grid:
        grid = runner._build_grid(args.benchmark, seeds, args.methods or DEFAULT_METHODS)
        print(f"Grid: {len(grid)} episodes")
        for cfg in grid[:20]:
            print(f"  task={cfg.task} seed={cfg.seed} method={cfg.method}")
        if len(grid) > 20:
            print(f"  ... and {len(grid)-20} more")
        return

    runner._t_start = time.perf_counter()
    runner.run(
        benchmark=args.benchmark,
        seeds=seeds,
        methods=args.methods,
        plan_model=args.plan_model,
        resume=args.resume,
    )
    runner.print_summary()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
C-ACT E5: Online Self-Evolution Runner.

10 rounds of accumulation → calibration → frozen test.
Each method+seed maintains its own persistent trust store.
Tracks SR, HRR, KPR, and knowledge lifecycle across rounds.

Usage:
  python experiments/online_runner.py --benchmark cact_train \
    --seeds 5001-5003 --rounds 10 \
    --methods Online-NoGate Online-BankCuration Online-ACT Online-C-ACT \
    --workers 4
"""

import sys, os, json, time, subprocess, argparse, shutil
from typing import Dict, List, Tuple
from pathlib import Path

_PROJ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _PROJ)

DEFAULT_SEEDS = [5001, 5002, 5003]
DEFAULT_METHODS = ["Online-NoGate", "Online-BankCuration", "Online-ACT", "Online-C-ACT"]
DEFAULT_ROUNDS = 10


class OnlineRunner:
    """Multi-round online self-evolution orchestrator."""

    def __init__(self, workers: int = 4, vlm_port: int = 12345,
                 rounds: int = DEFAULT_ROUNDS):
        self.workers = workers
        self.vlm_port = vlm_port
        self.rounds = rounds
        self._store_root = os.path.join(_PROJ, "exp_results", "online_stores")
        self._results_root = os.path.join(_PROJ, "exp_results", "online_results")
        self._vlm_proc = None
        os.makedirs(self._store_root, exist_ok=True)
        os.makedirs(self._results_root, exist_ok=True)

    def _trust_store_path(self, method: str, seed: int) -> str:
        safe = method.replace("-", "_").lower()
        return os.path.join(self._store_root, f"{safe}_seed{seed}")

    def _start_vlm(self, plan_model: str):
        cmd = [sys.executable, os.path.join(_PROJ, "app.py"),
               "--port", str(self.vlm_port), "--plan_model", plan_model]
        self._vlm_proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                          stderr=subprocess.DEVNULL)
        print(f"[VLM] Server on port {self.vlm_port} (PID {self._vlm_proc.pid})")
        time.sleep(8)

    def _stop_vlm(self):
        if self._vlm_proc:
            self._vlm_proc.terminate()
            try:
                self._vlm_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._vlm_proc.kill()
            print(f"[VLM] Stopped")

    def _run_episodes(self, benchmark: str, seeds: List[int],
                      method: str, phase: str,
                      trust_store_path: str = None,
                      frozen: bool = False) -> List[Dict]:
        """Run a batch of episodes via parallel_runner."""
        from experiments.parallel_runner import ParallelRunner

        runner = ParallelRunner(workers=self.workers, vlm_port=self.vlm_port)
        runner._t_start = time.perf_counter()

        # Use dedicated checkpoint dir per round
        ckpt = os.path.join(self._results_root, f"{method}_{phase}")
        runner.checkpoint_dir = ckpt
        os.makedirs(ckpt, exist_ok=True)
        runner._load_checkpoint()

        # Build methods list
        methods = [method]

        results = runner.run(
            benchmark=benchmark,
            seeds=seeds,
            methods=methods,
            plan_model="Qwen/Qwen2.5-VL-7B-Instruct",
            resume=False,
        )

        # Print summary
        from experiments.parallel_runner import ParallelRunner as PR
        runner.print_summary()
        return results

    def _compute_round_metrics(self, method: str, seed: int, round_num: int,
                                trust_store_path: str) -> Dict:
        """Read trust store and compute metrics for this round."""
        from cact.trust_store import TrustStore
        ts = TrustStore(store_path=trust_store_path)

        lifecycle = ts.lifecycle_stats()
        active = ts.get_active_knowledge()

        return {
            "round": round_num,
            "method": method,
            "seed": seed,
            "lifecycle": lifecycle,
            "active_knowledge": len(active),
            "certified": lifecycle.get("certified", 0),
            "deprecated": lifecycle.get("deprecated", 0),
            "disabled": lifecycle.get("disabled", 0),
            "total_knowledge": sum(lifecycle.values()),
        }

    def run_round(self, round_num: int, benchmark_accum: str,
                  benchmark_test: str, seeds: List[int],
                  methods: List[str]) -> Dict:
        """Execute one round for all methods and seeds."""

        print(f"\n{'='*70}")
        print(f"  ROUND {round_num}/{self.rounds}")
        print(f"{'='*70}")

        round_results = {}

        for method in methods:
            for seed in seeds:
                store_path = self._trust_store_path(method, seed)
                os.makedirs(store_path, exist_ok=True)

                key = f"{method}_seed{seed}"
                print(f"\n[{key}] Phase 1: Accumulation...")

                # Phase 1: Knowledge accumulation (updates trust store)
                self._run_episodes(
                    benchmark=benchmark_accum,
                    seeds=[seed],
                    method=method,
                    phase=f"R{round_num}_accum",
                    trust_store_path=store_path,
                    frozen=False,
                )

                print(f"[{key}] Phase 2: Calibration...")
                # Phase 2: Calibration (updates trust posteriors)
                self._run_episodes(
                    benchmark="cact_calib",
                    seeds=[seed],
                    method=method,
                    phase=f"R{round_num}_calib",
                    trust_store_path=store_path,
                    frozen=False,
                )

                print(f"[{key}] Phase 3: Frozen Test...")
                # Phase 3: Frozen evaluation
                test_results = self._run_episodes(
                    benchmark=benchmark_test,
                    seeds=[seed],
                    method=method,
                    phase=f"R{round_num}_test",
                    trust_store_path=store_path,
                    frozen=True,
                )

                # Compute round metrics
                metrics = self._compute_round_metrics(
                    method, seed, round_num, store_path)

                # Aggregate test success
                if test_results:
                    successes = sum(1 for r in test_results if r.get("status") == "success")
                    total = len(test_results)
                    metrics["test_sr"] = round(successes / max(total, 1), 3)
                    metrics["test_eps"] = total

                round_results[key] = metrics
                print(f"[{key}] SR={metrics.get('test_sr', '?')}, "
                      f"Certified={metrics['certified']}, "
                      f"Deprecated={metrics['deprecated']}")

        # Save round results
        round_file = os.path.join(self._results_root, f"round_{round_num:02d}.json")
        with open(round_file, "w") as f:
            json.dump(round_results, f, indent=2)

        return round_results

    def run(self, benchmark_accum: str, benchmark_test: str,
            seeds: List[int], methods: List[str],
            plan_model: str = "Qwen/Qwen2.5-VL-7B-Instruct"):
        """Execute the full online evolution experiment."""
        print(f"\n{'='*70}")
        print(f"  C-ACT Online Evolution Runner")
        print(f"  Rounds: {self.rounds}")
        print(f"  Methods: {len(methods)} ({', '.join(methods)})")
        print(f"  Seeds: {len(seeds)} ({seeds[0]}–{seeds[-1]})")
        print(f"  Total episodes: ~{self.rounds * len(methods) * len(seeds) * 36}")
        print(f"{'='*70}")

        self._start_vlm(plan_model)

        try:
            all_results = []
            for r in range(1, self.rounds + 1):
                results = self.run_round(
                    r, benchmark_accum, benchmark_test, seeds, methods)
                all_results.append(results)

            # Compile final report
            self._compile_report(all_results, methods, seeds)

        finally:
            self._stop_vlm()

        return all_results

    def _compile_report(self, all_results: List[Dict],
                        methods: List[str], seeds: List[int]):
        """Generate round-by-round report."""
        report_path = os.path.join(self._results_root, "online_report.json")
        report = {"methods": methods, "seeds": seeds, "rounds": []}

        for r_idx, round_data in enumerate(all_results):
            round_summary = {"round": r_idx + 1}
            for method in methods:
                srs = []
                certified = 0
                deprecated = 0
                active = 0
                for seed in seeds:
                    key = f"{method}_seed{seed}"
                    if key in round_data:
                        srs.append(round_data[key].get("test_sr", 0))
                        certified += round_data[key].get("certified", 0)
                        deprecated += round_data[key].get("deprecated", 0)
                        active += round_data[key].get("active_knowledge", 0)
                round_summary[method] = {
                    "avg_sr": round(sum(srs) / max(len(srs), 1), 3) if srs else 0,
                    "certified": certified,
                    "deprecated": deprecated,
                    "active": active,
                    "kpr": round(deprecated / max(certified, 1), 3),
                }
            report["rounds"].append(round_summary)

        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        # Print final report
        print(f"\n{'='*70}")
        print(f"  ONLINE EVOLUTION REPORT")
        print(f"{'='*70}")
        print(f"  {'Round':<6} ", end="")
        for m in methods:
            print(f"{'  ' + m[:12]:<16}", end="")
        print()
        print(f"  {'-'*6} ", end="")
        for _ in methods:
            print(f"{'  ' + '-'*12:<16}", end="")
        print()

        for rd in report["rounds"]:
            print(f"  R{rd['round']:<5} ", end="")
            for m in methods:
                s = rd.get(m, {})
                print(f"  SR={s.get('avg_sr',0):.2f} C={s.get('certified',0)} ", end="")
            print()

        # KPR trend
        print(f"\n  KPR (Knowledge Pollution Rate) over rounds:")
        for m in methods:
            kprs = [rd.get(m, {}).get("kpr", 0) for rd in report["rounds"]]
            print(f"    {m}: {kprs}")

        print(f"\n  Report saved: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="C-ACT Online Evolution Runner")
    parser.add_argument("--benchmark_accum", default="cact_train",
                       help="Benchmark for knowledge accumulation phases")
    parser.add_argument("--benchmark_test", default="cact_p3",
                       help="Benchmark for frozen test phases")
    parser.add_argument("--seeds", default="5001-5003",
                       help="Seed range, e.g. 5001-5003")
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS,
                       help="Number of evolution rounds (default: 10)")
    parser.add_argument("--methods", nargs="*", default=None,
                       help="Methods to run (default: 4 online methods)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--vlm_port", type=int, default=12345)

    args = parser.parse_args()

    if "-" in args.seeds:
        lo, hi = args.seeds.split("-")
        seeds = list(range(int(lo), int(hi) + 1))
    else:
        seeds = [int(s) for s in args.seeds.split(",")]

    methods = args.methods or DEFAULT_METHODS

    runner = OnlineRunner(workers=args.workers, vlm_port=args.vlm_port,
                          rounds=args.rounds)
    runner.run(
        benchmark_accum=args.benchmark_accum,
        benchmark_test=args.benchmark_test,
        seeds=seeds,
        methods=methods,
    )


if __name__ == "__main__":
    main()

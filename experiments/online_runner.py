#!/usr/bin/env python3
"""
C-ACT E5: Online Knowledge-Growth Evaluation.

10 rounds of accumulation → calibration → frozen test.
Each method+seed+round has persistent trust store.
Evaluation split: 6 retention tasks + 6 hard-transfer tasks per round.

Methods:
  Online-SuccessLifecycle  — lifecycle gate only (baseline)
  Online-FixedBayes        — fixed-threshold Bayesian gate
  Online-ACT               — counterfactual uplift, no contract
  Online-C-ACT             — full C-ACT pipeline

Per-round: 16 shared update + 4 calibration + 8 frozen eval = 28 episodes
Total per stream: 10 rounds × 20 episodes × 4 methods = 800 episodes

Usage:
  python experiments/online_runner.py --rounds 10 --workers 4

Design doc §24, Tables 134-141.
"""

import sys, os, json, time, subprocess, argparse, shutil
from typing import Dict, List, Tuple
from pathlib import Path

_PROJ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _PROJ)

DEFAULT_METHODS = ["Online-NoGate", "Online-FixedBayes",
                   "Online-C-ACT-Pointwise", "Online-C-ACT"]
DEFAULT_ROUNDS = 10
DEFAULT_SEED = 5001

# Per-round episodes
N_ACCUM = 16   # shared candidate/evidence update episodes
N_CALIB = 4     # calibration/check episodes
N_EVAL  = 8    # frozen evaluation (6 retention + 6 hard-transfer)


class OnlineRunner:
    """Multi-round online self-evolution with retention + hard-transfer."""

    def __init__(self, workers: int = 4, vlm_port: int = 12345,
                 rounds: int = DEFAULT_ROUNDS, protocol_path: str = ""):
        self.workers = workers
        self.vlm_port = vlm_port
        self.rounds = rounds
        self.protocol_path = protocol_path
        self._store_root = os.path.join(_PROJ, "exp_results", "online_stores")
        self._results_root = os.path.join(_PROJ, "exp_results", "online_results")
        self._vlm_proc = None
        os.makedirs(self._store_root, exist_ok=True)
        os.makedirs(self._results_root, exist_ok=True)

    def _trust_store_path(self, method: str, seed: int, round_num: int) -> str:
        safe = method.replace("-", "_").lower()
        return os.path.join(self._store_root, f"{safe}_seed{seed}_r{round_num:02d}")

    def _start_vlm(self, plan_model: str = "Qwen/Qwen2.5-VL-7B-Instruct"):
        cmd = [sys.executable, os.path.join(_PROJ, "app.py"),
               "--port", str(self.vlm_port), "--plan_model", plan_model]
        os.makedirs(os.path.join(_PROJ, "exp_results"), exist_ok=True)
        vlm_log = open(os.path.join(_PROJ, "exp_results", "vlm_server.log"), "a")
        self._vlm_proc = subprocess.Popen(cmd, stdout=vlm_log, stderr=vlm_log)
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

    # ── Per-phase runner ──
    def _run_phase(self, benchmark: str, seeds: List[int],
                   method: str, phase: str,
                   trust_store_path: str = None,
                   calibration_path: str = None,
                   frozen: bool = False,
                   active_calib_rate: float = 0.0,
                   label: str = "") -> List[Dict]:
        """Run one phase (accumulation/calibration/evaluation) via parallel_runner."""
        from experiments.parallel_runner import ParallelRunner

        runner = ParallelRunner(workers=1 if trust_store_path else self.workers, vlm_port=self.vlm_port)
        runner._t_start = time.perf_counter()

        ckpt = os.path.join(self._results_root, f"{method}_{phase}")
        runner.checkpoint_dir = ckpt
        os.makedirs(ckpt, exist_ok=True)
        runner._load_checkpoint()

        methods = [method]
        n_tasks = {"cact_online_stream": 16, "cact_calib": 4,
                   "cact_online_retention": 4,
                   "cact_online_hard_transfer": 4}.get(benchmark, 8)
        grid = runner._build_grid(benchmark, seeds, methods,
                                  plan_model="Qwen/Qwen2.5-VL-7B-Instruct",
                                  task_indices=list(range(n_tasks)))
        for cfg in grid:
            if trust_store_path:
                cfg.store_path = trust_store_path
            cfg.frozen = frozen
            if active_calib_rate:
                cfg.active_calib_rate = active_calib_rate
            cfg.calibration_path = calibration_path or ""
            cfg.protocol_path = self.protocol_path
            cfg.run_id = f"{method}_{phase}_seed{seeds[0]}_task{cfg.task_idx}"

        label_str = f" [{label}]" if label else ""
        print(f"  {phase}{label_str}: {len(grid)} episodes "
              f"(benchmark={benchmark}, frozen={frozen})")

        results = runner.run(
            benchmark=benchmark, seeds=seeds, methods=methods,
            plan_model="Qwen/Qwen2.5-VL-7B-Instruct",
            resume=False, grid=grid)
        runner.print_summary()
        return results

    def _run_round_update(self, seed: int, round_num: int, store_path: str,
                          prev_store_path: str = None) -> str:
        """Run the shared 16-update + 4-calibration stream phase once."""
        if prev_store_path and os.path.exists(prev_store_path):
            if os.path.exists(store_path):
                shutil.rmtree(store_path, ignore_errors=True)
            shutil.copytree(prev_store_path, store_path)
        self._run_phase("cact_online_stream", [seed], "Online-C-ACT",
                        f"R{round_num:02d}_shared_accum", store_path,
                        frozen=False, active_calib_rate=0.15, label=f"shared_accum({N_ACCUM})")
        self._run_phase("cact_calib", [seed], "Online-C-ACT",
                        f"R{round_num:02d}_shared_calib", store_path,
                        frozen=False, active_calib_rate=0.10, label=f"shared_calib({N_CALIB})")
        from cact.trust_store import TrustStore
        from cact.trust_gate import TrustGate
        calibration_path = os.path.join(store_path, "calibration.json")
        gate = TrustGate(); gate.calibrate_all_groups(TrustStore(store_path=store_path).get_calibration_data())
        gate.save_calibration(calibration_path)
        return store_path

    # ── Run one round for one method ──
    def _run_round_for_method(self, method: str, seed: int,
                               round_num: int,
                               prev_store_path: str = None,
                               shared_update_store: str = None) -> Dict:
        """Execute accumulation → calibration → evaluation for one (method, seed, round)."""
        store_path = self._trust_store_path(method, seed, round_num)

        # Copy previous round's store as starting point
        if prev_store_path and os.path.exists(prev_store_path):
            os.makedirs(store_path, exist_ok=True)
            for fname in ["cert_db.json", "contracts.json", "lifecycle.json"]:
                src = os.path.join(prev_store_path, fname)
                if os.path.exists(src):
                    shutil.copy(src, os.path.join(store_path, fname))

        if shared_update_store is None:
            # Phase 1: Accumulation (8 episodes, updates trust store)
            self._run_phase(
                benchmark="cact_online_stream",
                seeds=[seed], method=method,
                phase=f"R{round_num:02d}_accum",
                trust_store_path=store_path,
                frozen=False, active_calib_rate=0.15,
                label=f"accum({N_ACCUM})"
            )

            # Phase 2: Calibration (4 episodes, updates thresholds)
            self._run_phase(
                benchmark="cact_calib",
                seeds=[seed], method=method,
                phase=f"R{round_num:02d}_calib",
                trust_store_path=store_path,
                frozen=False, active_calib_rate=0.10,
                label=f"calib({N_CALIB})"
            )
            from cact.trust_store import TrustStore
            from cact.trust_gate import TrustGate
            calibration_path = os.path.join(store_path, "calibration.json")
            ts_cal = TrustStore(store_path=store_path)
            tg_cal = TrustGate()
            tg_cal.calibrate_all_groups(ts_cal.get_calibration_data())
            tg_cal.save_calibration(calibration_path)

        else:
            # E5 controlled-stream protocol: all methods receive the same
            # update/evidence snapshot; only frozen evaluation is method-specific.
            if os.path.exists(store_path):
                shutil.rmtree(store_path, ignore_errors=True)
            shutil.copytree(shared_update_store, store_path)
            calibration_path = os.path.join(store_path, "calibration.json")

        # Phase 3: Frozen Evaluation — Retention (4 episodes)
        ret_results = self._run_phase(
            benchmark="cact_online_retention",
            seeds=[seed], method=method,
            phase=f"R{round_num:02d}_retention",
            trust_store_path=store_path,
            calibration_path=calibration_path,
            frozen=True,
            label=f"retention(4)"
        )

        # Phase 3b: Frozen Evaluation — Hard-Transfer (4 episodes)
        ht_results = self._run_phase(
            benchmark="cact_online_hard_transfer",
            seeds=[seed], method=method,
            phase=f"R{round_num:02d}_hard_transfer",
            trust_store_path=store_path,
            calibration_path=calibration_path,
            frozen=True,
            label=f"hard_transfer(4)"
        )

        # Compile round metrics
        metrics = self._compute_round_metrics(method, seed, round_num, store_path)

        # Aggregate success rates
        for label, results in [("retention", ret_results), ("hard_transfer", ht_results)]:
            if results:
                verified = [r.get("task_success") for r in results if r.get("task_success") is not None]
                metrics[f"{label}_sr"] = round(sum(verified) / len(verified), 3) if verified else None

        ret = metrics.get("retention_sr")
        hard = metrics.get("hard_transfer_sr")
        metrics["eval_sr"] = round((ret * 4 + hard * 4) / 8, 3) if ret is not None and hard is not None else None

        return metrics

    def _compute_round_metrics(self, method: str, seed: int, round_num: int,
                                store_path: str) -> Dict:
        """Read trust store and compute lifecycle metrics."""
        from cact.trust_store import TrustStore
        ts = TrustStore(store_path=store_path)
        lifecycle = ts.lifecycle_stats()
        active = ts.get_active_knowledge()

        return {
            "round": round_num, "method": method, "seed": seed,
            "lifecycle": lifecycle,
            "active_knowledge": len(active),
            "certified": lifecycle.get("certified", 0),
            "deprecated": lifecycle.get("deprecated", 0),
            "disabled": lifecycle.get("disabled", 0),
            "total_knowledge": sum(lifecycle.values()),
            "harm_rate": (sum(bool(x.get("is_harmful")) for x in ts.get_observations() if x.get("used")) /
                          max(1, sum(1 for x in ts.get_observations() if x.get("used")))),
        }

    # ── Full experiment ──
    def run(self, methods: List[str] = None, seed: int = DEFAULT_SEED):
        methods = methods or DEFAULT_METHODS

        print(f"\n{'='*70}")
        print(f"  C-ACT E5: Online Knowledge-Growth Evaluation")
        print(f"  Rounds: {self.rounds}")
        print(f"  Methods: {len(methods)} ({', '.join(methods)})")
        print(f"  Seed: {seed}")
        print(f"  Per round: {N_ACCUM} acc + {N_CALIB} cal + {N_EVAL} eval")
        print(f"  Total: {self.rounds} × {N_ACCUM+N_CALIB+N_EVAL} × {len(methods)} = "
              f"{self.rounds * (N_ACCUM+N_CALIB+N_EVAL) * len(methods)} episodes")
        print(f"{'='*70}")

        self._start_vlm()
        all_results = []  # [round_num][method] = metrics

        try:
            # Track previous store paths per method for round-to-round persistence
            prev_stores: Dict[str, str] = {}  # method → previous store path

            for r in range(1, self.rounds + 1):
                print(f"\n{'─'*70}")
                print(f"  ROUND {r}/{self.rounds}")
                print(f"{'─'*70}")

                round_data = {}
                shared_path = self._trust_store_path("SharedStream", seed, r)
                shared_prev = prev_stores.get("__shared__")
                self._run_round_update(seed, r, shared_path, shared_prev)
                prev_stores["__shared__"] = shared_path
                for method in methods:
                    print(f"\n  [{method}] Round {r}...")
                    metrics = self._run_round_for_method(
                        method, seed, r, shared_update_store=shared_path)
                    prev_stores[method] = self._trust_store_path(method, seed, r)
                    round_data[method] = metrics

                    print(f"  [{method}] R{r} eval_sr={metrics.get('eval_sr','?')} "
                          f"ret_sr={metrics.get('retention_sr','?')} "
                          f"ht_sr={metrics.get('hard_transfer_sr','?')} "
                          f"cert={metrics['certified']} dep={metrics['deprecated']}")

                # Save round results
                round_file = os.path.join(
                    self._results_root, f"round_{r:02d}.json")
                with open(round_file, "w") as f:
                    json.dump(round_data, f, indent=2)
                all_results.append(round_data)

            # Compile final report
            self._compile_report(all_results, methods)

        finally:
            self._stop_vlm()

        return all_results

    # ── Final report ──
    def _compile_report(self, all_results: List[Dict], methods: List[str]):
        """Generate E5 metrics: RetentionSR, HardSR, SafetyDrift, KPR, AULC."""
        report_path = os.path.join(self._results_root, "e5_report.json")

        # Collect round-by-round series
        series = {m: {
            "eval_sr": [], "retention_sr": [], "hard_transfer_sr": [],
            "certified": [], "deprecated": [], "disabled": [],
            "active_knowledge": [], "harm_rate": [],
        } for m in methods}

        for round_data in all_results:
            for m in methods:
                if m in round_data:
                    rd = round_data[m]
                    for key in series[m]:
                        series[m][key].append(rd.get(key, 0))

        # Compute derived metrics
        report = {"methods": methods, "rounds": self.rounds, "series": {}}
        for m in methods:
            s = series[m]
            r = self.rounds

            # AULC: Area Under Learning Curve (average eval SR over rounds)
            valid_eval = [v for v in s["eval_sr"] if v is not None]
            aulc = sum(valid_eval) / len(valid_eval) if valid_eval else None

            hs = [v for v in s["harm_rate"] if v is not None]
            safety_drift = (hs[-1] - hs[0]) if len(hs) >= 2 else None

            # KPR at final round
            cert_last = s["certified"][-1] if s["certified"] else 0
            dep_last = s["deprecated"][-1] if s["deprecated"] else 0
            kpr = dep_last / max(cert_last, 1)

            report["series"][m] = {
                "aulc": round(aulc, 3) if aulc is not None else None,
                "eval_sr_final": round(s["eval_sr"][-1], 3) if s["eval_sr"] and s["eval_sr"][-1] is not None else None,
                "retention_sr_final": round(s["retention_sr"][-1], 3) if s["retention_sr"] and s["retention_sr"][-1] is not None else None,
                "hard_transfer_sr_final": round(s["hard_transfer_sr"][-1], 3) if s["hard_transfer_sr"] and s["hard_transfer_sr"][-1] is not None else None,
                "safety_drift": round(safety_drift, 3) if safety_drift is not None else None,
                "kpr": round(kpr, 3),
                "certified_final": cert_last,
                "deprecated_final": dep_last,
                "disabled_final": s["disabled"][-1] if s["disabled"] else 0,
                "active_knowledge_final": s["active_knowledge"][-1] if s["active_knowledge"] else 0,
                "eval_sr_series": [round(v, 3) for v in s["eval_sr"]],
                "retention_sr_series": [round(v, 3) for v in s["retention_sr"]],
                "hard_transfer_sr_series": [round(v, 3) for v in s["hard_transfer_sr"]],
            }

        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        # Print summary table
        print(f"\n{'='*90}")
        print(f"  E5 ONLINE EVOLUTION REPORT")
        print(f"{'='*90}")
        header = f"  {'Method':<28} {'AULC':>7} {'EvalSR':>7} {'RetSR':>6} {'HTSR':>6} {'Safety':>7} {'KPR':>6} {'Cert':>5} {'Dep':>5}"
        print(header)
        print(f"  {'─'*28} {'─'*7} {'─'*7} {'─'*6} {'─'*6} {'─'*7} {'─'*6} {'─'*5} {'─'*5}")
        def fmt(v, width=7):
            return f"{v:{width}.3f}" if isinstance(v, (int, float)) else f"{'n/a':>{width}}"
        for m in methods:
            s = report["series"][m]
            print(f"  {m:<28} {fmt(s['aulc'])} {fmt(s['eval_sr_final'])} "
                  f"{fmt(s['retention_sr_final'], 6)} {fmt(s['hard_transfer_sr_final'], 6)} "
                  f"{fmt(s['safety_drift'])} {fmt(s['kpr'], 6)} "
                  f"{s['certified_final']:>5} {s['deprecated_final']:>5}")

        # Round-by-round eval SR
        print(f"\n  Eval SR by round:")
        print(f"  {'Round':<7}", end="")
        for r in range(1, self.rounds + 1):
            print(f"{'R'+str(r):>7}", end="")
        print()
        for m in methods:
            print(f"  {m:<28}", end="")
            for v in report["series"][m]["eval_sr_series"]:
                print(f"{v:>7.3f}", end="")
            print()

        print(f"\n  Report: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="C-ACT E5 Online Evolution Runner")
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    parser.add_argument("--methods", nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--vlm_port", type=int, default=12345)
    parser.add_argument("--protocol_path", default="")

    args = parser.parse_args()
    methods = args.methods or DEFAULT_METHODS

    runner = OnlineRunner(workers=args.workers, vlm_port=args.vlm_port,
                          rounds=args.rounds, protocol_path=args.protocol_path)
    runner.run(methods=methods, seed=args.seed)


if __name__ == "__main__":
    main()

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
  # E3 main evaluation (36 tasks x 8 seeds x 7 methods = 2016 episodes)
  python experiments/parallel_runner.py \\
    --benchmark cact_p3 --seeds 4001-4008 --methods \\
    NoKnowledge NoGate FixedBayes PairwisePreferenceGate C-ACT-Pointwise C-ACT \\
    --workers 4 --vlm_port 12345 --mc_base_port 15000

  # Resume interrupted run
  python experiments/parallel_runner.py --resume --checkpoint_dir exp_results/ckpt/

Speed: ~4x with 4 workers. E3: 17h → 4-5h (single MC per worker).
       ~8x with 8 workers + MC instance sharing.
"""

import subprocess, sys, os, json, time, signal, shutil, argparse, socket, hashlib, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from experiments.world_identity import derive_snapshot_hash
from pathlib import Path

_PROJ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _PROJ)

# -- Defaults --
DEFAULT_METHODS = ["NoKnowledge", "NoGate", "FixedBayes",
                   "PairwisePreferenceGate", "C-ACT-Pointwise", "C-ACT"]
DEFAULT_SEEDS = [4001, 4002, 4003, 4004, 4005, 4006, 4007, 4008]

def _cleanup_owned_minecraft(run_id: str) -> None:
    """Remove only the Docker container labeled for this episode."""
    if not run_id or os.name == "nt":
        return
    try:
        listed = subprocess.run(["docker", "ps", "-aq", "--filter", f"label=cact.run_id={run_id}"], check=False, capture_output=True, text=True, timeout=10)
        ids = [x.strip() for x in listed.stdout.splitlines() if x.strip()]
        for container_id in ids:
            # Remove one at a time: a Java/Minecraft process can take longer
            # to reap than Docker's default batch request, and a failed batch
            # would otherwise leave every sibling container running.
            subprocess.run(["docker", "rm", "-f", container_id], check=False,
                           capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        pass
def _run_with_process_group(args, *, timeout=None, **kwargs):
    """Run a child in its own process group so Minecraft descendants are cleaned up."""
    kwargs.pop("text", None)
    if os.name != "nt":
        kwargs.setdefault("start_new_session", True)
    proc = subprocess.Popen(args, **kwargs)
    try:
        returncode = proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        if os.name != "nt":
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        else:
            proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            if os.name != "nt":
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            else:
                proc.kill()
            proc.wait()
        raise
    return subprocess.CompletedProcess(args, returncode)


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
    timeout: int = 180  # per-episode soft timeout in seconds (was 300)
    env_times: int = 1
    prefix: str = "cact"
    store_path: str = ""
    frozen: bool = False
    active_calib_rate: float = 0.0
    calibration_path: str = ""
    run_id: str = ""
    snapshot_path: str = ""
    protocol_path: str = ""
    branch_mode: str = ""
    branch_target_opportunity: str = ""
    branch_parent_id: str = ""
    branch_prefix_assignment: int = 0
    branch_prefix_trace: str = ""
    cact_kappa: str = ""       # E2: override policy kappa
    snapshot_hash: str = ""


def _clone_snapshot(src: str, dst: str) -> None:
    """Clone a frozen store; optional hardlinks avoid repeated Ubuntu copies."""
    if (os.environ.get("CACT_FROZEN_HARDLINK") != "1"
            or os.environ.get("CACT_ALLOW_UNSAFE_HARDLINK") != "1"):
        shutil.copytree(src, dst); return
    os.makedirs(dst, exist_ok=True)
    for root, dirs, files in os.walk(src):
        rel = os.path.relpath(root, src)
        target = dst if rel == "." else os.path.join(dst, rel)
        os.makedirs(target, exist_ok=True)
        for name in files:
            os.link(os.path.join(root, name), os.path.join(target, name))

class ParallelRunner:
    """Orchestrate parallel experiment execution."""

    def __init__(self, workers: int = 4, vlm_port: int = 12345,
                 mc_base_port: int = 15000, checkpoint_dir: str = None,
                 batch_proxy_port: int = 12346, vlm_ports: str = "",
                 world_snapshot_manifest: str = ""):
        self.workers = workers
        self.vlm_port = vlm_port
        self.mc_base_port = mc_base_port
        self.batch_proxy_port = batch_proxy_port
        self.checkpoint_dir = checkpoint_dir or os.path.join(
            _PROJ, "exp_results", "ckpt")
        self._vlm_ports: List[int] = []
        if vlm_ports:
            self._vlm_ports = [int(x) for x in vlm_ports.split(",") if x]
        self._world_snapshot_manifest_supplied = bool(world_snapshot_manifest)
        self._world_snapshot_hashes: Dict[str, str] = {}
        if world_snapshot_manifest:
            with open(world_snapshot_manifest, encoding="utf-8") as handle:
                payload = json.load(handle)
            hashes = payload.get("hashes", payload) if isinstance(payload, dict) else {}
            if not isinstance(hashes, dict):
                raise ValueError("world snapshot manifest must be a JSON mapping")
            self._world_snapshot_hashes = {str(k): str(v) for k, v in hashes.items() if str(v)}
        self._completed: set = set()
        self._results: List[Dict] = []
        self._server_procs: Dict[int, subprocess.Popen] = {}
        self._proxy_thread = None

        os.makedirs(self.checkpoint_dir, exist_ok=True)

    # -- Port management --
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

    # -- Checkpoint / resume --
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

    # -- Server lifecycle --
    def _start_vlm_server(self, plan_model: str):
        """Start or validate a healthy shared VLM planning server."""
        health = f"http://127.0.0.1:{self.vlm_port}/health"
        if not self._port_free(self.vlm_port):
            try:
                with urllib.request.urlopen(health, timeout=3):
                    print(f"[VLM] Reusing healthy server on port {self.vlm_port}")
                    return
            except Exception as exc:
                raise RuntimeError(f"VLM port {self.vlm_port} is occupied but unhealthy") from exc
        cmd = [sys.executable, os.path.join(_PROJ, "app.py"),
               "--port", str(self.vlm_port), "--plan_model", plan_model]
        os.makedirs(os.path.join(_PROJ, "exp_results"), exist_ok=True)
        vlm_log = open(os.path.join(_PROJ, "exp_results", "vlm_server.log"), "a", encoding="utf-8")
        env = {**os.environ, "PYTHONUNBUFFERED": "1",
               "PYTHONPATH": os.pathsep.join([_PROJ, os.path.join(_PROJ, "src"), os.path.join(_PROJ, "minerl")])}
        proc = subprocess.Popen(cmd, stdout=vlm_log, stderr=vlm_log, env=env, start_new_session=(os.name != "nt"))
        vlm_log.close(); self._server_procs[self.vlm_port] = proc
        print(f"[VLM] Started planning server on port {self.vlm_port} (PID {proc.pid})")
        ready = False
        for _ in range(int(os.environ.get("CACT_VLM_HEALTH_RETRIES", "240"))):
            try:
                with urllib.request.urlopen(health, timeout=2):
                    ready = True; break
            except Exception:
                time.sleep(0.5)
        if not ready:
            proc.terminate()
            raise RuntimeError(f"VLM server on port {self.vlm_port} failed health check")

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

    # -- Single episode execution --
    def _run_one(self, cfg: ExperimentConfig) -> Dict:
        """Run a single (task, seed, method) episode via subprocess."""
        key = (cfg.task, cfg.seed, cfg.method, cfg.frozen)
        if key in self._completed:
            return {"key": key, "status": "skipped", "reason": "already_completed"}

        cmd = [
            sys.executable, "-m", "optimus1.main_planning",
            f"server.port={cfg.vlm_port}",
            f"server.url=http://127.0.0.1",
            f"benchmark={cfg.benchmark}",
            f"+evaluate=[{cfg.task_idx}]",
            f"env.times={cfg.env_times}",
            f"seed={cfg.seed}",
            # Keep the statistical seed and the procedural Minecraft world
            # seed aligned.  MineRL's env.seed() only seeds spaces/token
            # generation; the mission XML uses the Hydra world_seed override.
            f"world_seed={cfg.seed}",
            f"prefix={cfg.prefix}",
            f"+cact_method={cfg.method}",
            f"plan_model={cfg.plan_model}",
        ]
        if cfg.snapshot_path and cfg.store_path:
            import shutil
            if os.path.exists(cfg.store_path):
                shutil.rmtree(cfg.store_path, ignore_errors=True)
            if os.path.exists(cfg.snapshot_path):
                _clone_snapshot(cfg.snapshot_path, cfg.store_path)
        if cfg.store_path:
            cmd.append(f"+cact_store_path={cfg.store_path}")
        if cfg.calibration_path:
            cmd.append(f"+cact_calibration_path={cfg.calibration_path}")
        if cfg.protocol_path:
            cmd.append(f"+cact_protocol_path={cfg.protocol_path}")
        if cfg.run_id:
            cmd.append(f"+cact_run_id={cfg.run_id}")
        if cfg.frozen:
            cmd.append("+cact_frozen=true")
        if cfg.active_calib_rate:
            cmd.append(f"+cact_active_calib_rate={cfg.active_calib_rate}")
        if cfg.cact_kappa:
            cmd.append(f"+cact_kappa={cfg.cact_kappa}")  # E2 direct select kappa override
        if not cfg.snapshot_hash:
            cfg.snapshot_hash = derive_snapshot_hash(cfg.task_idx, cfg.seed)
        if cfg.frozen and cfg.protocol_path and os.environ.get("CACT_REQUIRE_WORLD_SNAPSHOT_HASH") == "1" and not cfg.snapshot_hash:
            return {"key": key, "run_id": cfg.run_id, "task": cfg.task, "seed": cfg.seed, "method": cfg.method,
                    "status": "error", "error": "missing world snapshot hash for frozen protocol run"}
        if cfg.snapshot_hash:
            cmd.append(f"+cact_snapshot_hash={cfg.snapshot_hash}")
        if cfg.branch_mode:
            cmd.append(f"+cact_branch_mode={cfg.branch_mode}")
            cmd.append(f"+cact_branch_target_opportunity={cfg.branch_target_opportunity}")
            cmd.append(f"+cact_branch_parent_id={cfg.branch_parent_id}")
            cmd.append(f"+cact_branch_prefix_assignment={cfg.branch_prefix_assignment}")
            if cfg.branch_prefix_trace:
                cmd.append(f"+cact_branch_prefix_trace={json.dumps(cfg.branch_prefix_trace)}")

        def store_hash(path):
            h = hashlib.sha256()
            if path and os.path.isdir(path):
                for root, _, files in os.walk(path):
                    for name in sorted(files):
                        fp = os.path.join(root, name)
                        h.update(os.path.relpath(fp, path).encode())
                        with open(fp, "rb") as fh:
                            h.update(fh.read())
            return h.hexdigest()

        frozen_hash_before = store_hash(cfg.store_path) if cfg.frozen else None
        protocol_hash_before = store_hash(cfg.protocol_path) if (cfg.frozen and cfg.protocol_path and os.path.isfile(cfg.protocol_path)) else None
        t0 = time.perf_counter()
        try:
            runner_log_dir = os.path.join(_PROJ, "exp_results", "runner_logs")
            os.makedirs(runner_log_dir, exist_ok=True)
            log_id = str(cfg.run_id or f"{cfg.benchmark}_{cfg.seed}_{cfg.task_idx}_{cfg.method}").replace("\r", "").replace("\n", "")
            stdout_path = os.path.join(runner_log_dir, f"{log_id}.stdout.log")
            stderr_path = os.path.join(runner_log_dir, f"{log_id}.stderr.log")
            child_env = {**os.environ, "PYTHONUNBUFFERED": "1",
                         "CACT_RUN_ID": log_id,
                         "PYTHONPATH": os.pathsep.join([_PROJ, os.path.join(_PROJ, "src"), os.path.join(_PROJ, "minerl")]),
                         # External multi-GPU pools already expose one
                         # endpoint per worker; do not collapse all requests
                         # onto the single batch proxy/GPU 0.
                         "VLM_BATCH_PROXY": (f"http://127.0.0.1:{cfg.vlm_port}"
                                             if self._vlm_ports else
                                             f"http://127.0.0.1:{self.batch_proxy_port}"),
                         "CACT_TRUST_STORE_DIR": cfg.store_path or os.environ.get("CACT_TRUST_STORE_DIR", "")}
            # Assign Minecraft GPUs deterministically per episode. Each episode runs in
            # its own subprocess, so a process-local round-robin counter would always
            # restart at GPU 0 and silently defeat multi-GPU execution.
            gpu_ids = [x.strip() for x in os.environ.get("MINERL_GPU_IDS", "").split(",") if x.strip()]
            if gpu_ids:
                gpu_key = f"{cfg.seed}:{cfg.task_idx}:{cfg.method}"
                gpu_idx = int(hashlib.sha256(gpu_key.encode("utf-8")).hexdigest(), 16) % len(gpu_ids)
                child_env["CACT_MC_GPU_ID"] = gpu_ids[gpu_idx]
            with open(stdout_path, "w", encoding="utf-8") as stdout, open(stderr_path, "w", encoding="utf-8") as stderr:
                result = _run_with_process_group(cmd, stdout=stdout, stderr=stderr, text=True,
                                        timeout=cfg.timeout * cfg.env_times + 60,
                                        cwd=_PROJ, env=child_env)
            elapsed = time.perf_counter() - t0
            try:
                with open(stderr_path, encoding="utf-8", errors="replace") as fh:
                    fh.seek(0, os.SEEK_END); size = fh.tell(); fh.seek(max(0, size - 500), os.SEEK_SET)
                    stderr_tail = fh.read()
            except OSError:
                stderr_tail = ""
            success = result.returncode == 0
            frozen_hash_after = store_hash(cfg.store_path) if cfg.frozen else None
            protocol_hash_after = store_hash(cfg.protocol_path) if (cfg.frozen and cfg.protocol_path and os.path.isfile(cfg.protocol_path)) else None
            if cfg.frozen and frozen_hash_before != frozen_hash_after:
                success = False
                frozen_error = "frozen store mutated"
            elif cfg.frozen and protocol_hash_before != protocol_hash_after:
                success = False
                frozen_error = "frozen policy artifact mutated"
            else:
                frozen_error = ""
            task_success = None
            if cfg.run_id:
                episode_file = os.path.join(_PROJ, "exp_results", "cact_logs",
                                            cfg.run_id, "episode", "episode.jsonl")
                if os.path.exists(episode_file):
                    try:
                        with open(episode_file, encoding="utf-8") as fh:
                            rows = [json.loads(line) for line in fh if line.strip()]
                        if rows:
                            task_success = bool(rows[-1].get("success"))
                    except (OSError, json.JSONDecodeError):
                        task_success = None

            return {
                "key": key,
                "run_id": cfg.run_id,
                "task": cfg.task,
                "seed": cfg.seed,
                "method": cfg.method,
                "status": "success" if success else "failed",
                "returncode": result.returncode,
                "elapsed_sec": round(elapsed, 1),
                "stderr_tail": stderr_tail,
                "stdout_log": stdout_path,
                "stderr_log": stderr_path,
                "task_success": task_success,
                "frozen_store_hash": frozen_hash_after if cfg.frozen else None,
                "frozen_policy_hash": protocol_hash_after if cfg.frozen else None,
                "frozen_error": frozen_error,
            }

        except subprocess.TimeoutExpired:
            return {
                "key": key, "run_id": cfg.run_id, "task": cfg.task, "seed": cfg.seed,
                "method": cfg.method, "status": "timeout",
                "elapsed_sec": cfg.timeout + 120,
            }
        except Exception as e:
            return {
                "key": key, "run_id": cfg.run_id, "task": cfg.task, "seed": cfg.seed,
                "method": cfg.method, "status": "error",
                "error": str(e),
            }

        finally:
            _cleanup_owned_minecraft(log_id)

    def _snapshot_hash_for(self, task_idx: int, seed: int) -> str:
        key = f"{task_idx}|{seed}"
        value = self._world_snapshot_hashes.get(key)
        if self._world_snapshot_manifest_supplied and not value:
            raise ValueError(f"world snapshot manifest missing required cell {key}")
        return value or derive_snapshot_hash(task_idx, seed)

    # -- Build experiment grid --
    def _build_grid(self, benchmark: str, seeds: List[int],
                    methods: List[str],
                    plan_model: str = "Qwen/Qwen2.5-VL-7B-Instruct",
                    task_indices: List[int] = None) -> List[ExperimentConfig]:
        """Build the full experiment grid from benchmark YAML config."""
        bench_path = os.path.join(_PROJ, "src", "optimus1", "conf",
                                  "benchmark", f"{benchmark}.yaml")
        try:
            import yaml
            with open(bench_path, encoding="utf-8") as f:
                bench_cfg = yaml.safe_load(f)
            tasks = bench_cfg.get("all_task", [])
        except ModuleNotFoundError:
            # Lightweight fallback for the runner's task index selection when
            # PyYAML is unavailable in a minimal environment.
            import re
            tasks = []
            current = None
            with open(bench_path, encoding="utf-8") as f:
                for raw in f:
                    line = raw.strip()
                    if line.startswith("- {"):
                        current = {}
                        for key, quoted, bare in re.findall(r"([A-Za-z_]+):\s*(?:[\"']([^\"']*)[\"']|([^,}]+))", line[3:]):
                            current[key] = (quoted or bare).strip()
                        tasks.append(current)
                    elif line.startswith("- id:"):
                        current = {"id": line.split(":", 1)[1].strip()}
                        tasks.append(current)
                    elif current is not None and ":" in line and not line.startswith("#"):
                        key, value = line.split(":", 1)
                        value = value.strip().strip("\"'")
                        if key.strip() in {"type", "instruction", "goal", "group", "difficulty"}:
                            current[key.strip()] = value
        grid = []
        allowed = set(task_indices) if task_indices is not None else None
        if allowed and any(i < 0 or i >= len(tasks) for i in allowed):
            invalid = sorted(i for i in allowed if i < 0 or i >= len(tasks))
            raise ValueError(f"task_indices out of range for {benchmark}: {invalid}")
        for idx, task in enumerate(tasks):
            if allowed is not None and idx not in allowed:
                continue
            task_name = task if isinstance(task, str) else task.get("name", str(idx))
            for seed in seeds:
                for method in methods:
                    grid.append(ExperimentConfig(
                        task=task_name, task_idx=idx, seed=seed,
                        method=method, benchmark=benchmark,
                        vlm_port=self.vlm_port, mc_port=0,
                        store_path=os.path.join(_PROJ, "exp_results", "stores",
                                                benchmark,
                                                method.replace("/", "_").replace("-", "_"),
                                                f"seed_{seed}", f"task_{idx}"),
                        run_id=f"{benchmark}_{method}_seed{seed}_task{idx}",
                        snapshot_hash=self._snapshot_hash_for(idx, seed),
                    ))
        return grid

    # -- Main entry --
    def run(self, benchmark: str, seeds: List[int] = None,
            methods: List[str] = None, plan_model: str = None,
            resume: bool = False, grid: List[ExperimentConfig] = None):
        """Execute the full experiment grid in parallel.

        Args:
            grid: Optional pre-built grid (with custom store_path, frozen, etc.).
                  If provided, benchmark/seeds/methods are ignored for grid construction.
        """
        if grid is not None:
            seeds = seeds or DEFAULT_SEEDS
            methods = methods or DEFAULT_METHODS
            plan_model = plan_model or "Qwen/Qwen2.5-VL-7B-Instruct"
        else:
            seeds = seeds or DEFAULT_SEEDS
            methods = methods or DEFAULT_METHODS
            plan_model = plan_model or "Qwen/Qwen2.5-VL-7B-Instruct"

        if resume:
            self._load_checkpoint()

        # Build grid (unless caller provided pre-built grid)
        if grid is None:
            grid = self._build_grid(benchmark, seeds, methods)
        print(f"\n{'='*60}")
        print(f"  C-ACT Parallel Runner")
        print(f"  Benchmark: {benchmark}")
        print(f"  Methods: {len(methods)} ({', '.join(methods)})")
        print(f"  Seeds: {len(seeds)} ({seeds[0]}-{seeds[-1]})")
        print(f"  Total episodes: {len(grid)}")
        print(f"  Workers: {self.workers}")
        pending = [cfg for cfg in grid if (cfg.task, cfg.seed, cfg.method, cfg.frozen) not in self._completed]
        print(f"  Already completed: {len(grid) - len(pending)}")
        print(f"  To run: {len(pending)}")
        print(f"{'='*60}\n")

        if not pending:
            print("[Done] All episodes already completed. Nothing to run.")
            return self._results

        # Start VLM server + batch proxy (skip both when a pool is external).
        if self._vlm_ports:
            print(f"[VLM] Using external VLM pool: {self._vlm_ports}")
        else:
            self._start_vlm_server(plan_model)
            self._start_batch_proxy()

        # Pair each VLM endpoint with the same GPU used by its Minecraft
        # subprocess. This avoids cross-GPU model traffic and makes the
        # allocation auditable from the episode logs.
        ports = self._vlm_ports if self._vlm_ports else [self.vlm_port]
        gpu_ids = [int(x.strip()) for x in os.environ.get("MINERL_GPU_IDS", "").split(",") if x.strip()]
        gpu_to_port = {gpu: ports[i] for i, gpu in enumerate(gpu_ids) if i < len(ports)}
        for i, cfg in enumerate(grid):
            if gpu_ids and gpu_to_port:
                gpu_key = f"{cfg.seed}:{cfg.task_idx}:{cfg.method}"
                gpu_id = gpu_ids[int(hashlib.sha256(gpu_key.encode("utf-8")).hexdigest(), 16) % len(gpu_ids)]
                cfg.vlm_port = gpu_to_port.get(gpu_id, ports[i % len(ports)])
            else:
                cfg.vlm_port = ports[i % len(ports)]
            cfg.plan_model = plan_model

        try:
            # Run episodes in parallel
            completed = 0
            to_run = len(pending)

            batch_failures = []
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                futures = {pool.submit(self._run_one, cfg): cfg for cfg in pending}
                for future in as_completed(futures):
                    result = future.result()
                    key = result["key"]
                    if result.get("status") in {"success", "skipped", "failed", "timeout"}:
                        self._completed.add(key)
                    else:
                        batch_failures.append(result)
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

        if batch_failures:
            summary = ", ".join(f"{r.get('method', '?')}:{r.get('seed', '?')}/{r.get('task', '?')}={r.get('status')}" for r in batch_failures[:8])
            raise RuntimeError(f"{len(batch_failures)} experiment episodes failed; refusing a partial-success exit: {summary}")
        return self._results

    # -- Summary --
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
    parser.add_argument("--vlm_ports", type=str, default="",
                       help="Comma-separated VLM ports for multi-GPU pool (e.g. 12345,12346,12347)")
    parser.add_argument("--world_snapshot_manifest", default="",
                       help="JSON mapping task_id|world_seed to canonical world snapshot hash")
    parser.add_argument("--plan_model", default="Qwen/Qwen2.5-VL-7B-Instruct",
                       help="VLM model name")
    parser.add_argument("--resume", action="store_true",
                       help="Resume from checkpoint")
    parser.add_argument("--checkpoint_dir", default=None,
                       help="Checkpoint directory")
    parser.add_argument("--task_indices", default=None,
                       help="Comma-separated task indices to run")
    parser.add_argument("--frozen", action="store_true")
    parser.add_argument("--calibration_path", default="")
    parser.add_argument("--active_calib_rate", type=float, default=0.0)
    parser.add_argument("--kappa", default="", help="Override calibrated kappa for E2/E4 matched-policy rollouts")
    parser.add_argument("--store_path", default="")
    parser.add_argument("--snapshot_path", default="")
    parser.add_argument("--protocol_path", default="")
    parser.add_argument("--branch_mode", choices=["", "reuse", "base"], default="")
    parser.add_argument("--branch_target_opportunity", default="")
    parser.add_argument("--branch_parent_id", default="")
    parser.add_argument("--branch_prefix_assignment", type=int, default=0)
    parser.add_argument("--print_grid", action="store_true",
                       help="Print experiment grid without running")

    args = parser.parse_args()

    # Parse seeds
    if "-" in args.seeds:
        lo, hi = args.seeds.split("-")
        seeds = list(range(int(lo), int(hi) + 1))
    else:
        seeds = [int(s) for s in args.seeds.split(",")]

    task_indices = [int(x) for x in args.task_indices.split(",")] if args.task_indices else None

    runner = ParallelRunner(
        workers=args.workers,
        vlm_port=args.vlm_port,
        vlm_ports=args.vlm_ports,
        world_snapshot_manifest=args.world_snapshot_manifest,
        checkpoint_dir=args.checkpoint_dir,
    )

    if args.print_grid:
        grid = runner._build_grid(args.benchmark, seeds, args.methods or DEFAULT_METHODS, task_indices=task_indices)
        print(f"Grid: {len(grid)} episodes")
        for cfg in grid[:20]:
            print(f"  task={cfg.task} seed={cfg.seed} method={cfg.method}")
        if len(grid) > 20:
            print(f"  ... and {len(grid)-20} more")
        return

    grid = runner._build_grid(args.benchmark, seeds, args.methods or DEFAULT_METHODS,
                             plan_model=args.plan_model, task_indices=task_indices)
    for cfg in grid:
        cfg.frozen = args.frozen
        cfg.calibration_path = args.calibration_path
        cfg.active_calib_rate = args.active_calib_rate
        cfg.cact_kappa = str(args.kappa or "")
        if args.store_path:
            cfg.store_path = args.store_path
        if args.snapshot_path:
            cfg.snapshot_path = args.snapshot_path
        cfg.protocol_path = args.protocol_path
        cfg.branch_mode = args.branch_mode
        cfg.branch_target_opportunity = args.branch_target_opportunity
        cfg.branch_parent_id = args.branch_parent_id
        cfg.branch_prefix_assignment = args.branch_prefix_assignment
    runner._t_start = time.perf_counter()
    runner.run(
        benchmark=args.benchmark,
        seeds=seeds,
        methods=args.methods,
        plan_model=args.plan_model,
        resume=args.resume,
        grid=grid,
    )
    runner.print_summary()


if __name__ == "__main__":
    main()

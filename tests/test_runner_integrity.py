import json
import subprocess

import pytest

from experiments import parallel_runner
from experiments.parallel_runner import ExperimentConfig, ParallelRunner


def test_checkpoint_completed_contains_only_successful_runs(tmp_path, monkeypatch):
    runner = ParallelRunner(workers=1, checkpoint_dir=str(tmp_path / "ckpt"))
    grid = [
        ExperimentConfig("taskA", 0, 1001, "NoGate", "cact_e0", 12345, 0),
        ExperimentConfig("taskB", 1, 1001, "NoGate", "cact_e0", 12345, 0),
    ]

    def fake_run_one(cfg):
        status = "success" if cfg.task == "taskA" else "timeout"
        return {"key": (cfg.task, cfg.seed, cfg.method, cfg.frozen),
                "task": cfg.task, "seed": cfg.seed, "method": cfg.method,
                "status": status, "elapsed_sec": 0}

    monkeypatch.setattr(runner, "_start_vlm_server", lambda plan_model: None)
    monkeypatch.setattr(runner, "_start_batch_proxy", lambda: None)
    monkeypatch.setattr(runner, "_stop_vlm_server", lambda: None)
    monkeypatch.setattr(runner, "_run_one", fake_run_one)

    with pytest.raises(RuntimeError, match="experiment episodes failed"):
        runner.run(benchmark="cact_e0", grid=grid, resume=False)

    ckpt = json.loads((tmp_path / "ckpt" / "completed.json").read_text())
    assert ckpt["completed"] == [["taskA", 1001, "NoGate", False]]
    assert {row["status"] for row in ckpt["results"]} == {"success", "timeout"}

    resumed = ParallelRunner(workers=1, checkpoint_dir=str(tmp_path / "ckpt"))
    resumed._load_checkpoint()
    assert ("taskA", 1001, "NoGate", False) in resumed._completed
    assert ("taskB", 1001, "NoGate", False) not in resumed._completed


def test_frozen_timeout_records_mutation_hash_audit(tmp_path, monkeypatch):
    monkeypatch.setattr(parallel_runner, "_PROJ", str(tmp_path))
    store = tmp_path / "store"
    store.mkdir()
    (store / "state.json").write_text("before", encoding="utf-8")
    policy = tmp_path / "policy.json"
    policy.write_text("before", encoding="utf-8")

    def fake_subprocess_run(*args, **kwargs):
        (store / "state.json").write_text("after", encoding="utf-8")
        policy.write_text("after", encoding="utf-8")
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=1)

    monkeypatch.setattr(parallel_runner.subprocess, "run", fake_subprocess_run)
    runner = ParallelRunner(workers=1, checkpoint_dir=str(tmp_path / "ckpt"))
    cfg = ExperimentConfig("taskA", 0, 1001, "C-ACT", "cact_p3", 12345, 0,
                           timeout=1, store_path=str(store), frozen=True,
                           protocol_path=str(policy), run_id="timeout_audit")

    result = runner._run_one(cfg)

    assert result["status"] == "timeout"
    assert result["frozen_store_hash"]
    assert result["frozen_policy_hash"]
    assert "frozen store mutated" in result["frozen_error"]
    assert "frozen policy artifact mutated" in result["frozen_error"]


def test_main_grid_rejects_misspelled_cact_method():
    runner = ParallelRunner(workers=1)
    with pytest.raises(ValueError, match="unsupported C-ACT method"):
        runner._build_grid("cact_p3", [4001], ["C-ACT-Pointwis"],
                           task_indices=[0])

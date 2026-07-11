"""Paired hierarchical bootstrap utilities for C-ACT reports.

The resampling unit is task-template, then world seed within template; all
method rows for a task-seed remain paired.  Episode decisions are never split.
"""
from __future__ import annotations
import hashlib
from collections import defaultdict
from typing import Callable, Iterable, Mapping, Sequence
import numpy as np


def _seed(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "big")


def paired_hierarchical_bootstrap(rows: Sequence[Mapping], statistic: Callable[[list], float],
                                  replicates: int = 10000, seed: int = 17,
                                  strata_key: str = "stratum", task_key: str = "task_id",
                                  world_key: str = "world_seed") -> dict:
    """Return point estimate and percentile CI for a paired statistic.

    Rows are sampled by stratum -> task template -> world seed.  A row's
    episode_id is kept intact because the caller receives complete rows.
    """
    rows = list(rows)
    point = float(statistic(rows)) if rows else float("nan")
    strata = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for row in rows:
        strata[str(row.get(strata_key, "all"))][str(row.get(task_key, "unknown"))][str(row.get(world_key, "unknown"))].append(row)
    rng = np.random.default_rng(_seed(f"{seed}|{len(rows)}"))
    values = []
    for _ in range(int(replicates)):
        sampled = []
        for _, tasks in strata.items():
            task_names = list(tasks)
            if not task_names: continue
            for idx in rng.integers(0, len(task_names), size=len(task_names)):
                worlds = tasks[task_names[int(idx)]]
                world_names = list(worlds)
                if not world_names: continue
                for widx in rng.integers(0, len(world_names), size=len(world_names)):
                    sampled.extend(worlds[world_names[int(widx)]])
        values.append(float(statistic(sampled)) if sampled else float("nan"))
    finite = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if len(finite) < 2:
        return {"estimate": point, "low": float("nan"), "high": float("nan"), "replicates": len(finite)}
    return {"estimate": point, "low": float(np.quantile(finite, .025)),
            "high": float(np.quantile(finite, .975)), "replicates": len(finite)}

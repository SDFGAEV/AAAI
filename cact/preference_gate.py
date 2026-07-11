"""Auditable pairwise-preference admission baseline.

The model is deliberately small and deterministic.  It only consumes a
pre-registered D_pair-train artifact; no fallback labels are synthesized.
"""
from __future__ import annotations
import hashlib, json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

SCHEMA = "cact.preference.v1"
FIELDS = ("source", "type", "task_group", "failure_type", "risk_tier",
          "resource_scarcity", "boundary_status", "episode_phase",
          "prior_admission_bin", "prior_fallback_bin")


def _hash(s: str) -> int:
    return int.from_bytes(hashlib.sha256(s.encode("utf-8")).digest()[:8], "big")


def _features(row: Mapping[str, Any], width: int = 64) -> np.ndarray:
    x = np.zeros(width + 1, dtype=float); x[0] = 1.0
    for name in FIELDS:
        x[1 + (_hash(f"{name}={row.get(name, '')}") % width)] += 1.0
    return x


@dataclass
class PairwisePreferenceModel:
    weights: list[float]
    bias: float
    threshold: float = 0.5
    schema_version: str = SCHEMA
    train_rows: int = 0
    validation_rows: int = 0
    train_episode_count: int = 0
    validation_episode_count: int = 0

    @classmethod
    def fit(cls, rows: Sequence[Mapping[str, Any]], l2: float = 1.0):
        clean = [r for r in rows if r.get("preferred") in (0, 1)]
        if len(clean) < 20 or len({int(r["preferred"]) for r in clean}) < 2:
            raise ValueError("D_pair-train needs >=20 rows and both preference labels")
        x = np.vstack([_features(r) for r in clean])
        y = np.asarray([int(r["preferred"]) for r in clean], dtype=float)
        beta = np.zeros(x.shape[1], dtype=float)
        reg = l2 * np.eye(x.shape[1]); reg[0, 0] = 0.0
        for _ in range(80):
            p = 1.0 / (1.0 + np.exp(-np.clip(x @ beta, -30, 30)))
            w = np.maximum(p * (1 - p), 1e-4)
            try:
                step = np.linalg.solve(x.T @ (w[:, None] * x) + reg,
                                       x.T @ (p - y) + reg @ beta)
            except np.linalg.LinAlgError:
                break
            beta -= step
            if float(np.max(np.abs(step))) < 1e-6:
                break
        return cls(weights=beta[1:].tolist(), bias=float(beta[0]), train_rows=len(clean))

    def predict_proba(self, row: Mapping[str, Any]) -> float:
        x = _features(row)
        z = self.bias + float(np.dot(np.asarray(self.weights), x[1:]))
        return float(1.0 / (1.0 + np.exp(-np.clip(z, -30, 30))))

    def decide(self, row: Mapping[str, Any]) -> dict[str, Any]:
        p = self.predict_proba(row)
        return {"decision": "ADMIT" if p >= self.threshold else "FALLBACK",
                "preference_probability": p, "threshold": self.threshold}

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("schema_version") != SCHEMA:
            raise ValueError("unsupported pairwise preference artifact")
        return cls(**{k: data[k] for k in cls.__dataclass_fields__ if k in data})

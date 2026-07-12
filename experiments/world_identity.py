"""Deterministic identity for procedurally generated XENON worlds."""

from __future__ import annotations

import hashlib
import json
import os
from typing import Mapping, Optional


def generator_fingerprint(extra: Optional[Mapping[str, object]] = None) -> str:
    values = {
        "backend": "xenon",
        "generator": os.getenv("CACT_XENON_GENERATOR", "XENON.DefaultWorldGenerator(force_reset=True)"),
        "minecraft_version": os.getenv("CACT_MINECRAFT_VERSION", "unknown"),
        "server_jar_sha256": os.getenv("CACT_SERVER_JAR_SHA256", "unknown"),
        "xenon_commit": os.getenv("CACT_XENON_COMMIT", "unknown"),
        "generator_config_hash": os.getenv("CACT_WORLD_GENERATOR_CONFIG_HASH", "unknown"),
        "reset_protocol_version": os.getenv("CACT_RESET_PROTOCOL_VERSION", "xenon-reset-v1"),
    }
    if extra:
        values.update(dict(extra))
    return json.dumps(values, sort_keys=True, separators=(",", ":"))


def derive_snapshot_hash(task_id: object, world_seed: object, *, fingerprint: Optional[str] = None) -> str:
    payload = {"task_id": str(task_id), "world_seed": str(world_seed), "generator_fingerprint": fingerprint or generator_fingerprint()}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def ensure_snapshot_hash(task_id: object, world_seed: object, value: Optional[object]) -> str:
    if value is not None and str(value).strip():
        return str(value).strip()
    return derive_snapshot_hash(task_id, world_seed)

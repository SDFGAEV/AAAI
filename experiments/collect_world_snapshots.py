#!/usr/bin/env python3
"""Create a canonical world-snapshot manifest from real Minecraft saves.

The collector never creates a hash for a missing world.  A world path is
resolved from ``--world-root-template`` using ``{task_id}`` and
``{world_seed}``, then all non-volatile files are hashed in stable path order.
The resulting manifest is an input to E2/E1c/E3/E4/E5 and must be frozen.
"""
from __future__ import annotations
import argparse, hashlib, json, os
from pathlib import Path

VOLATILE_NAMES = {"session.lock", "uid.dat"}
VOLATILE_DIRS = {"logs", "crash-reports", "debug"}

def canonical_hash(world: Path) -> str:
    if not world.is_dir():
        raise FileNotFoundError(f"world snapshot directory not found: {world}")
    files = []
    for path in world.rglob("*"):
        if not path.is_file() or path.name in VOLATILE_NAMES:
            continue
        if any(part in VOLATILE_DIRS for part in path.relative_to(world).parts):
            continue
        files.append(path)
    if not files:
        raise ValueError(f"world snapshot contains no canonical files: {world}")
    h = hashlib.sha256()
    for path in sorted(files, key=lambda p: p.relative_to(world).as_posix()):
        rel = path.relative_to(world).as_posix().encode("utf-8")
        data = path.read_bytes()
        h.update(len(rel).to_bytes(8, "big")); h.update(rel)
        h.update(len(data).to_bytes(8, "big")); h.update(data)
    return h.hexdigest()

def parse_indices(value: str):
    out = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            lo, hi = [int(x) for x in token.split("-", 1)]
            out.extend(range(lo, hi + 1))
        else:
            out.append(int(token))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world-root-template", required=True,
                    help="Path template, e.g. /data/worlds/task_{task_id}_seed_{world_seed}")
    ap.add_argument("--task-indices", required=True)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--generator-version", default="CACTTaskEnv-v0:DefaultWorldGenerator-v1")
    args = ap.parse_args()
    tasks = parse_indices(args.task_indices)
    seeds = parse_indices(args.seeds)
    hashes = {}
    sources = {}
    for task_id in tasks:
        for seed in seeds:
            path = Path(args.world_root_template.format(task_id=task_id, world_seed=seed))
            key = f"{task_id}|{seed}"
            hashes[key] = canonical_hash(path)
            sources[key] = str(path)
    payload = {"schema_version": "cact.world_snapshot_manifest.v1", "sealed": True,
               "generator_version": args.generator_version, "hash_algorithm": "sha256-canonical-tree-v1",
               "hashes": hashes, "sources": sources}
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, out)
    print(json.dumps({"out": str(out), "cells": len(hashes), "sealed": True}))

if __name__ == "__main__":
    main()

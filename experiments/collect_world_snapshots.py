#!/usr/bin/env python3
"""Create a frozen identity manifest for filesystem or procedural worlds.

Filesystem mode resolves ``--world-root-template`` using ``{task_id}`` and
``{world_seed}`` and hashes all non-volatile files in stable path order; a
missing directory is a hard error. XENON procedural mode instead derives a
seed-plus-generator identity because ``force_reset=True`` creates the world at
reset time. The resulting manifest is optional for XENON and must be frozen
when supplied to E2/E1c/E3/E4/E5.
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.world_identity import derive_snapshot_hash, generator_fingerprint

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
    ap.add_argument("--world-root-template", default="",
                    help="Path template for filesystem snapshots; omit for XENON procedural mode")
    ap.add_argument("--procedural", action="store_true",
                    help="derive IDs from XENON seed and pinned generator provenance")
    ap.add_argument("--task-indices", required=True)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--generator-version", default="CACTTaskEnv-v0:DefaultWorldGenerator-v1")
    args = ap.parse_args()
    tasks = parse_indices(args.task_indices)
    seeds = parse_indices(args.seeds)
    if bool(args.world_root_template) == bool(args.procedural):
        raise SystemExit("choose exactly one of --world-root-template or --procedural")
    hashes = {}
    sources = {}
    for task_id in tasks:
        for seed in seeds:
            key = f"{task_id}|{seed}"
            if args.procedural:
                hashes[key] = derive_snapshot_hash(task_id, seed)
                sources[key] = f"xenon://DefaultWorldGenerator?task_id={task_id}&seed={seed}"
            else:
                path = Path(args.world_root_template.format(task_id=task_id, world_seed=seed))
                hashes[key] = canonical_hash(path)
                sources[key] = str(path)
    payload = {"schema_version": "cact.world_snapshot_manifest.v2", "sealed": True,
               "snapshot_backend": "xenon_procedural" if args.procedural else "filesystem",
               "generator_version": args.generator_version,
               "generator_fingerprint": generator_fingerprint() if args.procedural else args.generator_version,
               "hash_algorithm": "sha256-procedural-identity-v1" if args.procedural else "sha256-canonical-tree-v1",
               "hashes": hashes, "sources": sources}
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, out)
    print(json.dumps({"out": str(out), "cells": len(hashes), "sealed": True}))

if __name__ == "__main__":
    main()

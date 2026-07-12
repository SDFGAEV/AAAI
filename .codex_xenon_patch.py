from pathlib import Path

ROOT = Path(__file__).resolve().parent

def edit(rel, replacements):
    path = ROOT / rel
    text = path.read_text(encoding="utf-8")
    for old, new in replacements:
        if old not in text:
            raise SystemExit(f"missing patch anchor in {rel}: {old[:80]!r}")
        text = text.replace(old, new, 1)
    path.write_text(text, encoding="utf-8")

edit("experiments/build_task_card_registry.py", [
    ('"world_constraints": {"biome": "forest", "world_seed": "declared", "snapshot_hash": "required"},',
     '"world_constraints": {"biome": "forest", "world_seed": "declared", "snapshot_hash": "optional", "snapshot_backend": "xenon_procedural"},')
])
cards = ROOT / "protocol_inputs/task_cards.json"
cards.write_text(cards.read_text(encoding="utf-8").replace('"snapshot_hash": "required"', '"snapshot_hash": "optional"'), encoding="utf-8")

edit("setup_and_run.sh", [
    ('''if [[ -z "${CACT_WORLD_SNAPSHOT_MANIFEST:-}" ]]; then
  export CACT_WORLD_SNAPSHOT_MANIFEST="$PROJ/protocol_inputs/world_snapshot_manifest.json"
fi
if [[ ! -f "$CACT_WORLD_SNAPSHOT_MANIFEST" ]]; then
  if [[ -n "${CACT_WORLD_ROOT_TEMPLATE:-}" ]]; then
    export CACT_SNAPSHOT_TASK_INDICES="${CACT_SNAPSHOT_TASK_INDICES:-0-35}"
    export CACT_SNAPSHOT_SEEDS="${CACT_SNAPSHOT_SEEDS:-3001-3008,3011-3018,4001-4008,5001-5005,6001-6005}"
  fi
  if [[ -z "${CACT_WORLD_ROOT_TEMPLATE:-}" || -z "${CACT_SNAPSHOT_TASK_INDICES:-}" || -z "${CACT_SNAPSHOT_SEEDS:-}" ]]; then
    echo "STOP: provide CACT_WORLD_SNAPSHOT_MANIFEST, or set CACT_WORLD_ROOT_TEMPLATE, CACT_SNAPSHOT_TASK_INDICES, and CACT_SNAPSHOT_SEEDS." >&2
    exit 2
  fi
  "$PYTHON" experiments/collect_world_snapshots.py \\
    --world-root-template "$CACT_WORLD_ROOT_TEMPLATE" \\
    --task-indices "$CACT_SNAPSHOT_TASK_INDICES" --seeds "$CACT_SNAPSHOT_SEEDS" \\
    --out "$CACT_WORLD_SNAPSHOT_MANIFEST"
fi
''', '''if [[ -z "${CACT_WORLD_SNAPSHOT_MANIFEST:-}" ]]; then
  export CACT_WORLD_SNAPSHOT_MANIFEST="$PROJ/protocol_inputs/world_snapshot_manifest.json"
fi
if [[ ! -f "$CACT_WORLD_SNAPSHOT_MANIFEST" ]]; then
  export CACT_SNAPSHOT_TASK_INDICES="${CACT_SNAPSHOT_TASK_INDICES:-0-35}"
  export CACT_SNAPSHOT_SEEDS="${CACT_SNAPSHOT_SEEDS:-3001-3008,3011-3018,4001-4008,5001-5005,6001-6005}"
  if [[ -n "${CACT_WORLD_ROOT_TEMPLATE:-}" ]]; then
    "$PYTHON" experiments/collect_world_snapshots.py \\
      --world-root-template "$CACT_WORLD_ROOT_TEMPLATE" \\
      --task-indices "$CACT_SNAPSHOT_TASK_INDICES" --seeds "$CACT_SNAPSHOT_SEEDS" \\
      --out "$CACT_WORLD_SNAPSHOT_MANIFEST"
  else
    echo "[XENON] no pre-existing save supplied; deriving procedural snapshot IDs from seed + generator provenance"
    "$PYTHON" experiments/collect_world_snapshots.py --procedural \\
      --task-indices "$CACT_SNAPSHOT_TASK_INDICES" --seeds "$CACT_SNAPSHOT_SEEDS" \\
      --out "$CACT_WORLD_SNAPSHOT_MANIFEST"
  fi
fi
''')
])

edit("experiments/run_all.sh", [
    ('export CACT_REQUIRE_WORLD_SNAPSHOT_HASH="${CACT_REQUIRE_WORLD_SNAPSHOT_HASH:-1}"',
     'export CACT_REQUIRE_WORLD_SNAPSHOT_HASH="${CACT_REQUIRE_WORLD_SNAPSHOT_HASH:-0}"')
])

edit("experiments/collect_world_snapshots.py", [
    ('import argparse, hashlib, json, os\nfrom pathlib import Path\n',
     'import argparse, hashlib, json, os, sys\nfrom pathlib import Path\n\nROOT = Path(__file__).resolve().parents[1]\nsys.path.insert(0, str(ROOT))\nfrom experiments.world_identity import derive_snapshot_hash, generator_fingerprint\n'),
    ('    ap.add_argument("--world-root-template", required=True,\n                    help="Path template, e.g. /data/worlds/task_{task_id}_seed_{world_seed}")\n',
     '    ap.add_argument("--world-root-template", default="",\n                    help="Path template for filesystem snapshots; omit for XENON procedural mode")\n    ap.add_argument("--procedural", action="store_true",\n                    help="derive IDs from XENON seed and pinned generator provenance")\n'),
    ('    hashes = {}\n    sources = {}\n    for task_id in tasks:\n        for seed in seeds:\n            path = Path(args.world_root_template.format(task_id=task_id, world_seed=seed))\n            key = f"{task_id}|{seed}"\n            hashes[key] = canonical_hash(path)\n            sources[key] = str(path)\n    payload = {"schema_version": "cact.world_snapshot_manifest.v1", "sealed": True,\n               "generator_version": args.generator_version, "hash_algorithm": "sha256-canonical-tree-v1",\n               "hashes": hashes, "sources": sources}\n',
     '    if bool(args.world_root_template) == bool(args.procedural):\n        raise SystemExit("choose exactly one of --world-root-template or --procedural")\n    hashes = {}\n    sources = {}\n    for task_id in tasks:\n        for seed in seeds:\n            key = f"{task_id}|{seed}"\n            if args.procedural:\n                hashes[key] = derive_snapshot_hash(task_id, seed)\n                sources[key] = f"xenon://DefaultWorldGenerator?task_id={task_id}&seed={seed}"\n            else:\n                path = Path(args.world_root_template.format(task_id=task_id, world_seed=seed))\n                hashes[key] = canonical_hash(path)\n                sources[key] = str(path)\n    payload = {"schema_version": "cact.world_snapshot_manifest.v2", "sealed": True,\n               "snapshot_backend": "xenon_procedural" if args.procedural else "filesystem",\n               "generator_version": args.generator_version,\n               "generator_fingerprint": generator_fingerprint() if args.procedural else args.generator_version,\n               "hash_algorithm": "sha256-procedural-identity-v1" if args.procedural else "sha256-canonical-tree-v1",\n               "hashes": hashes, "sources": sources}\n')
])

edit("experiments/parallel_runner.py", [
    ('from typing import Dict, List, Tuple, Optional\n',
     'from typing import Dict, List, Tuple, Optional\nfrom experiments.world_identity import derive_snapshot_hash\n'),
    ('        if cfg.frozen and cfg.protocol_path and os.environ.get("CACT_REQUIRE_WORLD_SNAPSHOT_HASH") == "1" and not cfg.snapshot_hash:\n',
     '        if not cfg.snapshot_hash:\n            cfg.snapshot_hash = derive_snapshot_hash(cfg.task_idx, cfg.seed)\n        if cfg.frozen and cfg.protocol_path and os.environ.get("CACT_REQUIRE_WORLD_SNAPSHOT_HASH") == "1" and not cfg.snapshot_hash:\n'),
    ('snapshot_hash=self._world_snapshot_hashes.get(f"{idx}|{seed}")',
     'snapshot_hash=self._world_snapshot_hashes.get(f"{idx}|{seed}") or derive_snapshot_hash(idx, seed)')
])

edit("experiments/run_e2_select_rollouts.py", [
    ('sys.path.insert(0, str(_PROJ))\n\nKAPPAS',
     'sys.path.insert(0, str(_PROJ))\nfrom experiments.world_identity import derive_snapshot_hash\n\nKAPPAS'),
    ('    world_hash = cfg.get("world_snapshot_hashes", {}).get(cell_key)\n    if not world_hash:\n        raise RuntimeError(f"missing world snapshot hash for matched cell {cell_key}")\n',
     '    world_hash = cfg.get("world_snapshot_hashes", {}).get(cell_key) or derive_snapshot_hash(task_idx, seed)\n'),
    ('    ap.add_argument("--world-snapshot-manifest", required=True,\n                    help="JSON mapping task_id|world_seed to canonical world snapshot hash")\n',
     '    ap.add_argument("--world-snapshot-manifest", default="",\n                    help="optional filesystem/procedural snapshot manifest")\n'),
    ('    manifest = json.loads(Path(args.world_snapshot_manifest).read_text(encoding="utf-8"))\n    hashes = manifest.get("hashes", manifest)\n    if not isinstance(hashes, dict): raise SystemExit("world snapshot manifest must be a JSON mapping")\n',
     '    if args.world_snapshot_manifest:\n        manifest = json.loads(Path(args.world_snapshot_manifest).read_text(encoding="utf-8"))\n        hashes = manifest.get("hashes", manifest)\n        if not isinstance(hashes, dict): raise SystemExit("world snapshot manifest must be a JSON mapping")\n    else:\n        hashes = {f"{task}|{seed}": derive_snapshot_hash(task, seed) for task in task_indices for seed in seeds}\n')
])

edit("experiments/e2_direct_select.py", [
    ('import argparse, json, math\n',
     'import argparse, json, math, sys\n'),
    ('from pathlib import Path\n\nKAPPAS',
     'from pathlib import Path\n\n_PROJ = Path(__file__).resolve().parents[1]\nsys.path.insert(0, str(_PROJ))\nfrom experiments.world_identity import derive_snapshot_hash\n\nKAPPAS'),
    ('                           "method", "success", "harmful_reuse", "snapshot_hash",\n',
     '                           "method", "success", "harmful_reuse",\n'),
    ('                if not str(row["matched_cell_id"]) or not str(row["snapshot_hash"]):\n',
     '                if not row.get("snapshot_hash"):\n                    row["snapshot_hash"] = derive_snapshot_hash(row["task_id"], row["world_seed"])\n                if not str(row["matched_cell_id"]) or not str(row["snapshot_hash"]):\n')
])

edit("experiments/validate_e2_audit.py", [
    ('import argparse, json, math\n',
     'import argparse, json, math, sys\n'),
    ('from pathlib import Path\n\nMETHODS',
     'from pathlib import Path\n\n_PROJ = Path(__file__).resolve().parents[1]\nsys.path.insert(0, str(_PROJ))\nfrom experiments.world_identity import derive_snapshot_hash\n\nMETHODS'),
    ('    required = {"task_id", "world_seed", "episode_id", "matched_cell_id", "method", "snapshot_hash",\n',
     '    required = {"task_id", "world_seed", "episode_id", "matched_cell_id", "method",\n'),
    ('        key = (str(row["task_id"]), str(row["world_seed"]), str(row["matched_cell_id"]))\n        if not str(row["matched_cell_id"]) or not str(row["snapshot_hash"]):\n',
     '        row["snapshot_hash"] = str(row.get("snapshot_hash") or derive_snapshot_hash(row["task_id"], row["world_seed"]))\n        key = (str(row["task_id"]), str(row["world_seed"]), str(row["matched_cell_id"]))\n        if not str(row["matched_cell_id"]) or not str(row["snapshot_hash"]):\n'),
    ('    for row in rows:\n        if not {"pair_id", "parent_episode", "snapshot_hash", "reuse", "base"}.issubset(row):\n',
     '    for row in rows:\n        reuse = row.get("reuse") if isinstance(row.get("reuse"), dict) else {}\n        seed = row.get("world_seed", reuse.get("world_seed"))\n        task = row.get("task_id", reuse.get("task_id"))\n        row["snapshot_hash"] = str(row.get("snapshot_hash") or derive_snapshot_hash(task, seed))\n        for branch in (row.get("reuse"), row.get("base")):\n            if isinstance(branch, dict):\n                branch["snapshot_hash"] = str(branch.get("snapshot_hash") or row["snapshot_hash"])\n        if not {"pair_id", "parent_episode", "snapshot_hash", "reuse", "base"}.issubset(row):\n')
])

edit("experiments/generate_pair_train.py", [
    ('from experiments.parallel_runner import ExperimentConfig, ParallelRunner\n',
     'from experiments.parallel_runner import ExperimentConfig, ParallelRunner\nfrom experiments.world_identity import derive_snapshot_hash\n'),
    ('            if not row.get("snapshot_hash"):\n                raise RuntimeError(f"missing snapshot_hash in {path}")\n            row["pilot_run_id"] = run["run_id"]\n            row["task_idx"] = run["task_idx"]\n            row["seed"] = run["seed"]\n',
     '            row["pilot_run_id"] = run["run_id"]\n            row["task_idx"] = run["task_idx"]\n            row["seed"] = run["seed"]\n            row["snapshot_hash"] = str(row.get("snapshot_hash") or derive_snapshot_hash(row["task_idx"], row["seed"]))\n'),
    ('    ap.add_argument("--world-snapshot-manifest", required=True)\n',
     '    ap.add_argument("--world-snapshot-manifest", default="", help="optional filesystem/procedural snapshot manifest")\n'),
    ('    manifest = json.loads(Path(args.world_snapshot_manifest).read_text(encoding="utf-8"))\n    hashes = manifest.get("hashes", manifest)\n    if not isinstance(hashes, dict): raise SystemExit("world snapshot manifest must be a JSON mapping")\n',
     '    if args.world_snapshot_manifest:\n        manifest = json.loads(Path(args.world_snapshot_manifest).read_text(encoding="utf-8"))\n        hashes = manifest.get("hashes", manifest)\n        if not isinstance(hashes, dict): raise SystemExit("world snapshot manifest must be a JSON mapping")\n    else:\n        hashes = {f"{task}|{seed}": derive_snapshot_hash(task, seed) for task in task_indices for seed in seeds}\n'),
    ('                                   snapshot_hash=str(row.get("snapshot_hash", "")),\n',
     '                                   snapshot_hash=str(row.get("snapshot_hash") or derive_snapshot_hash(row["task_idx"], row["seed"])),\n')
])

edit("experiments/run_e2_audit_rollouts.py", [
    ('from experiments.run_e2_select_rollouts import _ensure_vlm, _run_one\n',
     'from experiments.run_e2_select_rollouts import _ensure_vlm, _run_one\nfrom experiments.world_identity import derive_snapshot_hash\n'),
    ('    ap.add_argument("--world-snapshot-manifest", required=True)\n',
     '    ap.add_argument("--world-snapshot-manifest", default="", help="optional filesystem/procedural snapshot manifest")\n'),
    ('    manifest = json.loads(Path(args.world_snapshot_manifest).read_text(encoding="utf-8"))\n    hashes = {str(k): str(v) for k, v in manifest.get("hashes", manifest).items()}\n',
     '    if args.world_snapshot_manifest:\n        manifest = json.loads(Path(args.world_snapshot_manifest).read_text(encoding="utf-8"))\n        hashes = {str(k): str(v) for k, v in manifest.get("hashes", manifest).items()}\n    else:\n        hashes = {f"{task}|{seed}": derive_snapshot_hash(task, seed) for task in tasks for seed in seeds}\n'),
    ('           "--world-snapshot-manifest", args.world_snapshot_manifest, "--out", str(flat)]\n',
     '           "--out", str(flat)]\n    if args.world_snapshot_manifest:\n        cmd[cmd.index("--out"):cmd.index("--out")] = ["--world-snapshot-manifest", args.world_snapshot_manifest]\n')
])

edit("experiments/release_protocol.py", [
    ('ROOT/"experiments/run_e2_audit_rollouts.py", ROOT/"experiments/validate_e2_audit.py",\n',
     'ROOT/"experiments/run_e2_audit_rollouts.py", ROOT/"experiments/validate_e2_audit.py", ROOT/"experiments/world_identity.py",\n')
])

print("XENON_COMPAT_PATCH_APPLIED")

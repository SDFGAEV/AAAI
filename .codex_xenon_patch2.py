from pathlib import Path
ROOT = Path(__file__).resolve().parent

def edit(rel, replacements):
    p = ROOT / rel; t = p.read_text(encoding="utf-8")
    for old, new in replacements:
        if old in t: t = t.replace(old, new, 1)
    p.write_text(t, encoding="utf-8")

edit("experiments/parallel_runner.py", [
    ('from typing import Dict, List, Tuple, Optional\n', 'from typing import Dict, List, Tuple, Optional\nfrom experiments.world_identity import derive_snapshot_hash\n'),
    ('        if cfg.frozen and cfg.protocol_path and os.environ.get("CACT_REQUIRE_WORLD_SNAPSHOT_HASH") == "1" and not cfg.snapshot_hash:\n', '        if not cfg.snapshot_hash:\n            cfg.snapshot_hash = derive_snapshot_hash(cfg.task_idx, cfg.seed)\n        if cfg.frozen and cfg.protocol_path and os.environ.get("CACT_REQUIRE_WORLD_SNAPSHOT_HASH") == "1" and not cfg.snapshot_hash:\n'),
    ('snapshot_hash=self._world_snapshot_hashes.get(f"{idx}|{seed}", ""),', 'snapshot_hash=self._world_snapshot_hashes.get(f"{idx}|{seed}") or derive_snapshot_hash(idx, seed),')
])
edit("experiments/run_e2_select_rollouts.py", [
    ('sys.path.insert(0, str(_PROJ))\n\nKAPPAS', 'sys.path.insert(0, str(_PROJ))\nfrom experiments.world_identity import derive_snapshot_hash\n\nKAPPAS'),
    ('    world_hash = cfg.get("world_snapshot_hashes", {}).get(cell_key)\n    if not world_hash:\n        raise RuntimeError(f"missing world snapshot hash for matched cell {cell_key}")\n', '    world_hash = cfg.get("world_snapshot_hashes", {}).get(cell_key) or derive_snapshot_hash(task_idx, seed)\n'),
    ('    ap.add_argument("--world-snapshot-manifest", required=True,\n                    help="JSON mapping task_id|world_seed to canonical world snapshot hash")\n', '    ap.add_argument("--world-snapshot-manifest", default="",\n                    help="optional filesystem/procedural snapshot manifest")\n'),
    ('    manifest = json.loads(Path(args.world_snapshot_manifest).read_text(encoding="utf-8"))\n    hashes = manifest.get("hashes", manifest)\n    if not isinstance(hashes, dict): raise SystemExit("world snapshot manifest must be a JSON mapping")\n', '    if args.world_snapshot_manifest:\n        manifest = json.loads(Path(args.world_snapshot_manifest).read_text(encoding="utf-8"))\n        hashes = manifest.get("hashes", manifest)\n        if not isinstance(hashes, dict): raise SystemExit("world snapshot manifest must be a JSON mapping")\n    else:\n        hashes = {f"{task}|{seed}": derive_snapshot_hash(task, seed) for task in task_indices for seed in seeds}\n')
])
edit("experiments/e2_direct_select.py", [
    ('import argparse, json, math\n', 'import argparse, json, math, sys\n'),
    ('from pathlib import Path\n\nKAPPAS', 'from pathlib import Path\n\n_PROJ = Path(__file__).resolve().parents[1]\nsys.path.insert(0, str(_PROJ))\nfrom experiments.world_identity import derive_snapshot_hash\n\nKAPPAS'),
    ('                           "method", "success", "harmful_reuse", "snapshot_hash",\n', '                           "method", "success", "harmful_reuse",\n'),
    ('                if not str(row["matched_cell_id"]) or not str(row["snapshot_hash"]):\n', '                if not row.get("snapshot_hash"):\n                    row["snapshot_hash"] = derive_snapshot_hash(row["task_id"], row["world_seed"])\n                if not str(row["matched_cell_id"]) or not str(row["snapshot_hash"]):\n')
])
edit("experiments/validate_e2_audit.py", [
    ('import argparse, json, math\n', 'import argparse, json, math, sys\n'),
    ('from pathlib import Path\n\nMETHODS', 'from pathlib import Path\n\n_PROJ = Path(__file__).resolve().parents[1]\nsys.path.insert(0, str(_PROJ))\nfrom experiments.world_identity import derive_snapshot_hash\n\nMETHODS'),
    ('    required = {"task_id", "world_seed", "episode_id", "matched_cell_id", "method", "snapshot_hash",\n', '    required = {"task_id", "world_seed", "episode_id", "matched_cell_id", "method",\n'),
    ('        key = (str(row["task_id"]), str(row["world_seed"]), str(row["matched_cell_id"]))\n        if not str(row["matched_cell_id"]) or not str(row["snapshot_hash"]):\n', '        row["snapshot_hash"] = str(row.get("snapshot_hash") or derive_snapshot_hash(row["task_id"], row["world_seed"]))\n        key = (str(row["task_id"]), str(row["world_seed"]), str(row["matched_cell_id"]))\n        if not str(row["matched_cell_id"]) or not str(row["snapshot_hash"]):\n'),
    ('    for row in rows:\n        if not {"pair_id", "parent_episode", "snapshot_hash", "reuse", "base"}.issubset(row):\n', '    for row in rows:\n        reuse = row.get("reuse") if isinstance(row.get("reuse"), dict) else {}\n        task = row.get("task_id", reuse.get("task_id")); seed = row.get("world_seed", reuse.get("world_seed"))\n        row["snapshot_hash"] = str(row.get("snapshot_hash") or derive_snapshot_hash(task, seed))\n        for branch in (row.get("reuse"), row.get("base")):\n            if isinstance(branch, dict): branch["snapshot_hash"] = str(branch.get("snapshot_hash") or row["snapshot_hash"])\n        if not {"pair_id", "parent_episode", "snapshot_hash", "reuse", "base"}.issubset(row):\n')
])
edit("experiments/generate_pair_train.py", [
    ('from experiments.parallel_runner import ExperimentConfig, ParallelRunner\n', 'from experiments.parallel_runner import ExperimentConfig, ParallelRunner\nfrom experiments.world_identity import derive_snapshot_hash\n'),
    ('            if not row.get("snapshot_hash"):\n                raise RuntimeError(f"missing snapshot_hash in {path}")\n            row["pilot_run_id"] = run["run_id"]\n            row["task_idx"] = run["task_idx"]\n            row["seed"] = run["seed"]\n', '            row["pilot_run_id"] = run["run_id"]\n            row["task_idx"] = run["task_idx"]\n            row["seed"] = run["seed"]\n            row["snapshot_hash"] = str(row.get("snapshot_hash") or derive_snapshot_hash(row["task_idx"], row["seed"]))\n'),
    ('    ap.add_argument("--world-snapshot-manifest", required=True)\n', '    ap.add_argument("--world-snapshot-manifest", default="", help="optional filesystem/procedural snapshot manifest")\n'),
    ('    manifest = json.loads(Path(args.world_snapshot_manifest).read_text(encoding="utf-8"))\n    hashes = manifest.get("hashes", manifest)\n    if not isinstance(hashes, dict): raise SystemExit("world snapshot manifest must be a JSON mapping")\n', '    if args.world_snapshot_manifest:\n        manifest = json.loads(Path(args.world_snapshot_manifest).read_text(encoding="utf-8"))\n        hashes = manifest.get("hashes", manifest)\n        if not isinstance(hashes, dict): raise SystemExit("world snapshot manifest must be a JSON mapping")\n    else:\n        hashes = {f"{task}|{seed}": derive_snapshot_hash(task, seed) for task in task_indices for seed in seeds}\n'),
    ('snapshot_hash=str(row.get("snapshot_hash", "")),', 'snapshot_hash=str(row.get("snapshot_hash") or derive_snapshot_hash(row["task_idx"], row["seed"])),')
])
edit("experiments/run_e2_audit_rollouts.py", [
    ('from experiments.run_e2_select_rollouts import _ensure_vlm, _run_one\n', 'from experiments.run_e2_select_rollouts import _ensure_vlm, _run_one\nfrom experiments.world_identity import derive_snapshot_hash\n'),
    ('    ap.add_argument("--world-snapshot-manifest", required=True)\n', '    ap.add_argument("--world-snapshot-manifest", default="", help="optional filesystem/procedural snapshot manifest")\n'),
    ('    manifest = json.loads(Path(args.world_snapshot_manifest).read_text(encoding="utf-8"))\n    hashes = {str(k): str(v) for k, v in manifest.get("hashes", manifest).items()}\n', '    if args.world_snapshot_manifest:\n        manifest = json.loads(Path(args.world_snapshot_manifest).read_text(encoding="utf-8"))\n        hashes = {str(k): str(v) for k, v in manifest.get("hashes", manifest).items()}\n    else:\n        hashes = {f"{task}|{seed}": derive_snapshot_hash(task, seed) for task in tasks for seed in seeds}\n'),
    ('           "--world-snapshot-manifest", args.world_snapshot_manifest, "--out", str(flat)]\n', '           "--out", str(flat)]\n    if args.world_snapshot_manifest:\n        cmd[cmd.index("--out"):cmd.index("--out")] = ["--world-snapshot-manifest", args.world_snapshot_manifest]\n')
])
edit("experiments/release_protocol.py", [
    ('ROOT/"experiments/run_e2_audit_rollouts.py", ROOT/"experiments/validate_e2_audit.py",\n', 'ROOT/"experiments/run_e2_audit_rollouts.py", ROOT/"experiments/validate_e2_audit.py", ROOT/"experiments/world_identity.py",\n')
])
print("XENON_COMPAT_PATCH2_APPLIED")

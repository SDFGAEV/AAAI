#!/usr/bin/env python3
"""Audit split leakage across role manifests and opportunity logs."""
import argparse, csv, json
from pathlib import Path
from collections import defaultdict

KEYS = ("exact_template_hash", "world_seed", "snapshot_hash", "branch_parent_id")

def rows(path: Path):
    if path.suffix.lower() == ".jsonl":
        return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, list) else data.get("rows", data.get("episodes", []))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--manifest", nargs="+", required=True); ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(); seen = defaultdict(list)
    for manifest in args.manifest:
        for row in rows(Path(manifest)):
            role = row.get("role", row.get("split", Path(manifest).stem))
            for key in KEYS:
                value = row.get(key)
                if value not in (None, "", "unknown"): seen[(key, str(value))].append(role)
    collisions = [{"field": k, "value": v, "roles": sorted(set(rs))}
                  for (k, v), rs in seen.items() if len(set(rs)) > 1]
    result = {"files": args.manifest, "collisions": collisions, "passed": not collisions}
    args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"passed": not collisions, "collisions": len(collisions)}))
    raise SystemExit(0 if not collisions else 2)

if __name__ == "__main__": main()

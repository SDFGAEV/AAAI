#!/usr/bin/env python3
"""Validate the manual's required task-card fields without inventing defaults."""
import argparse, json
from pathlib import Path

REQUIRED = ("task_id", "split", "task_group", "exact_template_hash",
            "world_generator_version", "initial_inventory", "world_constraints",
            "goal", "success_predicate", "window_trigger", "episode_budget",
            "allowed_randomization_state")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cards", nargs="+", type=Path); ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(); errors = []
    for path in args.cards:
        data = json.loads(path.read_text(encoding="utf-8")) if path.suffix == ".json" else {}
        for field in REQUIRED:
            if field not in data or data[field] in (None, "", {}): errors.append({"file": str(path), "missing": field})
    result = {"files": len(args.cards), "errors": errors, "passed": not errors}
    args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"passed": not errors, "errors": len(errors)})); raise SystemExit(0 if not errors else 2)

if __name__ == "__main__": main()

#!/usr/bin/env python3
"""Validate the manual's required task-card fields without inventing defaults."""
import argparse, json
from pathlib import Path

REQUIRED = ("task_id", "split", "task_group", "exact_template_hash",
            "world_generator_version", "initial_inventory", "world_constraints",
            "goal", "success_predicate", "window_trigger", "episode_budget",
            "allowed_randomization_state")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cards", nargs="+", type=Path); ap.add_argument("--out", type=Path, required=True); ap.add_argument("--require-sealed", action="store_true")
    args = ap.parse_args(); errors = []; loaded = []; seen_ids = set()
    for path in args.cards:
        if path.suffix.lower() == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
        elif path.suffix.lower() in {".yaml", ".yml"}:
            try:
                import yaml
            except ModuleNotFoundError as exc:
                raise SystemExit("PyYAML is required to validate YAML task cards; refusing silent skip") from exc
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        else:
            errors.append({"file": str(path), "missing": "unsupported_card_format"})
            data = {}
        if args.require_sealed and (not isinstance(data, dict) or data.get("sealed") is not True):
            errors.append({"file": str(path), "missing": "sealed:true"})
        if isinstance(data, dict) and isinstance(data.get("cards"), list):
            data = data["cards"]
        if isinstance(data, list):
            entries = [(f"{path}#{i}", item) for i, item in enumerate(data)]
        elif isinstance(data, dict):
            entries = [(str(path), data)]
        else:
            errors.append({"file": str(path), "missing": "task_card_mapping"})
            entries = []
        for label, card in entries:
            if not isinstance(card, dict):
                errors.append({"file": label, "missing": "task_card_mapping"})
                continue
            task_id = str(card.get("task_id", ""))
            if task_id and task_id in seen_ids:
                errors.append({"file": label, "missing": f"duplicate_task_id:{task_id}"})
            elif task_id:
                seen_ids.add(task_id)
            for field in REQUIRED:
                if field not in card or card[field] in (None, "", {}): errors.append({"file": label, "missing": field})
            loaded.append(label)
    result = {"files": len(args.cards), "cards": len(loaded), "errors": errors, "passed": not errors and bool(loaded)}
    args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"passed": result["passed"], "errors": len(errors), "cards": len(loaded)})); raise SystemExit(0 if result["passed"] else 2)

if __name__ == "__main__": main()

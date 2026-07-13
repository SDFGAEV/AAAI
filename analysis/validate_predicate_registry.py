#!/usr/bin/env python3
"""Validate the frozen deterministic predicate registry required by the protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED = {"kind", "input_fields", "parameters", "missing", "unit_test"}


def load(path: Path | str) -> dict:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml
        except ModuleNotFoundError as exc:
            raise SystemExit("registry must be JSON-compatible YAML when PyYAML is unavailable") from exc
        data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError("predicate registry must be a mapping")
    return data


def validate(data: dict) -> list[str]:
    errors: list[str] = []
    if data.get("schema_version") != "cact.predicate_registry.v1":
        errors.append("schema_version must be cact.predicate_registry.v1")
    if data.get("missing_value_policy") != "fail_closed":
        errors.append("missing_value_policy must be fail_closed")
    if not isinstance(data.get("version"), int) or data.get("version") < 1:
        errors.append("version must be a positive integer")
    predicates = data.get("predicates")
    if not isinstance(predicates, dict) or not predicates:
        return errors + ["predicates must be a non-empty mapping"]
    for name, spec in predicates.items():
        if not isinstance(name, str) or not name:
            errors.append("predicate names must be non-empty strings")
            continue
        if not isinstance(spec, dict):
            errors.append(f"{name}: specification must be a mapping")
            continue
        missing = REQUIRED - set(spec)
        errors.extend(f"{name}: missing {field}" for field in sorted(missing))
        fields = spec.get("input_fields")
        if not isinstance(fields, list) or not fields or not all(isinstance(x, str) and x for x in fields):
            errors.append(f"{name}: input_fields must be a non-empty list of strings")
        if not isinstance(spec.get("parameters"), dict):
            errors.append(f"{name}: parameters must be a mapping")
        if not isinstance(spec.get("missing"), bool):
            errors.append(f"{name}: missing must be boolean")
        if not isinstance(spec.get("unit_test"), str) or not spec["unit_test"]:
            errors.append(f"{name}: unit_test must be a non-empty string")
        if spec.get("kind") in {"llm_judge", "free_text", "natural_language"}:
            errors.append(f"{name}: free-form judgements are forbidden for primary predicates")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("registry", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        errors = validate(load(args.registry))
    except (OSError, ValueError) as exc:
        errors = [str(exc)]
    result = {"registry": str(args.registry), "errors": errors, "passed": not errors}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"passed": result["passed"], "errors": len(errors)}))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

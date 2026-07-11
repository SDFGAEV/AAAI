#!/usr/bin/env python3
"""Fail-fast validation for cact.v2 opportunity JSONL artifacts."""
import argparse, json
from pathlib import Path

REQUIRED = {
    "schema_version", "episode_id", "opportunity_id", "task_id", "world_seed",
    "knowledge_id", "assignment", "propensity_reuse", "propensity_base",
    "randomization_seed", "eligible", "eligibility_reason", "round",
    "stream_seed", "source", "type", "retrieval_rank", "retrieval_score",
    "raw_text_hash", "task_group", "failure_type", "risk_tier",
    "resource_scarcity", "boundary_status", "inventory_signature",
    "start_step", "end_step", "censor_flag", "second_intervention_flag",
    "window_type", "label_source", "annotator_status", "exclusion_reason",
    "snapshot_hash",
}


def validate(path: Path) -> int:
    count = 0
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = sorted(REQUIRED - row.keys())
            if missing:
                raise ValueError(f"{path}:{lineno}: missing {missing}")
            if row["schema_version"] != "cact.v2":
                raise ValueError(f"{path}:{lineno}: unsupported schema")
            if row["eligible"]:
                p = float(row["propensity_reuse"])
                if not 0.2 <= p <= 0.8 or abs(p + float(row["propensity_base"]) - 1) > 1e-6:
                    raise ValueError(f"{path}:{lineno}: positivity violation")
            count += 1
    return count


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("paths", nargs="+", type=Path)
    args = ap.parse_args(); total = sum(validate(p) for p in args.paths)
    print(json.dumps({"files": len(args.paths), "rows": total}))


if __name__ == "__main__":
    main()

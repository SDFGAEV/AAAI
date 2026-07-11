#!/usr/bin/env python3
"""Train PairwisePreferenceGate from a sealed D_pair-train JSONL artifact."""
import argparse, glob, json, hashlib
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cact.preference_gate import PairwisePreferenceModel, SCHEMA


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", nargs="+", required=True,
                    help="D_pair-train JSONL files; each row needs preferred=0/1")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows = []
    for pattern in args.input:
        for path in sorted(glob.glob(pattern)):
            with open(path, encoding="utf-8") as fh:
                rows.extend(json.loads(line) for line in fh if line.strip())
    if len(rows) != 320:
        raise SystemExit(f"D_pair-train requires exactly 320 paired opportunities, got {len(rows)}")
    groups = sorted({str(r.get("parent_episode", r.get("episode_id", ""))) for r in rows})
    if len(groups) < 2 or "" in groups:
        raise SystemExit("D_pair-train requires non-empty parent episode IDs for leakage-safe splitting")
    by_group = {g: [r for r in rows if str(r.get("parent_episode", r.get("episode_id", ""))) == g] for g in groups}
    ordered = sorted(groups, key=lambda g: hashlib.sha256(g.encode()).hexdigest())
    val_groups, count = set(), 0
    for g in ordered:
        size = len(by_group[g])
        if count + size <= 80:
            val_groups.add(g); count += size
        if count == 80: break
    if count != 80:
        raise SystemExit(f"parent-episode groups cannot form exactly 80 validation rows (got {count})")
    train = [r for r in rows if str(r.get("parent_episode", r.get("episode_id", ""))) not in val_groups]
    valid = [r for r in rows if str(r.get("parent_episode", r.get("episode_id", ""))) in val_groups]
    model = PairwisePreferenceModel.fit(train)
    candidates = [i / 20 for i in range(5, 16)]
    scored = [(sum((model.predict_proba(r) >= t) == bool(r["preferred"]) for r in valid) / len(valid), t) for t in candidates] if valid else [(0.0, 0.5)]
    model.threshold = max(scored)[1]
    model.validation_rows = len(valid); model.train_episode_count = len(groups) - len(val_groups); model.validation_episode_count = len(val_groups)
    model.save(args.out)
    print(json.dumps({"schema_version": SCHEMA, "rows": len(rows), "train_rows": len(train), "validation_rows": len(valid), "threshold": model.threshold, "out": args.out}))


if __name__ == "__main__":
    main()

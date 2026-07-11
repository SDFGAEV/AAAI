#!/usr/bin/env python3
"""Train PairwisePreferenceGate from a sealed D_pair-train JSONL artifact."""
import argparse, glob, json
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
    model = PairwisePreferenceModel.fit(rows)
    model.save(args.out)
    print(json.dumps({"schema_version": SCHEMA, "rows": len(rows), "out": args.out}))


if __name__ == "__main__":
    main()

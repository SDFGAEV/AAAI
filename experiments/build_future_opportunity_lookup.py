#!/usr/bin/env python3
"""Build the frozen future-opportunity lookup used by CAP q_t.

Only opportunity order and eligibility are used; outcomes and policy decisions
are deliberately ignored to prevent leakage from D_select/D_audit.
"""
from __future__ import annotations
import argparse, glob, hashlib, json
from collections import defaultdict
from pathlib import Path

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="JSONL files or globs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--clip", type=int, default=6)
    args = ap.parse_args()
    paths = sorted({str(p) for pattern in args.inputs for p in glob.glob(pattern, recursive=True)})
    if not paths:
        raise SystemExit("no opportunity logs matched")
    episodes = defaultdict(list)
    for path in paths:
        for lineno, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            for key in ("episode_id", "opportunity_id"):
                if not row.get(key):
                    raise ValueError(f"{path}:{lineno}: missing {key}")
            episodes[str(row["episode_id"])].append(row)
    lookup = {}
    for episode_id, rows in episodes.items():
        rows.sort(key=lambda r: (int(r.get("start_step", 0) or 0), str(r["opportunity_id"])))
        for i, row in enumerate(rows):
            future = sum(1 for other in rows[i + 1:] if bool(other.get("eligible", True)) and
                         not bool(other.get("censor_flag", False)) and
                         not bool(other.get("second_intervention_flag", False)))
            lookup[str(row["opportunity_id"])] = max(0, min(int(args.clip), future))
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.encode())
        digest.update(Path(path).read_bytes())
    payload = {"schema_version": "cact.future_opportunity_lookup.v1",
               "clip": int(args.clip), "source_hash": digest.hexdigest(),
               "episodes": len(episodes), "opportunities": len(lookup), "lookup": lookup}
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ("schema_version", "episodes", "opportunities", "source_hash")}))

if __name__ == "__main__":
    main()

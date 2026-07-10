#!/usr/bin/env python3
"""Create a hash-addressed protocol_release before formal data collection."""
import argparse, csv, hashlib, json, re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANUAL = ROOT.parent / "C-ACT_完整研究协议与论文写作手册.md"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_tasks(path: Path):
    tasks = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.lstrip().startswith("- {"):
            continue
        row = {}
        for key, quoted, bare in re.findall(r"([A-Za-z_]+):\s*(?:[\"']([^\"']*)[\"']|([^,}]+))", line[2:]):
            row[key] = (quoted or bare).strip()
        if row:
            tasks.append(row)
    return tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "protocol_release"))
    ap.add_argument("--label", default="protocol-candidate")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    files = [MANUAL, ROOT / "cact" / "protocol_v2.py", ROOT / "experiments" / "calibrate_v2.py",
             ROOT / "experiments" / "parallel_runner.py", ROOT / "experiments" / "run_all.sh",
             ROOT / "paper" / "PAPER_SUBMISSION_DRAFT_V2.md"]
    manifest = {
        "schema_version": "cact.protocol_release.v1",
        "label": args.label,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "method": "C-ACT: Contextual Admission via Counterfactual Treatment Effects",
        "frozen_claims": ["reliable admission", "applicability boundary and adaptive risk-coverage"],
        "hashes": {str(p.relative_to(ROOT.parent)): sha256(p) for p in files if p.exists()},
        "seeds": {
            "E0": [1001, 1002], "E1a": list(range(2001, 2006)),
            "E1b": list(range(2101, 2106)), "D_select": list(range(3001, 3009)),
            "D_audit": list(range(3011, 3019)), "E3": list(range(4001, 4009)),
            "E4": list(range(5001, 5006)), "E5": [6001, 6002, 6003, 6004, 6005],
        },
        "budgets": {"delta": 0.05, "eps_abs": 0.10, "eps_inc": 0.02,
                     "kappas": [0, .5, 1, 1.5, 2, 2.5, 3], "support_n": 12, "ess": 24},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    tasks = parse_tasks(ROOT / "src" / "optimus1" / "conf" / "benchmark" / "cact_p3.yaml")
    (out / "task_registry.json").write_text(json.dumps({"benchmark": "cact_p3", "tasks": tasks}, indent=2, ensure_ascii=False), encoding="utf-8")
    with (out / "deviation_log.csv").open("w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerow(["date", "proposer", "stage", "original_protocol", "change", "reason", "outcome_seen", "scope", "approver"])
    print(json.dumps({"release": str(out), "files": len(manifest["hashes"]), "tasks": len(tasks)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

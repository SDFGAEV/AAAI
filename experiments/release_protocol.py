#!/usr/bin/env python3
import argparse, csv, hashlib, json, re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANUAL = next((x for x in ROOT.parent.glob("C-ACT_*\u624b\u518c*.md") if "\u65b0" in x.name), ROOT.parent / "C-ACT_\u5b8c\u6574\u7814\u7a76\u534f\u8bae\u4e0e\u8bba\u6587\u5199\u4f5c\u624b\u518c.md")

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""): h.update(chunk)
    return h.hexdigest()

def parse_tasks(path):
    try:
        import yaml
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return [dict(t) for t in data.get("all_task", []) if isinstance(t, dict)]
    except ModuleNotFoundError:
        tasks, current = [], None
        for raw in Path(path).read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if line.startswith("- {"):
                row = {}
                for key, quoted, bare in re.findall(r"([A-Za-z_]+):\s*(?:[\"']([^\"']*)[\"']|([^,}]+))", line[2:]): row[key] = (quoted or bare).strip()
                tasks.append(row); current = row
            elif line.startswith("- id:"):
                current = {"id": line.split(":", 1)[1].strip()}; tasks.append(current)
            elif current is not None and ":" in line:
                key, value = line.split(":", 1)
                if key.strip() in {"type", "instruction", "goal", "group", "difficulty"}: current[key.strip()] = value.strip().strip("\"'")
        return tasks

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=str(ROOT / "protocol_release")); ap.add_argument("--label", default="protocol-candidate"); args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    files = [MANUAL, ROOT/"README.md", ROOT/"protocol_inputs/task_cards.json", ROOT/"setup_and_run.sh", ROOT/"cact/protocol_v2.py", ROOT/"experiments/calibrate_v2.py", ROOT/"experiments/parallel_runner.py", ROOT/"experiments/run_all.sh", ROOT/"paper/main.tex", ROOT/"paper/PAPER_SUBMISSION_DRAFT.md", ROOT/"paper/PAPER_SUBMISSION_DRAFT_NEW.md", ROOT/"paper/PAPER_SUBMISSION_DRAFT_V2.md", ROOT/"cact/preference_gate.py", ROOT/"experiments/train_pairwise.py", ROOT/"cact/metrics.py", ROOT/"analysis/bootstrap.py", ROOT/"analysis/split_audit.py", ROOT/"analysis/validate_task_cards.py",
           ROOT/"experiments/e2_direct_select.py", ROOT/"experiments/run_e2_select_rollouts.py",
           ROOT/"experiments/generate_pair_train.py", ROOT/"experiments/build_task_card_registry.py", ROOT/"experiments/collect_world_snapshots.py",
           ROOT/"experiments/run_e2_audit_rollouts.py", ROOT/"experiments/validate_e2_audit.py", ROOT/"experiments/world_identity.py",
           ROOT/"experiments/online_runner.py", ROOT/"experiments/batch_proxy.py",
           ROOT/"app.py", ROOT/"src/optimus1/util/server_api.py",
           ROOT/"src/optimus1/server/api/utils.py", ROOT/"src/optimus1/env/__init__.py",
           ROOT/"src/optimus1/env/custom_env.py", ROOT/"minerl/minerl/env/malmo.py",
           ROOT/"src/optimus1/models/steve1/embed_conditioned_policy.py",
           ROOT/"src/optimus1/models/steve1/VPT/lib/policy.py",
           ROOT/"src/optimus1/models/steve1/VPT/lib/impala_cnn.py",
           ROOT/"src/optimus1/models/steve1/VPT/lib/util.py", ROOT/"minerl/minerl/env/_multiagent.py",
           ROOT/"experiments/release_protocol.py", ROOT/"experiments/health_check.py",
           ROOT/"tests/test_controller_ledger.py",
           ROOT/"docs/UBUNTU_PERFORMANCE.md"]
    manifest = {"schema_version":"cact.protocol_release.v1", "label":args.label, "created_utc":datetime.now(timezone.utc).isoformat(), "method":"C-ACT: Contextual Admission via Counterfactual Treatment Effects", "frozen_claims":["reliable admission", "applicability boundary and adaptive risk-coverage"], "hashes":{str(p.relative_to(ROOT.parent)):sha256(p) for p in files if p.exists()}, "seeds":{"E0":[1001,1002],"E1a":list(range(2001,2006)),"E1b":list(range(2101,2106)),"D_select":list(range(3001,3009)),"D_audit":list(range(3011,3019)),"E3":list(range(4001,4009)),"E4":list(range(5001,5006)),"E5":[6001,6002,6003,6004,6005]}, "budgets":{"delta":.05,"eps_abs":.10,"eps_inc":.02,"kappas":[0,.5,1,1.5,2,2.5,3],"support_n":12,"ess":24}}
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding="utf-8")
    benchmarks = {}
    for name in ("cact_e0","cact_train","cact_calib","cact_p3","cact_ablation","cact_online_stream","cact_online_retention","cact_online_hard_transfer"):
        path = ROOT/"src/optimus1/conf/benchmark"/f"{name}.yaml"
        if path.exists(): benchmarks[name] = {"hash":sha256(path),"tasks":parse_tasks(path)}
    (out/"task_registry.json").write_text(json.dumps({"benchmarks":benchmarks},indent=2,ensure_ascii=False),encoding="utf-8")
    (out/"substrate_manifest.json").write_text(json.dumps({"schema_version":"cact.substrate.v1","commit":"unknown","benchmarks":{k:v["hash"] for k,v in benchmarks.items()},"planner":"same base planner","retriever":"fixed top-1","environment":"CACTTaskEnv-v0"},indent=2),encoding="utf-8")
    deviation_log = out / "deviation_log.csv"
    if not deviation_log.exists():
        with deviation_log.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["date","proposer","stage","original_protocol","change","reason","outcome_seen","scope","approver"])
    print(json.dumps({"release":str(out),"files":len(manifest["hashes"]),"benchmarks":len(benchmarks)},ensure_ascii=False))
if __name__ == "__main__": main()

#!/usr/bin/env python3
"""Build a sealed task-card registry from the frozen benchmark templates.

This only copies preregistered template/configuration fields and hashes the
canonical template. It never reads outcomes or invents labels from rollouts.
The generated registry must be reviewed and frozen before claiming results.
"""
from __future__ import annotations
import argparse, hashlib, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def _parse_tasks(path: Path):
    tasks = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line.startswith("- {"):
            continue
        fields = {}
        for key, quoted, bare in re.findall(r'([A-Za-z_]+):\s*(?:"([^"]*)"|([^,}]+))', line[3:]):
            fields[key] = (quoted or bare).strip()
        if "id" in fields:
            tasks.append(fields)
    return tasks

def _card(task, split: str, source: str):
    template = {"id": int(task["id"]), "type": task["type"],
                "instruction": task["instruction"], "goal": task["goal"],
                "group": task["group"], "difficulty": task["difficulty"]}
    canonical = json.dumps(template, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {
        "task_id": f"cact_p3_{int(task['id']):02d}",
        "split": split,
        "task_group": task["group"],
        "exact_template_hash": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "world_generator_version": "CACTTaskEnv-v0:DefaultWorldGenerator-v1",
        "initial_inventory": [],
        "world_constraints": {"biome": "forest", "world_seed": "declared", "snapshot_hash": "optional", "snapshot_backend": "xenon_procedural"},
        "goal": task["goal"],
        "success_predicate": f"goal_observed:{task['goal']}",
        "window_trigger": "eligible_retrieval_checkpoint",
        "episode_budget": {"max_minutes": 2, "max_steps": 2400},
        "allowed_randomization_state": {
            "world_seed": "declared_before_rollout",
            "assignment_seed": "declared_before_rollout",
            "outcome_adaptive_sampling": False,
        },
        "source_benchmark": source,
        "source_task_index": int(task["id"]),
        "template": template,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default=str(ROOT / "src/optimus1/conf/benchmark/cact_p3.yaml"))
    ap.add_argument("--out", default=str(ROOT / "protocol_inputs/task_cards.json"))
    ap.add_argument("--select-indices", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--audit-indices", default="8,9,10,11,12,13,14,15")
    args = ap.parse_args()
    tasks = {int(t["id"]): t for t in _parse_tasks(Path(args.benchmark))}
    select = [int(x) for x in args.select_indices.split(",") if x.strip()]
    audit = [int(x) for x in args.audit_indices.split(",") if x.strip()]
    if set(select) & set(audit):
        raise SystemExit("D_select and D_audit task indices must be disjoint")
    missing = sorted((set(select) | set(audit)) - set(tasks))
    if missing:
        raise SystemExit(f"task indices missing from benchmark: {missing}")
    cards = [_card(tasks[i], "D_select", Path(args.benchmark).name) for i in select]
    cards += [_card(tasks[i], "D_audit", Path(args.benchmark).name) for i in audit]
    payload = {"schema_version": "cact.task_cards.v1", "sealed": True,
               "selection_indices": select, "audit_indices": audit,
               "cards": cards}
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out), "cards": len(cards), "sealed": True}))

if __name__ == "__main__":
    main()

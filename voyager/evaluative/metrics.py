from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable


def summarize_result(case_name: str, result: dict) -> dict:
    metrics = result.get("metrics", {})
    goal_graph = result.get("goal_graph", {})
    decision_times = metrics.get("controller_decision_ms", [])
    max_decision_ms = max(decision_times) if decision_times else 0
    avg_decision_ms = (
        sum(decision_times) / len(decision_times) if decision_times else 0
    )
    return {
        "case": case_name,
        "task": result.get("task"),
        "success": result.get("success"),
        "goal_graph_source": goal_graph.get("source"),
        "goal_graph_generation_ms": goal_graph.get("generation_duration_ms"),
        "controller_decisions": metrics.get("controller_decisions", 0),
        "controller_avg_decision_ms": round(avg_decision_ms, 3),
        "controller_max_decision_ms": round(max_decision_ms, 3),
        "controller_total_decision_ms": metrics.get("controller_decision_total_ms", 0),
        "llm_calls_in_control_loop": metrics.get("llm_calls_in_control_loop", 0),
        "total_duration_s": metrics.get("total_duration_s"),
        "final_inventory": json.dumps(result.get("inventory", {}), ensure_ascii=False),
        "snapshot_path": goal_graph.get("snapshot_path"),
        "error": goal_graph.get("error"),
    }


def write_experiment_outputs(
    *,
    output_dir: str | Path,
    payload: dict,
    rows: Iterable[dict],
) -> tuple[Path, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    json_path = output_path / "latest.json"
    csv_path = output_path / "latest.csv"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    rows = list(rows)
    fieldnames = list(rows[0].keys()) if rows else []
    with csv_path.open("w", newline="", encoding="utf-8-sig") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)

    return json_path, csv_path

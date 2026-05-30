from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List

from .metrics import summarize_result, write_experiment_outputs
from .runner import EvaluativeRunner


@dataclass(frozen=True)
class ExperimentCase:
    name: str
    task: str
    initial_inventory: dict[str, int] = field(default_factory=dict)
    max_steps: int = 3
    use_llm_goal_graph: bool = False


class EvaluativeExperiment:
    def __init__(
        self,
        cases: List[ExperimentCase],
        mc_port: int = 25565,
        output_dir: str | Path = "ckpt_evaluative/experiments",
    ):
        self.cases = cases
        self.mc_port = mc_port
        self.output_dir = Path(output_dir)

    def run(self) -> dict[str, Any]:
        started_at = time.strftime("%Y-%m-%dT%H:%M:%S")
        results = []
        rows = []

        for case in self.cases:
            print(f"\n======== Evaluative experiment: {case.name} ========")
            runner = EvaluativeRunner(
                mc_port=self.mc_port,
                use_llm_goal_graph=case.use_llm_goal_graph,
            )
            try:
                result = runner.run_task(
                    case.task,
                    max_steps=case.max_steps,
                    initial_inventory=case.initial_inventory,
                )
            finally:
                runner.close()
            results.append({"case": case.name, "result": result})
            rows.append(summarize_result(case.name, result))

        payload = {
            "started_at": started_at,
            "cases": [case.__dict__ for case in self.cases],
            "summary": rows,
            "results": results,
        }
        json_path, csv_path = write_experiment_outputs(
            output_dir=self.output_dir,
            payload=payload,
            rows=rows,
        )
        print(f"\nExperiment JSON saved to {json_path}")
        print(f"Experiment CSV saved to {csv_path}")
        return payload

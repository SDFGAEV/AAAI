from __future__ import annotations

import os

from voyager.evaluative.experiment import EvaluativeExperiment, ExperimentCase


def main() -> None:
    use_llm_dag = os.environ.get("VOYAGER_USE_LLM_DAG") == "1"
    cases = [
        ExperimentCase(
            name="mine_sand_fallback_dag",
            task="Mine 1 sand",
            max_steps=2,
        ),
        ExperimentCase(
            name="craft_table_fallback_dag",
            task="Craft 1 crafting_table",
            initial_inventory={"oak_log": 1},
            max_steps=3,
        ),
        ExperimentCase(
            name="craft_table_llm_dag",
            task="Craft 1 crafting_table",
            initial_inventory={"oak_log": 1},
            max_steps=3,
            use_llm_goal_graph=use_llm_dag,
        ),
    ]
    EvaluativeExperiment(cases).run()


if __name__ == "__main__":
    main()

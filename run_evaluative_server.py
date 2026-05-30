import os

from voyager.evaluative import EvaluativeRunner


runner = EvaluativeRunner(
    mc_port=25565,
    use_llm_goal_graph=os.environ.get("VOYAGER_USE_LLM_DAG") == "1",
)

try:
    runner.run_task(
        "Craft 1 crafting_table",
        max_steps=3,
        initial_inventory={"oak_log": 1},
    )
finally:
    runner.close()

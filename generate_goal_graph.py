from __future__ import annotations

import argparse
import json
import os

from voyager.evaluative.goal_graph import GoalGraphManager
from voyager.evaluative.llm_worker import GoalGraphLLMWorker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and validate an evaluative-interface goal DAG."
    )
    parser.add_argument(
        "task",
        nargs="?",
        default="Craft 1 crafting_table",
        help="Task text to convert into a goal DAG.",
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        default=os.environ.get("VOYAGER_USE_LLM_DAG") == "1",
        help="Use the configured OpenAI-compatible API before falling back.",
    )
    parser.add_argument(
        "--ckpt-dir",
        default="ckpt_evaluative",
        help="Directory for generated goal graph snapshots.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    goal_manager = GoalGraphManager()
    worker = GoalGraphLLMWorker(
        goal_manager,
        snapshot_dir=f"{args.ckpt_dir}/goal_graphs",
    )

    if args.use_llm:
        graph, source, error = worker.generate_or_fallback_with_source(args.task)
    else:
        graph, source, error = goal_manager.parse_task(args.task), "fallback", None

    snapshot_path = worker.save_snapshot(
        task=args.task,
        graph=graph,
        source=source,
        error=error,
    )
    payload = {
        "task": args.task,
        "source": source,
        "validated": True,
        "snapshot_path": str(snapshot_path),
        "graph": goal_manager.to_dict(graph),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

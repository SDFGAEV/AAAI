from __future__ import annotations

import json
import time
from typing import List, Tuple

from voyager.control_primitives import load_control_primitives
from voyager.env import VoyagerEnv

from .action_renderer import ActionRenderer
from .controller import EvaluativeController
from .async_worker import AsyncLLMWorker
from .config import EvaluativeConfig
from .feedback import FeedbackTracker
from .goal_graph import GoalGraphManager
from .llm_worker import EvaluativeLLMWorker
from .schemas import Decision, GoalGraph, Observation, StructuredAction
from .shared_state import SharedState
from .snapshot_store import SnapshotStore


class EvaluativeRunner:
    """Minimal evaluative-interface runner on top of VoyagerEnv."""

    def __init__(
        self,
        mc_port: int,
        server_port: int = 3000,
        env_wait_ticks: int = 20,
        env_request_timeout: int = 600,
        use_llm_goal_graph: bool = False,
        ckpt_dir: str = "ckpt_evaluative",
        use_async_worker: bool = True,
        worker_timeout: float = 30.0,
    ):
        self.config = EvaluativeConfig(
            ckpt_dir=ckpt_dir,
            env_wait_ticks=env_wait_ticks,
            worker_timeout=worker_timeout,
        )
        self.env = VoyagerEnv(
            mc_port=mc_port,
            server_port=server_port,
            request_timeout=env_request_timeout,
        )
        self.env_wait_ticks = env_wait_ticks
        self.goal_manager = GoalGraphManager()
        self.shared_state = SharedState()
        self.snapshot_store = SnapshotStore(
            goal_manager=self.goal_manager,
            goal_graph_dir=self.config.goal_graph_dir,
            shared_snapshot_dir=self.config.shared_snapshot_dir,
        )
        self.llm_worker = EvaluativeLLMWorker(
            self.goal_manager,
            snapshot_store=self.snapshot_store,
        )
        self.use_llm_goal_graph = use_llm_goal_graph
        self.use_async_worker = use_async_worker
        self.worker_timeout = worker_timeout
        self.async_worker = AsyncLLMWorker(
            shared_state=self.shared_state,
            goal_manager=self.goal_manager,
            llm_worker=self.llm_worker,
            snapshot_store=self.snapshot_store,
            value_matrix=self.config.value_matrix,
            constraints=self.config.constraints,
        )
        self.controller = EvaluativeController(
            shared_state=self.shared_state,
            goal_manager=self.goal_manager,
        )
        self.action_renderer = ActionRenderer()
        self.feedback = FeedbackTracker()
        self.last_goal_snapshot: dict = {}
        self.programs = "\n\n".join(
            load_control_primitives(
                [
                    "mineBlock",
                    "exploreUntil",
                    "craftItem",
                    "placeItem",
                    "smeltItem",
                    "killMob",
                ]
            )
        )

    def close(self) -> None:
        if self.use_async_worker:
            self.async_worker.stop()
        self.env.close()

    def run_task(
        self,
        task: str,
        max_steps: int = 3,
        reset_env: bool = True,
        initial_inventory: dict | None = None,
    ) -> dict:
        total_start = time.perf_counter()
        graph_start = time.perf_counter()
        graph = self._build_goal_graph(task)
        graph_duration_s = time.perf_counter() - graph_start
        print("\n**** Goal graph snapshot ****")
        print(json.dumps(self.goal_manager.to_dict(graph), indent=2, ensure_ascii=False))
        events = self._reset_or_observe(reset_env, initial_inventory or {})
        trace = []
        completed_tasks: set[str] = set()

        for step_idx in range(max_steps):
            observation = self._last_observation(events)
            self._attach_completed_tasks(observation, completed_tasks)
            completed_tasks.update(
                self.goal_manager.completed_task_ids(graph, observation)
            )
            self._attach_completed_tasks(observation, completed_tasks)
            if self.goal_manager.is_complete(graph, observation):
                return self._result(
                    True,
                    graph,
                    observation,
                    trace,
                    total_start=total_start,
                    graph_duration_s=graph_duration_s,
                )

            decision_start = time.perf_counter()
            decision = self.controller.decide(observation)
            decision_duration_s = time.perf_counter() - decision_start
            self.feedback.record_decision(
                step=step_idx,
                decision=decision,
                observation=observation,
            )
            code = self.action_renderer.render(decision.action)
            print(self._format_decision(step_idx, graph, observation, decision, code))

            events = self.env.step(code, programs=self.programs)
            observation = self._last_observation(events)
            self._attach_completed_tasks(observation, completed_tasks)
            event_types = [event_type for event_type, _ in events]
            self.feedback.record_execution(
                step=step_idx,
                observation=observation,
                event_types=event_types,
            )
            trace.append(
                {
                    "step": step_idx,
                    "decision": {
                        "action": decision.action.id,
                        "score": decision.score,
                        "reason": decision.reason,
                        "duration_ms": round(decision_duration_s * 1000, 3),
                        "value_breakdown": decision.value_breakdown,
                        "constraint_rejections": decision.constraint_rejections,
                    },
                    "inventory": observation.get("inventory", {}),
                    "events": event_types,
                }
            )
            completed_tasks.update(
                self.goal_manager.completed_task_ids(graph, observation)
            )
            self._attach_completed_tasks(observation, completed_tasks)

            if self.goal_manager.is_complete(graph, observation):
                return self._result(
                    True,
                    graph,
                    observation,
                    trace,
                    total_start=total_start,
                    graph_duration_s=graph_duration_s,
                )

        return self._result(
            False,
            graph,
            self._last_observation(events),
            trace,
            total_start=total_start,
            graph_duration_s=graph_duration_s,
        )

    def _build_goal_graph(self, task: str) -> GoalGraph:
        if self.use_async_worker:
            self.async_worker.start()
            self.async_worker.request_goal_graph(
                task,
                use_llm=self.use_llm_goal_graph,
            )
            if not self.async_worker.wait_until_ready(timeout=self.worker_timeout):
                raise TimeoutError(
                    f"Async LLM Worker did not produce a goal graph within "
                    f"{self.worker_timeout} seconds."
                )
            snapshot = self.shared_state.get_snapshot()
            if snapshot.goal_graph is None:
                raise RuntimeError("Async LLM Worker finished without a goal graph.")
            self.last_goal_snapshot = {
                "path": self.async_worker.last_metadata.get("snapshot_path"),
                "shared_path": self.async_worker.last_metadata.get("shared_snapshot_path"),
                "source": self.async_worker.last_metadata.get("source"),
                "error": self.async_worker.last_metadata.get("error"),
            }
            print(
                "Async LLM Worker updated SharedState "
                f"v{snapshot.version} from {snapshot.source}"
            )
            print(f"Goal graph snapshot saved to {self.last_goal_snapshot['path']}")
            return snapshot.goal_graph

        if self.use_llm_goal_graph:
            graph, source, error = self.llm_worker.generate_or_fallback_with_source(
                task
            )
        else:
            graph, source, error = self.goal_manager.parse_task(task), "fallback", None
        snapshot_path = self.llm_worker.save_snapshot(
            task=task,
            graph=graph,
            source=source,
            error=error,
        )
        self.last_goal_snapshot = {
            "path": str(snapshot_path),
            "source": source,
            "error": error,
        }
        shared_snapshot = self.shared_state.update(
            goal_graph=graph,
            value_matrix=self.config.value_matrix.to_dict(),
            constraints=self.config.constraints.to_dict(),
            source=source,
        )
        shared_snapshot_path = self.snapshot_store.save_shared_snapshot(
            shared_snapshot
        )
        self.last_goal_snapshot["shared_path"] = str(shared_snapshot_path)
        print(f"Goal graph snapshot saved to {snapshot_path}")
        return graph

    def _reset_or_observe(
        self,
        reset_env: bool,
        initial_inventory: dict,
    ) -> List[Tuple[str, dict]]:
        if reset_env:
            events = self.env.reset(
                options={
                    "mode": "soft",
                    "wait_ticks": self.env_wait_ticks,
                }
            )
            give_lines = []
            for item_name, count in initial_inventory.items():
                give_lines.append(f"bot.chat('/give @s minecraft:{item_name} {int(count)}');")
            return self.env.step(
                "bot.chat('/gamemode survival @s');\n"
                "bot.chat('/clear @s');\n"
                + "\n".join(give_lines)
                + "\n"
                "await bot.waitForTicks(bot.waitTicks);"
            )
        return self.env.step("")

    def _last_observation(self, events: List[Tuple[str, dict]]) -> Observation:
        if not events or events[-1][0] != "observe":
            raise RuntimeError("Expected the last event to be an observe event.")
        return events[-1][1]

    def _attach_completed_tasks(
        self,
        observation: Observation,
        completed_tasks: set[str],
    ) -> None:
        completed_tasks.update(
            task_id
            for task_id in observation.get("_completed_tasks", [])
        )
        observation["_completed_tasks"] = sorted(completed_tasks)

    def _format_decision(
        self,
        step_idx: int,
        graph: GoalGraph,
        observation: Observation,
        decision: Decision,
        code: str,
    ) -> str:
        return (
            "\n**** Evaluative Controller decision ****\n"
            f"step: {step_idx}\n"
            f"task: {graph.task}\n"
            f"actionable: {self.goal_manager.explain_actionable(graph, observation)}\n"
            f"selected: {decision.action.id}\n"
            f"score: {decision.score}\n"
            f"value: {decision.value_breakdown}\n"
            f"rejections: {decision.constraint_rejections}\n"
            f"reason: {decision.reason}\n"
            f"inventory: {observation.get('inventory', {})}\n"
            f"nearby blocks: {', '.join(observation.get('voxels', []))}\n"
            f"code:\n{code}\n"
        )

    def _result(
        self,
        success: bool,
        graph: GoalGraph,
        observation: Observation,
        trace: list,
        total_start: float,
        graph_duration_s: float,
    ) -> dict:
        decision_durations = [
            step["decision"]["duration_ms"]
            for step in trace
            if "decision" in step and "duration_ms" in step["decision"]
        ]
        result = {
            "task": graph.task,
            "success": success,
            "inventory": observation.get("inventory", {}),
            "position": observation.get("status", {}).get("position"),
            "goal_graph": {
                "source": self.last_goal_snapshot.get("source"),
                "snapshot_path": self.last_goal_snapshot.get("path"),
                "shared_snapshot_path": self.last_goal_snapshot.get("shared_path"),
                "error": self.last_goal_snapshot.get("error"),
                "generation_duration_ms": round(graph_duration_s * 1000, 3),
            },
            "metrics": {
                "total_duration_s": round(time.perf_counter() - total_start, 3),
                "controller_decisions": len(trace),
                "controller_decision_ms": decision_durations,
                "controller_decision_total_ms": round(sum(decision_durations), 3),
                "llm_calls_in_control_loop": 0,
            },
            "feedback": self.feedback.summarize(),
            "trace": trace,
        }
        print("\n**** Evaluative result ****")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return result

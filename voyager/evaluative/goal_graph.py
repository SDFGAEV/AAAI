from __future__ import annotations

import re
from typing import List

from .schemas import ActionType, GoalGraph, GoalNode, Observation, StructuredAction


class GoalGraphManager:
    """Structured task goals and local goal-progress checks."""

    ACTION_TYPES = {"mine", "craft", "kill", "noop"}

    LOG_ITEMS = [
        "oak_log",
        "birch_log",
        "spruce_log",
        "jungle_log",
        "acacia_log",
        "dark_oak_log",
        "mangrove_log",
    ]

    def parse_task(self, task: str) -> GoalGraph:
        normalized = task.strip().lower()
        mine_match = re.fullmatch(r"mine\s+(\d+)\s+([a-z_ ]+)", normalized)
        if mine_match:
            count = int(mine_match.group(1))
            target = mine_match.group(2).strip().replace(" ", "_")
            task_id = f"mine_{count}_{target}"
            return GoalGraph(
                task=task,
                root=task_id,
                nodes={
                    task_id: GoalNode(
                        task_id=task_id,
                        action_type="mine",
                        target_item=target,
                        target_count=count,
                        action_count=count,
                        needs=[],
                    )
                },
            )

        craft_match = re.fullmatch(r"craft\s+(\d+)\s+([a-z_ ]+)", normalized)
        if craft_match:
            count = int(craft_match.group(1))
            target = craft_match.group(2).strip().replace(" ", "_")
            if count == 1 and target == "crafting_table":
                return self._crafting_table_graph(task)
        raise ValueError(f"Unsupported minimal evaluative task: {task!r}")

    def _crafting_table_graph(self, task: str) -> GoalGraph:
        mine_log = "mine_1_oak_log"
        craft_planks = "craft_4_oak_planks"
        craft_table = "craft_1_crafting_table"
        return GoalGraph(
            task=task,
            root=craft_table,
            nodes={
                mine_log: GoalNode(
                    task_id=mine_log,
                    action_type="mine",
                    target_item="oak_log",
                    target_count=1,
                    action_count=1,
                    needs=[],
                ),
                craft_planks: GoalNode(
                    task_id=craft_planks,
                    action_type="craft",
                    target_item="oak_planks",
                    target_count=4,
                    action_count=1,
                    needs=[mine_log],
                ),
                craft_table: GoalNode(
                    task_id=craft_table,
                    action_type="craft",
                    target_item="crafting_table",
                    target_count=1,
                    action_count=1,
                    needs=[craft_planks],
                ),
            },
        )

    def from_dict(self, task: str, data: dict) -> GoalGraph:
        if not isinstance(data, dict):
            raise ValueError("Goal graph must be a JSON object.")
        root = data.get("root")
        nodes_data = data.get("nodes")
        if not isinstance(root, str) or not root:
            raise ValueError("Goal graph root must be a non-empty string.")
        if not isinstance(nodes_data, dict) or not nodes_data:
            raise ValueError("Goal graph nodes must be a non-empty object.")

        nodes = {}
        for task_id, node_data in nodes_data.items():
            if not isinstance(task_id, str) or not task_id:
                raise ValueError("Each goal node key must be a non-empty string.")
            if not isinstance(node_data, dict):
                raise ValueError(f"Goal node {task_id} must be an object.")
            action_type = node_data.get("action_type")
            if action_type not in self.ACTION_TYPES:
                raise ValueError(f"Goal node {task_id} has invalid action_type.")
            target_item = node_data.get("target_item")
            if not isinstance(target_item, str) or not target_item:
                raise ValueError(f"Goal node {task_id} target_item must be a string.")
            needs = node_data.get("needs", [])
            if not isinstance(needs, list) or not all(
                isinstance(need, str) for need in needs
            ):
                raise ValueError(f"Goal node {task_id} needs must be a string list.")

            nodes[task_id] = GoalNode(
                task_id=task_id,
                action_type=action_type,
                target_item=target_item,
                target_count=self._positive_int(
                    node_data.get("target_count", 1),
                    f"{task_id}.target_count",
                ),
                action_count=self._positive_int(
                    node_data.get("action_count", 1),
                    f"{task_id}.action_count",
                ),
                needs=needs,
            )

        if root not in nodes:
            raise ValueError(f"Goal graph root {root!r} is not in nodes.")
        for task_id, node in nodes.items():
            for need in node.needs:
                if need not in nodes:
                    raise ValueError(
                        f"Goal node {task_id} depends on unknown node {need}."
                    )
        return GoalGraph(task=task, root=root, nodes=nodes)

    def to_dict(self, graph: GoalGraph) -> dict:
        return {
            "root": graph.root,
            "nodes": {
                task_id: {
                    "action_type": node.action_type,
                    "target_item": node.target_item,
                    "target_count": node.target_count,
                    "action_count": node.action_count,
                    "needs": list(node.needs),
                }
                for task_id, node in graph.nodes.items()
            },
        }

    def _positive_int(self, value, field_name: str) -> int:
        if not isinstance(value, int) or value < 1:
            raise ValueError(f"{field_name} must be a positive integer.")
        return value

    def completed_task_ids(
        self, graph: GoalGraph, observation: Observation
    ) -> List[str]:
        inventory = observation.get("inventory", {})
        completed = list(observation.get("_completed_tasks", []))
        for task_id, node in graph.nodes.items():
            if task_id not in completed and inventory.get(node.target_item, 0) >= node.target_count:
                completed.append(task_id)
        return completed

    def is_complete(self, graph: GoalGraph, observation: Observation) -> bool:
        return graph.root in self.completed_task_ids(graph, observation)

    def actionable_nodes(
        self, graph: GoalGraph, observation: Observation
    ) -> List[GoalNode]:
        completed = set(self.completed_task_ids(graph, observation))
        actionable = []
        for node in graph.nodes.values():
            if node.task_id in completed:
                continue
            if all(need in completed for need in node.needs):
                actionable.append(node)
        return actionable

    def advances_goal(
        self,
        action: StructuredAction,
        graph: GoalGraph,
        observation: Observation,
    ) -> GoalNode | None:
        for node in self.actionable_nodes(graph, observation):
            if action.type != node.action_type:
                continue
            if action.target != node.target_item:
                continue
            if action.count < node.action_count:
                continue
            return node
        return None

    def explain_actionable(self, graph: GoalGraph, observation: Observation) -> str:
        nodes = self.actionable_nodes(graph, observation)
        if not nodes:
            return "No actionable goal nodes."
        return ", ".join(
            f"{node.task_id}({node.action_type} {node.target_item} x{node.target_count})"
            for node in nodes
        )

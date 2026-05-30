from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from .goal_graph import GoalGraphManager
from .schemas import GoalGraph, Observation, StructuredAction


@dataclass(frozen=True)
class ConstraintDecision:
    action: StructuredAction
    allowed: bool
    reasons: List[str]


class ConstraintEngine:
    """Minimal local constraint evaluator."""

    LOG_TO_PLANKS = {
        "oak_log": "oak_planks",
        "birch_log": "birch_planks",
        "spruce_log": "spruce_planks",
        "jungle_log": "jungle_planks",
        "acacia_log": "acacia_planks",
        "dark_oak_log": "dark_oak_planks",
        "mangrove_log": "mangrove_planks",
    }

    def __init__(self, goal_manager: GoalGraphManager | None = None):
        self.goal_manager = goal_manager or GoalGraphManager()

    def filter_actions(
        self,
        actions: Iterable[StructuredAction],
        graph: GoalGraph,
        observation: Observation,
    ) -> tuple[List[StructuredAction], List[str]]:
        allowed = []
        rejections = []
        for action in actions:
            decision = self.evaluate(action, graph, observation)
            if decision.allowed:
                allowed.append(action)
            else:
                rejections.append(f"{action.id}: {'; '.join(decision.reasons)}")
        return allowed, rejections

    def evaluate(
        self,
        action: StructuredAction,
        graph: GoalGraph,
        observation: Observation,
    ) -> ConstraintDecision:
        reasons = []
        inventory = observation.get("inventory", {})
        nearby_blocks = set(observation.get("voxels", []))

        if self.goal_manager.is_complete(graph, observation):
            reasons.append("goal is already complete")

        if action.type == "noop":
            non_noop_exists = self._has_non_noop_candidate(graph, observation)
            if non_noop_exists:
                reasons.append("noop is only allowed when no productive action exists")

        if action.type == "mine":
            if not action.target or action.target == "air":
                reasons.append("cannot mine air or an empty target")
            if action.target not in nearby_blocks:
                reasons.append(f"{action.target} is not visible nearby")

        if action.type == "craft":
            material_reason = self._craft_material_reason(action, inventory)
            if material_reason:
                reasons.append(material_reason)

        return ConstraintDecision(action=action, allowed=not reasons, reasons=reasons)

    def _has_non_noop_candidate(
        self,
        graph: GoalGraph,
        observation: Observation,
    ) -> bool:
        actionable = self.goal_manager.actionable_nodes(graph, observation)
        return bool(actionable)

    def _craft_material_reason(
        self,
        action: StructuredAction,
        inventory: dict,
    ) -> str:
        if action.target.endswith("_planks"):
            required_log = None
            for log_name, plank_name in self.LOG_TO_PLANKS.items():
                if plank_name == action.target:
                    required_log = log_name
                    break
            if required_log and inventory.get(required_log, 0) < 1:
                return f"crafting {action.target} requires {required_log}"
        if action.target == "crafting_table":
            has_planks = any(
                inventory.get(plank_name, 0) >= 4
                for plank_name in self.LOG_TO_PLANKS.values()
            )
            if not has_planks:
                return "crafting crafting_table requires 4 planks"
        return ""

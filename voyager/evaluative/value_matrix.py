from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from .goal_graph import GoalGraphManager
from .schemas import GoalGraph, Observation, StructuredAction


@dataclass(frozen=True)
class ValueScore:
    total: float
    breakdown: Dict[str, float]
    reason: str


class ValueMatrix:
    """Minimal local V matrix with task/safety/exploration dimensions."""

    def __init__(self, goal_manager: GoalGraphManager | None = None):
        self.goal_manager = goal_manager or GoalGraphManager()

    def score(
        self,
        action: StructuredAction,
        graph: GoalGraph,
        observation: Observation,
    ) -> ValueScore:
        matched_goal = self.goal_manager.advances_goal(action, graph, observation)
        task_score = 100.0 if matched_goal else 0.0
        safety_score = 0.0 if action.type != "noop" else -5.0
        exploration_score = self._exploration_score(action, observation)
        total = task_score + safety_score + exploration_score

        if matched_goal:
            reason = f"directly advances actionable goal {matched_goal.task_id}"
        elif action.type == "noop":
            reason = "fallback action with low value"
        else:
            reason = "does not directly advance current goal"

        return ValueScore(
            total=total,
            breakdown={
                "task": task_score,
                "safety": safety_score,
                "exploration": exploration_score,
            },
            reason=reason,
        )

    def _exploration_score(
        self,
        action: StructuredAction,
        observation: Observation,
    ) -> float:
        known_inventory = observation.get("inventory", {})
        if action.type == "mine" and action.target not in known_inventory:
            return 1.0
        return 0.0

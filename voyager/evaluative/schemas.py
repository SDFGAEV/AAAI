from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional


ActionType = Literal["mine", "craft", "kill", "noop"]


@dataclass(frozen=True)
class StructuredAction:
    type: ActionType
    target: str
    count: int = 1
    reason: str = ""

    @property
    def id(self) -> str:
        return f"{self.type}:{self.target}:{self.count}"


@dataclass(frozen=True)
class GoalNode:
    task_id: str
    action_type: ActionType
    target_item: str
    target_count: int = 1
    action_count: int = 1
    needs: List[str] = field(default_factory=list)


@dataclass
class GoalGraph:
    task: str
    nodes: Dict[str, GoalNode]
    root: str


@dataclass
class Decision:
    action: StructuredAction
    score: float
    matched_goal: Optional[GoalNode]
    reason: str
    value_breakdown: Dict[str, float] = field(default_factory=dict)
    constraint_rejections: List[str] = field(default_factory=list)


Observation = Dict[str, Any]

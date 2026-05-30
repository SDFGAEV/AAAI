from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .schemas import Decision, Observation


@dataclass
class FeedbackTracker:
    """Records decision/execution evidence for later diagnosis."""

    events: list[dict[str, Any]] = field(default_factory=list)

    def record_decision(
        self,
        *,
        step: int,
        decision: Decision,
        observation: Observation,
    ) -> None:
        self.events.append(
            {
                "type": "decision",
                "step": step,
                "action": decision.action.id,
                "score": decision.score,
                "value_breakdown": decision.value_breakdown,
                "constraint_rejections": decision.constraint_rejections,
                "inventory": observation.get("inventory", {}),
                "nearby_blocks": observation.get("voxels", []),
            }
        )

    def record_execution(
        self,
        *,
        step: int,
        observation: Observation,
        event_types: list[str],
    ) -> None:
        self.events.append(
            {
                "type": "execution",
                "step": step,
                "inventory": observation.get("inventory", {}),
                "event_types": event_types,
            }
        )

    def record_failure(self, *, reason: str, observation: Observation | None = None) -> None:
        self.events.append(
            {
                "type": "failure",
                "reason": reason,
                "inventory": observation.get("inventory", {}) if observation else {},
            }
        )

    def summarize(self) -> dict[str, Any]:
        return {
            "event_count": len(self.events),
            "decision_count": sum(1 for event in self.events if event["type"] == "decision"),
            "execution_count": sum(1 for event in self.events if event["type"] == "execution"),
            "failure_count": sum(1 for event in self.events if event["type"] == "failure"),
            "events": self.events,
        }

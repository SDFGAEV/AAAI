from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from threading import RLock
from typing import Any

from .schemas import GoalGraph


@dataclass
class SharedSnapshot:
    goal_graph: GoalGraph | None = None
    value_matrix: dict[str, Any] = field(default_factory=dict)
    constraints: dict[str, Any] = field(default_factory=dict)
    version: int = 0
    updated_at: str = ""
    source: str = "local"


class SharedState:
    """Thread-safe V/C/G snapshot store.

    The current implementation is used synchronously, but the API is designed
    for a future background LLM Worker that atomically swaps validated snapshots.
    """

    def __init__(self):
        self._lock = RLock()
        self._snapshot = SharedSnapshot()

    def update(
        self,
        *,
        goal_graph: GoalGraph | None = None,
        value_matrix: dict[str, Any] | None = None,
        constraints: dict[str, Any] | None = None,
        source: str = "local",
    ) -> SharedSnapshot:
        with self._lock:
            current = self._snapshot
            self._snapshot = SharedSnapshot(
                goal_graph=goal_graph if goal_graph is not None else current.goal_graph,
                value_matrix=value_matrix
                if value_matrix is not None
                else dict(current.value_matrix),
                constraints=constraints
                if constraints is not None
                else dict(current.constraints),
                version=current.version + 1,
                updated_at=datetime.now().isoformat(timespec="milliseconds"),
                source=source,
            )
            return self._snapshot

    def get_snapshot(self) -> SharedSnapshot:
        with self._lock:
            current = self._snapshot
            return SharedSnapshot(
                goal_graph=current.goal_graph,
                value_matrix=dict(current.value_matrix),
                constraints=dict(current.constraints),
                version=current.version,
                updated_at=current.updated_at,
                source=current.source,
            )

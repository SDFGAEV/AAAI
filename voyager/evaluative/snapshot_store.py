from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .goal_graph import GoalGraphManager
from .schemas import GoalGraph
from .shared_state import SharedSnapshot


class SnapshotStore:
    """Persists validated architecture snapshots."""

    def __init__(
        self,
        *,
        goal_manager: GoalGraphManager | None = None,
        goal_graph_dir: str | Path = "ckpt_evaluative/goal_graphs",
        shared_snapshot_dir: str | Path = "ckpt_evaluative/shared_snapshots",
    ):
        self.goal_manager = goal_manager or GoalGraphManager()
        self.goal_graph_dir = Path(goal_graph_dir)
        self.shared_snapshot_dir = Path(shared_snapshot_dir)

    def save_goal_graph(
        self,
        *,
        task: str,
        graph: GoalGraph,
        source: str,
        model: str | None = None,
        base_url: str | None = None,
        error: str | None = None,
    ) -> Path:
        payload = {
            "task": task,
            "source": source,
            "validated": True,
            "model": model if source == "llm" else None,
            "base_url": base_url if source == "llm" else None,
            "error": error,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "graph": self.goal_manager.to_dict(graph),
        }
        return self._write_snapshot(self.goal_graph_dir, payload)

    def save_shared_snapshot(self, snapshot: SharedSnapshot) -> Path:
        payload: dict[str, Any] = {
            "version": snapshot.version,
            "updated_at": snapshot.updated_at,
            "source": snapshot.source,
            "value_matrix": snapshot.value_matrix,
            "constraints": snapshot.constraints,
            "goal_graph": self.goal_manager.to_dict(snapshot.goal_graph)
            if snapshot.goal_graph
            else None,
        }
        return self._write_snapshot(self.shared_snapshot_dir, payload)

    def _write_snapshot(self, directory: Path, payload: dict) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        snapshot_path = directory / f"{timestamp}.json"
        latest_path = directory / "latest.json"
        text = json.dumps(payload, indent=2, ensure_ascii=False)
        snapshot_path.write_text(text, encoding="utf-8")
        latest_path.write_text(text, encoding="utf-8")
        return snapshot_path

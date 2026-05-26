import json
import os
import threading
from dataclasses import dataclass, field, asdict


@dataclass
class DAGNode:
    id: str
    action_type: str    # "mine" | "craft" | "move" | "attack" | "collect" | "place" | "smelt"
    target: str         # MC item/block name, e.g. "oak_log", "stone_pickaxe"
    depends_on: list = field(default_factory=list)   # list of node ids
    completed: bool = False


class TaskDAG:
    """
    G: task goal decomposed as a dependency graph.

    Generated once by LLM worker; read every frame by Controller for local
    goal-advancement checks — no further LLM calls needed.

    The key invariant: any semantic reasoning (what steps does this task require?)
    happens at DAG-generation time. Subsequent goal checks are pure local lookups.
    """

    def __init__(self, goal: str, nodes: list[DAGNode]):
        self.goal = goal
        self._lock = threading.RLock()
        self._nodes: dict[str, DAGNode] = {n.id: n for n in nodes}

    # ── read (Controller side) ────────────────────────────────────────────────

    def executable_leaves(self) -> list[DAGNode]:
        """Nodes ready to execute: not yet completed, all dependencies done."""
        with self._lock:
            done = {nid for nid, n in self._nodes.items() if n.completed}
            return [
                n for n in self._nodes.values()
                if not n.completed and all(dep in done for dep in n.depends_on)
            ]

    def matches_goal(self, action_type: str, target: str) -> bool:
        """
        True if (action_type, target) matches any executable leaf node.
        This is the local goal-bonus check — no LLM involved.
        """
        for node in self.executable_leaves():
            if node.action_type == action_type and node.target == target:
                return True
        return False

    def is_complete(self) -> bool:
        with self._lock:
            return all(n.completed for n in self._nodes.values())

    # ── write ─────────────────────────────────────────────────────────────────

    def mark_completed(self, node_id: str):
        with self._lock:
            if node_id in self._nodes:
                self._nodes[node_id].completed = True

    def mark_completed_by_action(self, action_type: str, target: str):
        """Mark the first matching executable leaf as completed."""
        with self._lock:
            for node in self.executable_leaves():
                if node.action_type == action_type and node.target == target:
                    node.completed = True
                    return

    # ── persistence ───────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        with self._lock:
            return {
                "goal": self.goal,
                "nodes": [asdict(n) for n in self._nodes.values()],
            }

    @classmethod
    def from_dict(cls, data: dict) -> "TaskDAG":
        nodes = [DAGNode(**n) for n in data.get("nodes", [])]
        return cls(goal=data.get("goal", ""), nodes=nodes)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "TaskDAG":
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def __repr__(self):
        leaves = [f"{n.action_type}:{n.target}" for n in self.executable_leaves()]
        return f"TaskDAG(goal={self.goal!r}, executable={leaves})"

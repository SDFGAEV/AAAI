from __future__ import annotations

import json
from pathlib import Path

from .action_space import ActionSpace
from .config import ThreatConfig
from .constraint_engine import ConstraintEngine
from .goal_graph import GoalGraphManager
from .schemas import Decision, Observation
from .shared_state import SharedState
from .value_matrix import ValueMatrix


class EvaluativeController:
    """High-frequency local controller.

    This class must not call an LLM. It only reads structured goal snapshots and
    the current observation, then chooses a structured action.
    """

    def __init__(
        self,
        shared_state: SharedState,
        goal_manager: GoalGraphManager | None = None,
        action_space: ActionSpace | None = None,
        constraint_engine: ConstraintEngine | None = None,
        value_matrix: ValueMatrix | None = None,
        config_path: Path | None = None,
    ):
        self.shared_state = shared_state
        self.goal_manager = goal_manager or GoalGraphManager()
        self.action_space = action_space or ActionSpace()

        if config_path is None:
            config_path = Path(__file__).parent / "ckpt" / "controller_config.json"
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        threat_config = ThreatConfig.from_dict(config.get("threat_detection", {}))

        if constraint_engine is not None:
            self.constraint_engine = constraint_engine
        else:
            self.constraint_engine = ConstraintEngine(self.goal_manager, threat_config=threat_config)

        if value_matrix is not None:
            self.value_matrix = value_matrix
        else:
            self.value_matrix = ValueMatrix(self.goal_manager, threat_config=threat_config)

    def decide(self, observation: Observation) -> Decision:
        snapshot = self.shared_state.get_snapshot()
        graph = snapshot.goal_graph
        if graph is None:
            raise RuntimeError("SharedState has no goal_graph snapshot.")

        candidates = self.action_space.candidates(observation)
        allowed, rejections = self.constraint_engine.filter_actions(
            candidates,
            graph,
            observation,
        )
        if not allowed:
            allowed = [action for action in candidates if action.type == "noop"]
            rejections.append("all productive actions rejected; falling back to noop")

        best: Decision | None = None
        for action in allowed:
            value = self.value_matrix.score(action, graph, observation)
            matched_goal = self.goal_manager.advances_goal(action, graph, observation)
            decision = Decision(
                action=action,
                score=value.total,
                matched_goal=matched_goal,
                reason=value.reason,
                value_breakdown=value.breakdown,
                constraint_rejections=rejections,
            )
            if best is None or decision.score > best.score:
                best = decision
        assert best is not None
        return best

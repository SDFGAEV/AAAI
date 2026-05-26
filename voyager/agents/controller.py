from dataclasses import dataclass, field

from .value_matrix import ValueMatrix
from .constraint_engine import ConstraintEngine
from .entity_registry import EntityRegistry
from .task_dag import TaskDAG

# Additive bonus when an action directly advances the current DAG leaf.
# Separating it from raw V scores makes the goal-tracking contribution visible.
GOAL_BONUS = 0.5


@dataclass
class Action:
    action_type: str    # "mine" | "craft" | "move" | "attack" | "collect" | "place" | "smelt" | "idle"
    target: str         # MC item/block/mob name; "" for move/idle
    raw_score: float = 0.0      # from ValueMatrix
    goal_bonus: float = 0.0     # from TaskDAG local check
    final_score: float = 0.0    # raw_score + goal_bonus

    def __str__(self):
        return f"{self.action_type}:{self.target}  (raw={self.raw_score:.3f}  goal={self.goal_bonus:.1f}  final={self.final_score:.3f})"


class Controller:
    """
    Real-time action selector. Runs every frame; never calls LLM.

    Decision pipeline (all local):
      1. Update EntityRegistry from latest events
      2. Generate candidate actions from DAG leaves, voxels, threats
      3. Filter hard-blocked candidates via ConstraintEngine
      4. Score remaining candidates: ValueMatrix score + goal bonus
      5. Return the highest-scoring Action

    The LLM only touches this class indirectly — by writing V, C, G snapshots
    that this class reads. Once the snapshots are in place, the LLM is not
    on the critical path.
    """

    def __init__(
        self,
        value_matrix: ValueMatrix,
        constraint_engine: ConstraintEngine,
        entity_registry: EntityRegistry,
        dag: TaskDAG | None = None,
    ):
        self.value_matrix = value_matrix
        self.constraint_engine = constraint_engine
        self.entity_registry = entity_registry
        self.dag = dag

    # ── public API ────────────────────────────────────────────────────────────

    def set_dag(self, dag: TaskDAG):
        self.dag = dag

    def select_action(self, events: list) -> Action:
        """
        Given a Voyager event list, return the best local action.
        This is the only method called per-frame; it never touches the LLM.
        """
        self.entity_registry.update_from_observation(events)
        state = self._build_state(events)
        candidates = self._generate_candidates(state)
        allowed = self._filter(candidates, state)
        if not allowed:
            return Action("idle", "", final_score=0.0)
        scored = self._score(allowed)
        best = max(scored, key=lambda a: a.final_score)
        self._log(scored, best)
        return best

    def mark_action_done(self, action: Action):
        """
        Call after the bot successfully executes an action so the DAG advances.
        """
        if self.dag:
            self.dag.mark_completed_by_action(action.action_type, action.target)

    # ── pipeline steps ────────────────────────────────────────────────────────

    def _build_state(self, events: list) -> dict:
        """Flat state dict used by ConstraintEngine and candidate generator."""
        state: dict = {
            "health": 20.0,
            "threat_nearby": False,
            "voxels": [],
            "threats": [],
        }
        for event_type, event in reversed(events):
            if event_type != "observe":
                continue
            status = event.get("status", {})
            state["health"] = float(status.get("health", 20.0))
            state["voxels"] = list(event.get("voxels", []))
            break
        state["threats"] = self.entity_registry.nearby_threats()
        state["threat_nearby"] = bool(state["threats"])
        return state

    def _generate_candidates(self, state: dict) -> list[Action]:
        seen: set[tuple] = set()
        candidates: list[Action] = []

        def add(action_type: str, target: str):
            key = (action_type, target)
            if key not in seen:
                seen.add(key)
                candidates.append(Action(action_type, target))

        voxel_set = set(state["voxels"])
        threat_names = {t.name for t in state["threats"]}

        # DAG leaves: only generate actions that are currently feasible.
        # - mine/collect: target must be in voxels (otherwise bot can't reach it)
        # - attack: target must be a nearby entity
        # - move/craft/place/smelt/idle: always feasible, add unconditionally
        if self.dag:
            for node in self.dag.executable_leaves():
                if node.action_type in ("mine", "collect"):
                    if node.target in voxel_set:
                        add(node.action_type, node.target)
                elif node.action_type == "attack":
                    if node.target in threat_names:
                        add(node.action_type, node.target)
                else:
                    add(node.action_type, node.target)

        # Mine anything available in the voxel list
        for block in state["voxels"]:
            add("mine", block)

        # Attack any nearby threats
        for threat in state["threats"]:
            add("attack", threat.name)

        # Fallbacks always available
        add("move", "")
        add("idle", "")

        return candidates

    def _filter(self, candidates: list[Action], state: dict) -> list[Action]:
        """Remove candidates blocked by hard constraints."""
        all_types = list({c.action_type for c in candidates})
        allowed_types = set(self.constraint_engine.filter(all_types, state))
        return [c for c in candidates if c.action_type in allowed_types]

    def _score(self, candidates: list[Action]) -> list[Action]:
        for action in candidates:
            action.raw_score = self.value_matrix.score(action.action_type)
            if self.dag and self.dag.matches_goal(action.action_type, action.target):
                action.goal_bonus = GOAL_BONUS
            else:
                action.goal_bonus = 0.0
            action.final_score = action.raw_score + action.goal_bonus
        return candidates

    # ── logging ───────────────────────────────────────────────────────────────

    def _log(self, scored: list[Action], best: Action):
        print("\033[34m[Controller] Candidate actions:\033[0m")
        for a in sorted(scored, key=lambda x: x.final_score, reverse=True):
            marker = " ← selected" if a is best else ""
            print(f"\033[34m  {a}{marker}\033[0m")

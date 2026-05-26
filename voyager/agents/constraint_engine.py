import json
import os
import threading
from dataclasses import dataclass, field, asdict


@dataclass
class Constraint:
    id: str
    type: str           # "hard" | "soft"
    description: str
    forbidden_actions: list = field(default_factory=list)
    # Condition keys understood by applies():
    #   "always": True           – always active
    #   "health_below": float    – active when bot health < threshold
    #   "threat_nearby": bool    – active when hostile mob is within default radius
    condition: dict = field(default_factory=lambda: {"always": True})

    def applies(self, state: dict) -> bool:
        """Return True if this constraint is active given current state."""
        for key, threshold in self.condition.items():
            if key == "always":
                continue
            elif key == "health_below":
                if state.get("health", 20.0) >= float(threshold):
                    return False
            elif key == "threat_nearby":
                has_threat = bool(state.get("threat_nearby", False))
                if has_threat != bool(threshold):
                    return False
        return True


_DEFAULT_CONSTRAINTS = [
    Constraint(
        id="no_attack_players",
        type="hard",
        description="Never attack other players.",
        forbidden_actions=["attack_player"],
        condition={"always": True},
    ),
    Constraint(
        id="retreat_low_health",
        type="soft",
        description="Avoid combat when health is critically low.",
        forbidden_actions=["attack"],
        condition={"health_below": 5.0},
    ),
]


class ConstraintEngine:
    """
    C: set of hard/soft constraints on action selection.

    Hard constraints are enforced by filtering candidate actions before scoring.
    Soft constraints are surfaced to the Controller for optional score penalties.

    Written asynchronously by LLM worker; queried synchronously by Controller.
    All public methods are thread-safe.
    """

    def __init__(self, constraints: list | None = None):
        self._lock = threading.RLock()
        self._constraints: dict[str, Constraint] = {}
        for c in (constraints or _DEFAULT_CONSTRAINTS):
            if isinstance(c, dict):
                c = Constraint(**c)
            self._constraints[c.id] = c

    # ── read (Controller side) ────────────────────────────────────────────────

    def filter(self, candidate_actions: list[str], state: dict) -> list[str]:
        """Remove hard-constrained actions given current state."""
        with self._lock:
            blocked: set[str] = set()
            for c in self._constraints.values():
                if c.type == "hard" and c.applies(state):
                    blocked.update(c.forbidden_actions)
            return [a for a in candidate_actions if a not in blocked]

    def active_soft(self, state: dict) -> list[Constraint]:
        """Return soft constraints that apply to current state."""
        with self._lock:
            return [
                c for c in self._constraints.values()
                if c.type == "soft" and c.applies(state)
            ]

    def is_blocked(self, action: str, state: dict) -> bool:
        """True if action is hard-blocked in current state."""
        with self._lock:
            for c in self._constraints.values():
                if c.type == "hard" and action in c.forbidden_actions and c.applies(state):
                    return True
            return False

    # ── write (LLM worker side) ───────────────────────────────────────────────

    def add(self, constraint: Constraint):
        with self._lock:
            self._constraints[constraint.id] = constraint

    def remove(self, constraint_id: str):
        with self._lock:
            self._constraints.pop(constraint_id, None)

    def replace_all(self, constraints: list[Constraint]):
        """Bulk replace all constraints (called when LLM returns new C snapshot)."""
        with self._lock:
            self._constraints = {c.id: c for c in constraints}

    # ── persistence ───────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        with self._lock:
            return {"constraints": [asdict(c) for c in self._constraints.values()]}

    @classmethod
    def from_dict(cls, data: dict) -> "ConstraintEngine":
        return cls(constraints=[Constraint(**c) for c in data.get("constraints", [])])

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ConstraintEngine":
        with open(path) as f:
            return cls.from_dict(json.load(f))

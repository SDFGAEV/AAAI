import json
import os
import threading

DIMENSIONS = ("efficiency", "safety", "task_progress")

_DEFAULT_SCORES = {
    "mine":    {"efficiency": 0.7, "safety": 0.6, "task_progress": 0.8},
    "craft":   {"efficiency": 0.5, "safety": 0.9, "task_progress": 0.9},
    "attack":  {"efficiency": 0.4, "safety": 0.2, "task_progress": 0.3},
    "move":    {"efficiency": 0.6, "safety": 0.7, "task_progress": 0.2},
    "idle":    {"efficiency": 0.1, "safety": 0.9, "task_progress": 0.0},
    "collect": {"efficiency": 0.8, "safety": 0.7, "task_progress": 0.7},
    "place":   {"efficiency": 0.5, "safety": 0.7, "task_progress": 0.6},
    "smelt":   {"efficiency": 0.4, "safety": 0.8, "task_progress": 0.7},
}

_DEFAULT_WEIGHTS = {"efficiency": 0.3, "safety": 0.4, "task_progress": 0.3}


class ValueMatrix:
    """
    V: action_type -> per-dimension scores, aggregated via weighted sum.

    Written asynchronously by LLM worker; read synchronously by Controller.
    All public methods are thread-safe.
    """

    def __init__(self, scores: dict | None = None, weights: dict | None = None):
        self._lock = threading.RLock()
        self._scores: dict[str, dict[str, float]] = {
            k: dict(v) for k, v in (scores or _DEFAULT_SCORES).items()
        }
        self._weights: dict[str, float] = dict(weights or _DEFAULT_WEIGHTS)

    # ── read (Controller side) ────────────────────────────────────────────────

    def score(self, action_type: str) -> float:
        """Weighted aggregate score for one action type. Returns 0.0 if unknown."""
        with self._lock:
            dims = self._scores.get(action_type)
            if dims is None:
                return 0.0
            return sum(self._weights.get(d, 0.0) * dims.get(d, 0.0) for d in DIMENSIONS)

    def scores_for(self, action_type: str) -> dict[str, float]:
        """Per-dimension scores for one action type."""
        with self._lock:
            return dict(self._scores.get(action_type, {}))

    def rank(self, action_types: list[str]) -> list[str]:
        """Return action_types sorted by descending aggregate score."""
        with self._lock:
            return sorted(action_types, key=self.score, reverse=True)

    # ── write (LLM worker side) ───────────────────────────────────────────────

    def update(self, action_type: str, dim_scores: dict[str, float]):
        """Replace dimension scores for one action type."""
        with self._lock:
            self._scores[action_type] = {
                d: float(dim_scores.get(d, 0.0)) for d in DIMENSIONS
            }

    def set_weights(self, weights: dict[str, float]):
        with self._lock:
            self._weights = {
                d: float(weights.get(d, self._weights.get(d, 0.0)))
                for d in DIMENSIONS
            }

    def replace_all(self, scores: dict, weights: dict | None = None):
        """Bulk replace entire matrix (called when LLM returns a new V snapshot)."""
        with self._lock:
            self._scores = {k: dict(v) for k, v in scores.items()}
            if weights is not None:
                self._weights = dict(weights)

    # ── persistence ───────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        with self._lock:
            return {"scores": dict(self._scores), "weights": dict(self._weights)}

    @classmethod
    def from_dict(cls, data: dict) -> "ValueMatrix":
        return cls(scores=data.get("scores"), weights=data.get("weights"))

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ValueMatrix":
        with open(path) as f:
            return cls.from_dict(json.load(f))

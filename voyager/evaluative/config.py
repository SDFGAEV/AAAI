from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ValueMatrixConfig:
    dimensions: list[str] = field(
        default_factory=lambda: ["task", "safety", "exploration"]
    )
    weights: dict[str, float] = field(
        default_factory=lambda: {"task": 1.0, "safety": 1.0, "exploration": 1.0}
    )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ConstraintSet:
    hard: list[str] = field(
        default_factory=lambda: [
            "no_noop_when_productive_action_exists",
            "mine_only_visible_blocks",
            "craft_requires_materials",
            "no_action_after_goal_complete",
        ]
    )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ThreatConfig:
    hostile_entities: set[str] = field(default_factory=set)
    dangerous_blocks: set[str] = field(default_factory=set)
    entity_scan_radius: int = 16

    @classmethod
    def from_dict(cls, data: dict) -> "ThreatConfig":
        return cls(
            hostile_entities=set(data.get("hostile_entities", [])),
            dangerous_blocks=set(data.get("dangerous_blocks", [])),
            entity_scan_radius=data.get("entity_scan_radius", 16),
        )

    def to_dict(self) -> dict:
        return {
            "hostile_entities": list(self.hostile_entities),
            "dangerous_blocks": list(self.dangerous_blocks),
            "entity_scan_radius": self.entity_scan_radius,
        }


@dataclass(frozen=True)
class EvaluativeConfig:
    ckpt_dir: str = "ckpt_evaluative"
    env_wait_ticks: int = 20
    worker_timeout: float = 30.0
    value_matrix: ValueMatrixConfig = field(default_factory=ValueMatrixConfig)
    constraints: ConstraintSet = field(default_factory=ConstraintSet)

    @property
    def goal_graph_dir(self) -> Path:
        return Path(self.ckpt_dir) / "goal_graphs"

    @property
    def shared_snapshot_dir(self) -> Path:
        return Path(self.ckpt_dir) / "shared_snapshots"

    @property
    def trace_dir(self) -> Path:
        return Path(self.ckpt_dir) / "traces"

    @property
    def experiment_dir(self) -> Path:
        return Path(self.ckpt_dir) / "experiments"

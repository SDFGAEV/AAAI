import json
import os
import threading
import time
from dataclasses import dataclass, field, asdict

# Mobs that are hostile toward the player by default in vanilla MC 1.21.
HOSTILE_MOBS: frozenset[str] = frozenset({
    "zombie", "skeleton", "creeper", "spider", "cave_spider",
    "enderman", "witch", "blaze", "ghast", "phantom", "drowned",
    "husk", "stray", "pillager", "vindicator", "ravager", "vex",
    "slime", "magma_cube", "guardian", "elder_guardian",
    "wither", "ender_dragon", "hoglin", "piglin_brute", "zoglin", "warden",
    "bogged", "breeze",  # MC 1.21 new mobs
})


@dataclass
class EntityEntry:
    name: str
    distance: float
    hostile: bool
    last_seen: float = field(default_factory=time.time)


class EntityRegistry:
    """
    Tracks entities and nearby blocks observed from mineflayer.

    The mineflayer status observation provides:
      - status.entities: {entity_name: closest_distance}  (mobs within 32 blocks)
      - voxels: [block_type, ...]                         (blocks within ~8 blocks)

    Updated each observe cycle by Controller; queried locally for threat
    assessment and resource availability without any LLM call.
    All public methods are thread-safe.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._entities: dict[str, EntityEntry] = {}
        self._voxels: list[str] = []

    # ── write (called each observe cycle) ────────────────────────────────────

    def update_from_observation(self, events: list):
        """
        Refresh registry from a Voyager event list.
        Only 'observe' events carry entity/voxel data; others are skipped.
        """
        now = time.time()
        with self._lock:
            for event_type, event in events:
                if event_type != "observe":
                    continue
                status = event.get("status", {})
                raw = status.get("entities", {})   # {name: distance}
                self._entities = {
                    name: EntityEntry(
                        name=name,
                        distance=float(dist),
                        hostile=name in HOSTILE_MOBS,
                        last_seen=now,
                    )
                    for name, dist in raw.items()
                }
                self._voxels = list(event.get("voxels", []))

    # ── read (Controller side) ────────────────────────────────────────────────

    def nearby_threats(self, radius: float = 16.0) -> list[EntityEntry]:
        """Hostile mobs within `radius` blocks."""
        with self._lock:
            return [e for e in self._entities.values() if e.hostile and e.distance <= radius]

    def nearest_threat(self) -> EntityEntry | None:
        threats = self.nearby_threats()
        return min(threats, key=lambda e: e.distance) if threats else None

    def has_threat(self, radius: float = 16.0) -> bool:
        return bool(self.nearby_threats(radius))

    def has_block(self, block_name: str) -> bool:
        """True if block_name appears in the nearby voxel list."""
        with self._lock:
            return block_name in self._voxels

    def nearby_voxels(self) -> list[str]:
        with self._lock:
            return list(self._voxels)

    def all_entities(self) -> list[EntityEntry]:
        with self._lock:
            return list(self._entities.values())

    # ── persistence ───────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        with self._lock:
            return {
                "entities": {k: asdict(v) for k, v in self._entities.items()},
                "voxels": list(self._voxels),
            }

    @classmethod
    def from_dict(cls, data: dict) -> "EntityRegistry":
        reg = cls()
        for name, e in data.get("entities", {}).items():
            reg._entities[name] = EntityEntry(**e)
        reg._voxels = data.get("voxels", [])
        return reg

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "EntityRegistry":
        with open(path) as f:
            return cls.from_dict(json.load(f))

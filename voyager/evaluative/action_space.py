from __future__ import annotations

from typing import List

from .schemas import Observation, StructuredAction


class ActionSpace:
    """Builds local candidate actions from the latest observation."""

    LOG_TO_PLANKS = {
        "oak_log": "oak_planks",
        "birch_log": "birch_planks",
        "spruce_log": "spruce_planks",
        "jungle_log": "jungle_planks",
        "acacia_log": "acacia_planks",
        "dark_oak_log": "dark_oak_planks",
        "mangrove_log": "mangrove_planks",
    }

    def candidates(self, observation: Observation) -> List[StructuredAction]:
        actions: List[StructuredAction] = []
        nearby_blocks = observation.get("voxels", [])
        inventory = observation.get("inventory", {})

        for block_name in nearby_blocks:
            if block_name in inventory:
                continue
            actions.append(
                StructuredAction(
                    type="mine",
                    target=block_name,
                    count=1,
                    reason=f"{block_name} is visible nearby",
                )
            )

        for log_name, plank_name in self.LOG_TO_PLANKS.items():
            if inventory.get(log_name, 0) >= 1:
                actions.append(
                    StructuredAction(
                        type="craft",
                        target=plank_name,
                        count=1,
                        reason=f"{log_name} in inventory can be crafted into planks",
                    )
                )

        for plank_name in self.LOG_TO_PLANKS.values():
            if inventory.get(plank_name, 0) >= 4:
                actions.append(
                    StructuredAction(
                        type="craft",
                        target="crafting_table",
                        count=1,
                        reason=f"{plank_name} x4 can be crafted into a crafting_table",
                    )
                )

        actions.append(
            StructuredAction(
                type="noop",
                target="wait",
                count=1,
                reason="fallback when no useful local action is available",
            )
        )
        return actions

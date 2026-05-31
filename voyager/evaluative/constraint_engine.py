from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from .schemas import Observation, StructuredAction


@dataclass
class ConstraintResult:
    allowed: bool
    level: str
    reasons: List[str] = field(default_factory=list)
    penalty: float = 0.0
    required_action: Optional[str] = None


class ConstraintEngine:
    def __init__(
        self,
        templates_path: Optional[str | Path] = None,
        config_path: Optional[str | Path] = None,
    ):
        if templates_path is None:
            templates_path = Path(__file__).parent / "ckpt" / "constraint_templates.json"
        if config_path is None:
            config_path = Path(__file__).parent / "ckpt" / "controller_config.json"

        with open(templates_path, "r", encoding="utf-8") as f:
            templates_data = json.load(f)

        self.templates = templates_data["templates"]
        self.levels = templates_data["levels"]

        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        self.threat_config = config.get("threat_detection", {})
        self.hostile_entities: Set[str] = set(self.threat_config.get("hostile_entities", []))
        self.dangerous_blocks: Set[str] = set(self.threat_config.get("dangerous_blocks", []))
        self.entity_scan_radius = config.get("observation", {}).get("entity_scan_radius", 16)

    def evaluate(
        self,
        action: StructuredAction,
        observation: Observation,
    ) -> ConstraintResult:
        all_reasons = []
        max_level_priority = -1
        required_action = None
        total_penalty = 0.0

        for template in self.templates:
            triggered, reason = self._check_condition(template, action, observation)
            if triggered:
                all_reasons.append(f"[{template['id']}] {reason}")

                level_priority = self.levels[template["level"]]["priority"]
                if level_priority > max_level_priority:
                    max_level_priority = level_priority

                if template["level"] == "L0":
                    if template.get("required_action"):
                        required_action = template["required_action"]
                elif template["level"] in ("L1", "L2"):
                    total_penalty += template.get("penalty", 0)

        if max_level_priority == 0:
            return ConstraintResult(
                allowed=False,
                level="L0",
                reasons=all_reasons,
                penalty=0.0,
                required_action=required_action,
            )

        return ConstraintResult(
            allowed=True,
            level="L1" if max_level_priority == 1 else "L2",
            reasons=all_reasons,
            penalty=total_penalty,
            required_action=required_action,
        )

    def _check_condition(
        self,
        template: Dict[str, Any],
        action: StructuredAction,
        observation: Observation,
    ) -> tuple[bool, str]:
        condition = template.get("condition", {})
        condition_type = condition.get("type")

        if condition_type == "health_percent":
            threshold = condition.get("threshold", 0.2)
            health_percent = observation.get("health_percent", 1.0)
            comparison = condition.get("comparison", "less_than")

            if self._compare(health_percent, threshold, comparison):
                return True, template.get("reason", f"Health {health_percent} {comparison} {threshold}")

        elif condition_type == "hunger_percent":
            threshold = condition.get("threshold", 0.3)
            hunger_percent = observation.get("hunger_percent", 1.0)
            comparison = condition.get("comparison", "less_than")

            if self._compare(hunger_percent, threshold, comparison):
                return True, template.get("reason", f"Hunger {hunger_percent} {comparison} {threshold}")

        elif condition_type == "entity_proximity":
            entity_type = condition.get("entity_type")
            distance_threshold = condition.get("distance", 5)
            comparison = condition.get("comparison", "less_than_or_equal")

            nearby_entities = observation.get("nearby_entities", [])
            for entity in nearby_entities:
                entity_name = entity.get("name", "")
                entity_distance = entity.get("distance", 999)

                if entity_name == entity_type and self._compare(entity_distance, distance_threshold, comparison):
                    return True, template.get("reason", f"{entity_type} at distance {entity_distance}")

        elif condition_type == "block_proximity":
            block_type = condition.get("block_type")
            distance_threshold = condition.get("distance", 3)
            comparison = condition.get("comparison", "less_than_or_equal")

            nearby_blocks = observation.get("nearby_blocks", [])
            for block in nearby_blocks:
                if block.get("type") == block_type:
                    distance = block.get("distance", 999)
                    if self._compare(distance, distance_threshold, comparison):
                        return True, template.get("reason", f"{block_type} at distance {distance}")

        elif condition_type == "position_y":
            threshold = condition.get("threshold", 5)
            comparison = condition.get("comparison", "less_than")
            y = observation.get("position", {}).get("y", 100)

            if self._compare(y, threshold, comparison):
                return True, template.get("reason", f"Y position {y} {comparison} {threshold}")

        elif condition_type == "target_empty":
            if observation.get("target_block") in (None, "air", ""):
                return True, template.get("reason", "Target is empty")

        elif condition_type == "tool_required":
            equipped_item = observation.get("equipped_item", "")
            target = action.target or ""

            if "ore" in target and equipped_item not in ("wooden_pickaxe", "stone_pickaxe", "iron_pickaxe", "diamond_pickaxe", "netherite_pickaxe"):
                return True, template.get("reason", "Tool required to mine ore")

        elif condition_type == "inventory_full_percent":
            threshold = condition.get("threshold", 0.9)
            comparison = condition.get("comparison", "greater_than_or_equal")

            inventory = observation.get("inventory", {})
            used_slots = len([item for item, count in inventory.items() if count > 0])
            full_percent = used_slots / 36

            if self._compare(full_percent, threshold, comparison):
                return True, template.get("reason", f"Inventory {full_percent:.0%} full")

        elif condition_type == "time_night_no_weapon":
            time_of_day = observation.get("time_of_day", 0)
            has_weapon = observation.get("has_weapon", False)

            if time_of_day > 12000 and not has_weapon:
                return True, template.get("reason", "Night without weapon")

        elif condition_type == "submerged":
            if observation.get("in_water", False) or observation.get("submerged", False):
                return True, template.get("reason", "Underwater action")

        elif condition_type == "no_crafting_table":
            if "craft" in action.type and not observation.get("crafting_table_nearby", False):
                return True, template.get("reason", "No crafting table nearby")

        elif condition_type == "no_furnace":
            if action.type == "smelt" and not observation.get("furnace_nearby", False):
                return True, template.get("reason", "No furnace nearby")

        elif condition_type == "long_distance":
            threshold = condition.get("threshold", 10)
            comparison = condition.get("comparison", "greater_than")
            distance = observation.get("target_distance", 0)

            if self._compare(distance, threshold, comparison):
                return True, template.get("reason", f"Distance {distance} > {threshold}")

        elif condition_type == "unnecessary_grass_break":
            if action.target == "grass" and not observation.get("has_purpose", True):
                return True, template.get("reason", "Unnecessary grass breaking")

        elif condition_type == "unnecessary_leaves_break":
            if "leaves" in action.target and not observation.get("has_purpose", True):
                return True, template.get("reason", "Unnecessary leaves breaking")

        return False, ""

    def _compare(self, value: float, threshold: float, comparison: str) -> bool:
        if comparison == "less_than":
            return value < threshold
        elif comparison == "less_than_or_equal":
            return value <= threshold
        elif comparison == "greater_than":
            return value > threshold
        elif comparison == "greater_than_or_equal":
            return value >= threshold
        elif comparison == "equal":
            return value == threshold
        return False

    def filter_actions(
        self,
        actions: List[StructuredAction],
        observation: Observation,
    ) -> tuple[List[StructuredAction], Dict[str, ConstraintResult]]:
        allowed = []
        results = {}

        for action in actions:
            result = self.evaluate(action, observation)
            results[action.id] = result

            if result.allowed:
                allowed.append(action)

        return allowed, results

    def filter_actions_with_penalty(
        self,
        actions: List[StructuredAction],
        observation: Observation,
    ) -> tuple[List[tuple[StructuredAction, float]], Dict[str, ConstraintResult]]:
        scored_actions = []
        results = {}

        for action in actions:
            result = self.evaluate(action, observation)
            results[action.id] = result

            if result.allowed:
                scored_actions.append((action, result.penalty))

        scored_actions.sort(key=lambda x: x[1], reverse=True)

        return scored_actions, results

    def get_violations(
        self,
        action: StructuredAction,
        observation: Observation,
    ) -> List[str]:
        violations = []
        for template in self.templates:
            triggered, reason = self._check_condition(template, action, observation)
            if triggered:
                violations.append(f"[{template['id']}] {template['name']}: {reason}")
        return violations

    def get_required_action(
        self,
        observation: Observation,
    ) -> Optional[str]:
        for template in self.templates:
            if template.get("level") != "L0":
                continue

            condition = template.get("condition", {})
            condition_type = condition.get("type")

            triggered = False
            if condition_type == "entity_proximity":
                entity_type = condition.get("entity_type")
                distance_threshold = condition.get("distance", 5)
                nearby_entities = observation.get("nearby_entities", [])

                for entity in nearby_entities:
                    if entity.get("name") == entity_type:
                        distance = entity.get("distance", 999)
                        if distance <= distance_threshold:
                            triggered = True
                            break

            elif condition_type == "block_proximity":
                block_type = condition.get("block_type")
                distance_threshold = condition.get("distance", 3)
                nearby_blocks = observation.get("nearby_blocks", [])

                for block in nearby_blocks:
                    if block.get("type") == block_type:
                        distance = block.get("distance", 999)
                        if distance <= distance_threshold:
                            triggered = True
                            break

            elif condition_type == "health_percent":
                threshold = condition.get("threshold", 0.2)
                health_percent = observation.get("health_percent", 1.0)
                if health_percent < threshold:
                    triggered = True

            if triggered:
                required = template.get("required_action")
                if required:
                    return required

        return None

    def get_all_templates(self) -> List[Dict[str, Any]]:
        return self.templates

    def get_templates_by_level(self, level: str) -> List[Dict[str, Any]]:
        return [t for t in self.templates if t.get("level") == level]

    def get_l0_templates(self) -> List[Dict[str, Any]]:
        return self.get_templates_by_level("L0")

    def get_l1_templates(self) -> List[Dict[str, Any]]:
        return self.get_templates_by_level("L1")

    def get_l2_templates(self) -> List[Dict[str, Any]]:
        return self.get_templates_by_level("L2")


def load_constraint_engine(
    templates_path: Optional[str | Path] = None,
    config_path: Optional[str | Path] = None,
) -> ConstraintEngine:
    return ConstraintEngine(templates_path, config_path)

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from .schemas import Observation, StructuredAction


@dataclass
class ConstraintResult:
    """约束评估结果"""
    allowed: bool  # 是否允许执行
    level: str  # 约束级别 L0/L1/L2
    reasons: List[str] = field(default_factory=list)  # 违规原因
    penalty: float = 0.0  # 惩罚值（L1/L2）
    required_action: Optional[str] = None  # 必需动作（L0）


class ConstraintEngine:
    """约束求值引擎 - 实现 L0/L1/L2 三级约束检查

    约束级别：
    - L0: 硬约束（立即禁止，必须执行 required_action）
    - L1: 软约束（允许但扣分）
    - L2: 建议约束（警告）
    """

    def __init__(
        self,
        goal_manager=None,
        templates_path: Optional[str | Path] = None,
        config_path: Optional[str | Path] = None,
    ):
        self.goal_manager = goal_manager
        if templates_path is None:
            templates_path = Path(__file__).parent / "ckpt" / "constraint_templates.json"
        if config_path is None:
            config_path = Path(__file__).parent / "ckpt" / "controller_config.json"

        # 加载约束模板
        with open(templates_path, "r", encoding="utf-8") as f:
            templates_data = json.load(f)

        self.templates = templates_data["templates"]  # 约束规则列表
        self.levels = templates_data["levels"]  # 级别定义

        # 加载控制器配置
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
        """评估单个动作是否违反约束

        返回 ConstraintResult：
        - allowed: 是否允许执行
        - level: 最高违规级别
        - reasons: 所有违规原因
        - penalty: 总惩罚值
        - required_action: L0 必需动作
        """
        all_reasons = []
        max_level_priority = -1
        required_action = None
        total_penalty = 0.0

        # 遍历所有模板检查约束
        for template in self.templates:
            triggered, reason = self._check_condition(template, action, observation)
            if triggered:
                all_reasons.append(f"[{template['id']}] {reason}")

                level_priority = self.levels[template["level"]]["priority"]
                if level_priority > max_level_priority:
                    max_level_priority = level_priority

                # L0: 记录必需动作
                if template["level"] == "L0":
                    if template.get("required_action"):
                        required_action = template["required_action"]
                # L1/L2: 累加惩罚
                elif template["level"] in ("L1", "L2"):
                    total_penalty += template.get("penalty", 0)

        # L0 硬约束
        if max_level_priority == 0:
            return ConstraintResult(
                allowed=False,
                level="L0",
                reasons=all_reasons,
                penalty=0.0,
                required_action=required_action,
            )

        # L1/L2 软约束
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
        """检查单个约束条件是否触发"""
        condition = template.get("condition", {})
        condition_type = condition.get("type")

        # 健康度检查
        if condition_type == "health_percent":
            threshold = condition.get("threshold", 0.2)
            health_percent = observation.get("health_percent", 1.0)
            comparison = condition.get("comparison", "less_than")

            if self._compare(health_percent, threshold, comparison):
                return True, template.get("reason", f"Health {health_percent} {comparison} {threshold}")

        # 饥饿度检查
        elif condition_type == "hunger_percent":
            threshold = condition.get("threshold", 0.3)
            hunger_percent = observation.get("hunger_percent", 1.0)
            comparison = condition.get("comparison", "less_than")

            if self._compare(hunger_percent, threshold, comparison):
                return True, template.get("reason", f"Hunger {hunger_percent} {comparison} {threshold}")

        # 实体距离检查（如苦力怕在 5 格内）
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

        # 方块距离检查（如岩浆在 3 格内）
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

        # Y 轴位置检查（防止在悬崖底部）
        elif condition_type == "position_y":
            threshold = condition.get("threshold", 5)
            comparison = condition.get("comparison", "less_than")
            y = observation.get("position", {}).get("y", 100)

            if self._compare(y, threshold, comparison):
                return True, template.get("reason", f"Y position {y} {comparison} {threshold}")

        # 目标为空检查
        elif condition_type == "target_empty":
            if observation.get("target_block") in (None, "air", ""):
                return True, template.get("reason", "Target is empty")

        # 工具需求检查
        elif condition_type == "tool_required":
            equipped_item = observation.get("equipped_item", "")
            target = action.target or ""

            if "ore" in target and equipped_item not in ("wooden_pickaxe", "stone_pickaxe", "iron_pickaxe", "diamond_pickaxe", "netherite_pickaxe"):
                return True, template.get("reason", "Tool required to mine ore")

        # 背包空间检查
        elif condition_type == "inventory_full_percent":
            threshold = condition.get("threshold", 0.9)
            comparison = condition.get("comparison", "greater_than_or_equal")

            inventory = observation.get("inventory", {})
            used_slots = len([item for item, count in inventory.items() if count > 0])
            full_percent = used_slots / 36

            if self._compare(full_percent, threshold, comparison):
                return True, template.get("reason", f"Inventory {full_percent:.0%} full")

        # 夜间无武器检查
        elif condition_type == "time_night_no_weapon":
            time_of_day = observation.get("time_of_day", 0)
            has_weapon = observation.get("has_weapon", False)

            if time_of_day > 12000 and not has_weapon:
                return True, template.get("reason", "Night without weapon")

        # 水下动作检查
        elif condition_type == "submerged":
            if observation.get("in_water", False) or observation.get("submerged", False):
                return True, template.get("reason", "Underwater action")

        # 需要工作台检查
        elif condition_type == "no_crafting_table":
            if "craft" in action.type and not observation.get("crafting_table_nearby", False):
                return True, template.get("reason", "No crafting table nearby")

        # 需要熔炉检查
        elif condition_type == "no_furnace":
            if action.type == "smelt" and not observation.get("furnace_nearby", False):
                return True, template.get("reason", "No furnace nearby")

        # 远距离检查
        elif condition_type == "long_distance":
            threshold = condition.get("threshold", 10)
            comparison = condition.get("comparison", "greater_than")
            distance = observation.get("target_distance", 0)

            if self._compare(distance, threshold, comparison):
                return True, template.get("reason", f"Distance {distance} > {threshold}")

        # 不必要的草方块破坏
        elif condition_type == "unnecessary_grass_break":
            if action.target == "grass" and not observation.get("has_purpose", True):
                return True, template.get("reason", "Unnecessary grass breaking")

        # 不必要的树叶破坏
        elif condition_type == "unnecessary_leaves_break":
            if "leaves" in action.target and not observation.get("has_purpose", True):
                return True, template.get("reason", "Unnecessary leaves breaking")

        return False, ""

    def _compare(self, value: float, threshold: float, comparison: str) -> bool:
        """数值比较"""
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
        graph=None,
        observation: Observation = None,
    ) -> tuple[List[StructuredAction], List[str]]:
        """过滤允许的动作列表"""
        # 兼容两种调用方式
        if observation is None:
            observation = graph
            graph = None

        allowed = []
        rejections = []

        for action in actions:
            result = self.evaluate(action, observation)
            if not result.allowed:
                rejections.extend(result.reasons)
            else:
                allowed.append(action)

        return allowed, rejections

    def filter_actions_with_penalty(
        self,
        actions: List[StructuredAction],
        observation: Observation,
    ) -> tuple[List[tuple[StructuredAction, float]], Dict[str, ConstraintResult]]:
        """过滤动作并返回惩罚分数（按惩罚从高到低排序）"""
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
        """获取动作的所有违规信息"""
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
        """获取当前必需的强制动作（如逃离、进食）

        用于 L0 约束触发时指导 Agent 执行相应动作
        """
        for template in self.templates:
            if template.get("level") != "L0":
                continue

            condition = template.get("condition", {})
            condition_type = condition.get("type")

            triggered = False

            # 实体距离检查
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

            # 方块距离检查
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

            # 健康度检查
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
        """获取所有约束模板"""
        return self.templates

    def get_templates_by_level(self, level: str) -> List[Dict[str, Any]]:
        """获取指定级别的所有模板"""
        return [t for t in self.templates if t.get("level") == level]

    def get_l0_templates(self) -> List[Dict[str, Any]]:
        """获取所有 L0 硬约束"""
        return self.get_templates_by_level("L0")

    def get_l1_templates(self) -> List[Dict[str, Any]]:
        """获取所有 L1 软约束"""
        return self.get_templates_by_level("L1")

    def get_l2_templates(self) -> List[Dict[str, Any]]:
        """获取所有 L2 建议约束"""
        return self.get_templates_by_level("L2")


def load_constraint_engine(
    templates_path: Optional[str | Path] = None,
    config_path: Optional[str | Path] = None,
) -> ConstraintEngine:
    """便捷加载函数"""
    return ConstraintEngine(templates_path, config_path)

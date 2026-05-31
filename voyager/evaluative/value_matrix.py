from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from .entity_registry import EntityRegistry, load_entity_registry
from .schemas import Observation, StructuredAction


@dataclass
class ValueScore:
    """动作评分结果"""
    total: float  # 总分
    breakdown: Dict[str, float]  # 各项得分明细
    reason: str  # 评分原因
    dimensions: Dict[str, float] = field(default_factory=dict)  # 三维价值


@dataclass
class FeatureWeights:
    """特征权重 - LLM Worker 输出的特征权重表"""
    weights: Dict[str, Dict[str, float]]  # feature -> dimension -> weight
    confidence: float = 1.0  # 置信度
    timestamp: float = 0.0  # 更新时间戳

    def get(self, feature: str, dimension: str) -> float:
        """获取指定特征在指定维度上的权重"""
        if feature not in self.weights:
            return 0.0
        return self.weights[feature].get(dimension, 0.0)

    def update(self, feature: str, dimension: str, value: float):
        """更新特征权重"""
        if feature not in self.weights:
            self.weights[feature] = {}
        self.weights[feature][dimension] = value

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            "weights": self.weights,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeatureWeights":
        """从字典反序列化"""
        return cls(
            weights=data.get("weights", {}),
            confidence=data.get("confidence", 1.0),
            timestamp=data.get("timestamp", 0.0),
        )


class ValueMatrix:
    """价值矩阵 - 实现特征空间 × 多维 × 继承 × 置信度"""

    def __init__(
        self,
        goal_manager=None,
        entity_registry: Optional[EntityRegistry] = None,
        config_path: Optional[str | Path] = None,
        feature_weights: Optional[FeatureWeights] = None,
    ):
        self.goal_manager = goal_manager
        
        # 加载实体注册表
        self.entity_registry = entity_registry or load_entity_registry()

        # 加载配置文件
        if config_path is None:
            config_path = Path(__file__).parent / "ckpt" / "controller_config.json"
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        # 从配置中读取权重参数
        self.value_weights_config = self.config.get("value_weights", {})
        self.alpha = self.value_weights_config.get("alpha", 0.7)  # 健康度权重
        self.beta = self.value_weights_config.get("beta", 0.3)  # 威胁密度权重
        self.goal_bonus = self.value_weights_config.get("goal_bonus", 2.0)  # 目标奖励
        self.default_weights = self.value_weights_config.get("default_weights", {})

        # 威胁检测配置
        self.threat_config = self.config.get("threat_detection", {})
        self.hostile_entities: Set[str] = set(self.threat_config.get("hostile_entities", []))

        # 初始化特征权重（LLM 更新的部分）
        self._feature_weights = feature_weights or self._init_default_feature_weights()

    def _init_default_feature_weights(self) -> FeatureWeights:
        """初始化默认特征权重"""
        weights = {
            "attack_damage": {"safety": -0.5, "task": 0.0, "exploration": 0.0},
            "move_speed": {"safety": -0.3, "task": 0.0, "exploration": 0.0},
            "can_explode": {"safety": -5.0, "task": 0.0, "exploration": 0.0},
            "drop_value": {"safety": 0.0, "task": 0.5, "exploration": 0.3},
            "health": {"safety": 0.1, "task": 0.0, "exploration": 0.0},
            "minable": {"safety": 0.0, "task": 1.0, "exploration": 0.5},
            "hostile": {"safety": -2.0, "task": 0.0, "exploration": 0.0},
            "tool_level": {"safety": 0.0, "task": 0.3, "exploration": 0.0},
            "reach": {"safety": -0.2, "task": 0.0, "exploration": 0.0},
            "swimming": {"safety": 0.2, "task": 0.0, "exploration": 0.1},
            "flying": {"safety": -0.5, "task": 0.0, "exploration": 0.2},
            "tame": {"safety": 0.5, "task": 0.0, "exploration": 0.0},
            "breeding": {"safety": 0.3, "task": 0.5, "exploration": 0.0},
        }
        return FeatureWeights(weights=weights)

    def compute_dynamic_weights(
        self,
        health_percent: float,
        threat_density: float,
        has_active_goal: bool = True,
    ) -> Dict[str, float]:
        """计算动态权重 w(t)

        根据健康度和威胁密度动态调整三维权重：
        - w_safe(t): 安全权重，随健康度降低和威胁密度增加而增加
        - w_task(t): 任务权重，有活动目标时为主权重
        - w_expl(t): 探索权重，剩余部分

        公式: w_safe(t) = clamp(α*(1-h) + β*d, 0, 1)
        """
        w_safe = self.alpha * (1 - health_percent) + self.beta * threat_density
        w_safe = max(0.0, min(1.0, w_safe))

        if has_active_goal:
            w_task = 1.0 - w_safe
        else:
            w_task = 0.3 * (1.0 - w_safe)

        w_expl = max(0.0, 1.0 - w_safe - w_task)

        # 归一化
        total = w_safe + w_task + w_expl
        if total > 0:
            w_safe /= total
            w_task /= total
            w_expl /= total

        return {
            "safety": w_safe,
            "task": w_task,
            "exploration": w_expl,
        }

    def compute_entity_value(
        self,
        entity_name: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """获取实体的基础价值（查表 + 继承）"""
        default_values = self.entity_registry.get_default_values(entity_name)
        return default_values

    def compute_feature_based_value(
        self,
        entity_name: str,
        dynamic_weights: Dict[str, float],
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """计算基于特征的价值

        K_v(e, c) = Σ w_k · φ_k(e, c)

        将实体的特征值与特征权重相乘并求和
        """
        features = self.entity_registry.get_all_features_for_entity(entity_name, context)

        result = {"safety": 0.0, "task": 0.0, "exploration": 0.0}

        for feature_name, feature_value in features.items():
            if feature_value == 0:
                continue

            for dimension in ("safety", "task", "exploration"):
                weight = self._feature_weights.get(feature_name, dimension)
                result[dimension] += weight * feature_value

        return result

    def score(
        self,
        action: StructuredAction,
        graph=None,
        observation: Observation = None,
        dynamic_weights: Optional[Dict[str, float]] = None,
        goal_progress: float = 0.0,
    ) -> ValueScore:
        """对动作进行综合评分

        评分公式: Score(a) = Σ w_d(t) · v_d(entity(a), c) + goal_bonus

        支持两种调用方式：
        1. 新方式（带动态权重）: score(action, observation, dynamic_weights, goal_progress)
        2. 兼容旧方式: score(action, graph, observation)

        Args:
            action: 待评分动作
            graph: GoalGraph（兼容旧接口）
            observation: 当前观察
            dynamic_weights: 动态权重（可选，默认自动计算）
            goal_progress: 目标进度 0.0-1.0
        """
        # 兼容两种调用方式
        if observation is None:
            observation = graph
            graph = None

        # 获取健康度和威胁密度
        health_percent = observation.get("health_percent", 1.0)
        threat_density = self._compute_threat_density(observation)

        # 计算动态权重（核心功能）
        if dynamic_weights is None:
            dynamic_weights = self.compute_dynamic_weights(
                health_percent, threat_density, has_active_goal=(goal_progress > 0)
            )

        # 获取目标实体
        entity_name = action.target or "unknown"

        # 基础价值（查表 + 继承）
        base_values = self.compute_entity_value(entity_name, observation)

        # 特征价值（特征权重 × 特征值）
        feature_values = self.compute_feature_based_value(
            entity_name, dynamic_weights, observation
        )

        # 合并价值
        for dim in ("safety", "task", "exploration"):
            base_values[dim] += feature_values.get(dim, 0.0)

        # 加权求和
        weighted_score = 0.0
        for dim, weight in dynamic_weights.items():
            weighted_score += weight * base_values.get(dim, 0.0)

        # 加上目标奖励
        if goal_progress > 0:
            weighted_score += self.goal_bonus * goal_progress

        # 构建得分明细
        breakdown = {
            f"base_{dim}": base_values.get(dim, 0.0)
            for dim in ("safety", "task", "exploration")
        }
        for dim in ("safety", "task", "exploration"):
            breakdown[f"weighted_{dim}"] = dynamic_weights.get(dim, 0.0) * base_values.get(dim, 0.0)
        breakdown["goal_bonus"] = self.goal_bonus * goal_progress

        # 生成评分原因
        reason = self._generate_reason(action, entity_name, base_values, goal_progress)

        return ValueScore(
            total=weighted_score,
            breakdown=breakdown,
            reason=reason,
            dimensions=base_values,
        )

    def _compute_threat_density(self, observation: Observation) -> float:
        """计算威胁密度

        在16格范围内的敌对实体数量，归一化到0-1
        """
        nearby_entities = observation.get("nearby_entities", [])
        if not nearby_entities:
            return 0.0

        hostile_count = 0
        for entity in nearby_entities:
            if entity.get("name") in self.hostile_entities:
                distance = entity.get("distance", 999)
                if distance <= 16:
                    hostile_count += 1

        max_threats = 5
        return min(hostile_count / max_threats, 1.0)

    def _generate_reason(
        self,
        action: StructuredAction,
        entity_name: str,
        base_values: Dict[str, float],
        goal_progress: float,
    ) -> str:
        """生成评分原因说明"""
        reasons = []

        if goal_progress > 0:
            reasons.append(f"advances goal ({goal_progress:.0%})")

        if base_values.get("safety", 0) < -5:
            reasons.append("high safety risk")
        elif base_values.get("safety", 0) > 5:
            reasons.append("safety beneficial")

        if base_values.get("task", 0) > 5:
            reasons.append("high task value")

        if base_values.get("exploration", 0) > 3:
            reasons.append("exploration beneficial")

        if not reasons:
            reasons.append("neutral action")

        return "; ".join(reasons)

    def update_feature_weights(self, new_weights: Dict[str, Dict[str, float]]):
        """更新特征权重（供 LLM Worker 调用）"""
        for feature, dim_weights in new_weights.items():
            for dim, value in dim_weights.items():
                self._feature_weights.update(feature, dim, value)

    def get_feature_weights(self) -> FeatureWeights:
        """获取当前特征权重"""
        return self._feature_weights

    def apply_patch(self, patch: Dict[str, Any]):
        """应用 LLM 生成的增量补丁"""
        if "feature_weights" in patch:
            self.update_feature_weights(patch["feature_weights"])

    def compute_action_scores(
        self,
        actions: List[StructuredAction],
        observation: Observation,
        dynamic_weights: Optional[Dict[str, float]] = None,
        goal_progress: float = 0.0,
    ) -> List[ValueScore]:
        """批量计算动作评分"""
        scores = []
        for action in actions:
            score = self.score(
                action,
                observation=observation,
                dynamic_weights=dynamic_weights,
                goal_progress=goal_progress,
            )
            scores.append(score)
        return scores

    def get_best_action(
        self,
        actions: List[StructuredAction],
        observation: Observation,
        dynamic_weights: Optional[Dict[str, float]] = None,
        goal_progress: float = 0.0,
    ) -> tuple[Optional[StructuredAction], Optional[ValueScore]]:
        """获取最高分动作"""
        if not actions:
            return None, None

        best_action = None
        best_score = None

        for action in actions:
            score = self.score(
                action,
                observation=observation,
                dynamic_weights=dynamic_weights,
                goal_progress=goal_progress,
            )
            if best_score is None or score.total > best_score.total:
                best_action = action
                best_score = score

        return best_action, best_score


def load_value_matrix(
    entity_registry: Optional[EntityRegistry] = None,
    config_path: Optional[str | Path] = None,
) -> ValueMatrix:
    """便捷加载函数"""
    return ValueMatrix(entity_registry, config_path)

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


class EntityRegistry:
    """实体注册表 - 管理所有实体的特征和价值信息

    支持：
    - 实体的直接特征和默认价值
    - 类别继承（hostile_mob -> entity）
    - 上下文覆盖（从当前观察中获取特征值）
    """

    def __init__(self, json_path: str | Path):
        self.json_path = Path(json_path)
        with open(self.json_path, "r", encoding="utf-8") as f:
            self.data: Dict[str, Any] = json.load(f)

        self.features: List[str] = self.data["features"]  # 所有特征列表
        self.entities: Dict[str, Any] = self.data["entities"]  # 实体数据
        self.hierarchy: Dict[str, Dict[str, float]] = self.data["hierarchy"]  # 类别层级价值
        self.category_parents: Dict[str, str] = self.data.get("category_parents", {})  # 类别父节点
        self.features_description: Dict[str, str] = self.data.get("features_description", {})  # 特征描述

    def get_entity_features(self, entity_name: str) -> Dict[str, float]:
        """获取实体的直接特征值"""
        if entity_name not in self.entities:
            return {}
        return self.entities[entity_name].get("features", {})

    def get_entity_defaults(self, entity_name: str) -> Dict[str, float]:
        """获取实体的直接默认价值"""
        if entity_name not in self.entities:
            return {}
        return self.entities[entity_name].get("defaults", {})

    def get_entity_category(self, entity_name: str) -> Optional[str]:
        """获取实体所属类别"""
        if entity_name not in self.entities:
            return None
        return self.entities[entity_name].get("category")

    def get_feature_value(
        self,
        entity_name: str,
        feature_name: str,
        context: Optional[Dict[str, Any]] = None
    ) -> float:
        """获取实体在指定特征上的值

        优先级：
        1. 实体的直接特征值
        2. 上下文中的特征值（如当前观察）
        3. 0（未定义）
        """
        entity_features = self.get_entity_features(entity_name)
        if feature_name in entity_features:
            return float(entity_features[feature_name])

        # 上下文覆盖
        if context and feature_name in context:
            return float(context[feature_name])

        return 0.0

    def lookup_value(
        self,
        entity_name: str,
        dimension: str,
        default: float = 0.0
    ) -> float:
        """查询实体在指定维度上的价值（带继承）

        优先级：
        1. 实体的直接默认价值
        2. 所属类别的继承价值
        3. 父类别的继承价值
        4. 默认值
        """
        if entity_name not in self.entities:
            # 未知实体，尝试通过类别回退
            category = self._get_entity_category_fallback(entity_name)
            if category:
                return self._lookup_category_value(category, dimension, default)
            return default

        # 直接价值
        entity_defaults = self.entities[entity_name].get("defaults", {})
        if dimension in entity_defaults:
            return float(entity_defaults[dimension])

        # 类别继承
        category = self.get_entity_category(entity_name)
        if category:
            return self._lookup_category_value(category, dimension, default)

        return default

    def _lookup_category_value(
        self,
        category: str,
        dimension: str,
        default: float = 0.0
    ) -> float:
        """递归查询类别层级的价值"""
        # 直接在 hierarchy 中查找
        if category in self.hierarchy and dimension in self.hierarchy[category]:
            return float(self.hierarchy[category][dimension])

        # 递归查找父类别
        if category in self.category_parents:
            parent = self.category_parents[category]
            return self._lookup_category_value(parent, dimension, default)

        # 回退到基础 entity 类别
        if category in self.hierarchy and dimension in self.hierarchy["entity"]:
            return float(self.hierarchy["entity"][dimension])

        return default

    def _get_entity_category_fallback(self, entity_name: str) -> Optional[str]:
        """未知实体的类别回退（默认返回 None）"""
        return None

    def get_default_values(
        self,
        entity_name: str
    ) -> Dict[str, float]:
        """获取实体的三维默认价值（安全/任务/探索）

        通过继承链解析：
        entity -> category -> specific_entity
        """
        result = {
            "safety": 0.0,
            "task": 0.0,
            "exploration": 0.0
        }

        if entity_name not in self.entities:
            # 未知实体，尝试类别继承
            category = self._get_entity_category_fallback(entity_name)
            if category:
                for dim in result:
                    result[dim] = self._lookup_category_value(category, dim, result[dim])
            return result

        # 直接价值
        entity_defaults = self.entities[entity_name].get("defaults", {})
        for dim in result:
            if dim in entity_defaults:
                result[dim] = float(entity_defaults[dim])
            else:
                # 类别继承
                category = self.get_entity_category(entity_name)
                if category:
                    result[dim] = self._lookup_category_value(category, dim, result[dim])

        return result

    def get_all_features_for_entity(
        self,
        entity_name: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """获取实体的所有特征值（用于特征权重计算）"""
        result = {}
        for feature in self.features:
            result[feature] = self.get_feature_value(entity_name, feature, context)
        return result

    def get_category_hierarchy(self, category: str) -> List[str]:
        """获取类别的继承链

        例如: hostile_mob -> entity
        """
        hierarchy = [category]
        current = category
        while current in self.category_parents:
            parent = self.category_parents[current]
            hierarchy.append(parent)
            current = parent
        hierarchy.append("entity")
        return hierarchy

    def entity_exists(self, entity_name: str) -> bool:
        """检查实体是否存在"""
        return entity_name in self.entities

    def get_all_entities(self) -> List[str]:
        """获取所有已知实体"""
        return list(self.entities.keys())

    def get_entities_by_category(self, category: str) -> List[str]:
        """获取指定类别的所有实体"""
        result = []
        for entity_name, entity_data in self.entities.items():
            if entity_data.get("category") == category:
                result.append(entity_name)
        return result

    def compute_value_vector(
        self,
        entity_name: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """计算实体的价值向量（别名方法）"""
        return self.get_default_values(entity_name)

    def get_all_categories(self) -> List[str]:
        """获取所有类别"""
        return list(set(self.category_parents.keys()) | set(self.hierarchy.keys()))


def load_entity_registry(
    json_path: Optional[str | Path] = None
) -> EntityRegistry:
    """便捷加载函数"""
    if json_path is None:
        current_dir = Path(__file__).parent
        json_path = current_dir / "ckpt" / "entity_registry.json"
    return EntityRegistry(json_path)

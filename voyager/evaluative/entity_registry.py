from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


class EntityRegistry:
    def __init__(self, json_path: str | Path):
        self.json_path = Path(json_path)
        with open(self.json_path, "r", encoding="utf-8") as f:
            self.data: Dict[str, Any] = json.load(f)

        self.features: List[str] = self.data["features"]
        self.entities: Dict[str, Any] = self.data["entities"]
        self.hierarchy: Dict[str, Dict[str, float]] = self.data["hierarchy"]
        self.category_parents: Dict[str, str] = self.data.get("category_parents", {})
        self.features_description: Dict[str, str] = self.data.get("features_description", {})

    def get_entity_features(self, entity_name: str) -> Dict[str, float]:
        if entity_name not in self.entities:
            return {}
        return self.entities[entity_name].get("features", {})

    def get_entity_defaults(self, entity_name: str) -> Dict[str, float]:
        if entity_name not in self.entities:
            return {}
        return self.entities[entity_name].get("defaults", {})

    def get_entity_category(self, entity_name: str) -> Optional[str]:
        if entity_name not in self.entities:
            return None
        return self.entities[entity_name].get("category")

    def get_feature_value(
        self,
        entity_name: str,
        feature_name: str,
        context: Optional[Dict[str, Any]] = None
    ) -> float:
        entity_features = self.get_entity_features(entity_name)
        if feature_name in entity_features:
            return float(entity_features[feature_name])

        if context and feature_name in context:
            return float(context[feature_name])

        return 0.0

    def lookup_value(
        self,
        entity_name: str,
        dimension: str,
        default: float = 0.0
    ) -> float:
        if entity_name not in self.entities:
            category = self._get_entity_category_fallback(entity_name)
            if category:
                return self._lookup_category_value(category, dimension, default)
            return default

        entity_defaults = self.entities[entity_name].get("defaults", {})
        if dimension in entity_defaults:
            return float(entity_defaults[dimension])

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
        if category in self.hierarchy and dimension in self.hierarchy[category]:
            return float(self.hierarchy[category][dimension])

        if category in self.category_parents:
            parent = self.category_parents[category]
            return self._lookup_category_value(parent, dimension, default)

        if category in self.hierarchy and dimension in self.hierarchy["entity"]:
            return float(self.hierarchy["entity"][dimension])

        return default

    def _get_entity_category_fallback(self, entity_name: str) -> Optional[str]:
        return None

    def get_default_values(
        self,
        entity_name: str
    ) -> Dict[str, float]:
        result = {
            "safety": 0.0,
            "task": 0.0,
            "exploration": 0.0
        }

        if entity_name not in self.entities:
            category = self._get_entity_category_fallback(entity_name)
            if category:
                for dim in result:
                    result[dim] = self._lookup_category_value(category, dim, result[dim])
            return result

        entity_defaults = self.entities[entity_name].get("defaults", {})
        for dim in result:
            if dim in entity_defaults:
                result[dim] = float(entity_defaults[dim])
            else:
                category = self.get_entity_category(entity_name)
                if category:
                    result[dim] = self._lookup_category_value(category, dim, result[dim])

        return result

    def get_all_features_for_entity(
        self,
        entity_name: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        result = {}
        for feature in self.features:
            result[feature] = self.get_feature_value(entity_name, feature, context)
        return result

    def get_category_hierarchy(self, category: str) -> List[str]:
        hierarchy = [category]
        current = category
        while current in self.category_parents:
            parent = self.category_parents[current]
            hierarchy.append(parent)
            current = parent
        hierarchy.append("entity")
        return hierarchy

    def entity_exists(self, entity_name: str) -> bool:
        return entity_name in self.entities

    def get_all_entities(self) -> List[str]:
        return list(self.entities.keys())

    def get_entities_by_category(self, category: str) -> List[str]:
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
        return self.get_default_values(entity_name)

    def get_all_categories(self) -> List[str]:
        return list(set(self.category_parents.keys()) | set(self.hierarchy.keys()))


def load_entity_registry(
    json_path: Optional[str | Path] = None
) -> EntityRegistry:
    if json_path is None:
        current_dir = Path(__file__).parent
        json_path = current_dir / "ckpt" / "entity_registry.json"
    return EntityRegistry(json_path)

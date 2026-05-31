"""
Phase 1 单元测试 - 评估式接口架构核心组件测试

测试组件：
1. EntityRegistry - 实体注册表
2. ConstraintEngine - 约束求值引擎
3. ValueMatrix - 价值矩阵
"""

import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from voyager.evaluative.entity_registry import EntityRegistry, load_entity_registry
from voyager.evaluative.constraint_engine import ConstraintEngine, load_constraint_engine
from voyager.evaluative.value_matrix import ValueMatrix, load_value_matrix
from voyager.evaluative.schemas import StructuredAction, Observation


class TestEntityRegistry(unittest.TestCase):
    """EntityRegistry 单元测试"""

    @classmethod
    def setUpClass(cls):
        cls.registry = load_entity_registry()

    def test_load_entity_registry(self):
        """测试实体注册表加载"""
        self.assertIsNotNone(self.registry)
        self.assertIsInstance(self.registry, EntityRegistry)

    def test_get_entity_features(self):
        """测试获取实体特征"""
        features = self.registry.get_entity_features("zombie")
        self.assertIn("attack_damage", features)
        self.assertIn("hostile", features)

        features = self.registry.get_entity_features("oak_log")
        self.assertIn("minable", features)

    def test_get_entity_defaults(self):
        """测试获取实体默认价值"""
        zombie_defaults = self.registry.get_entity_defaults("zombie")
        self.assertIn("safety", zombie_defaults)
        self.assertLess(zombie_defaults["safety"], 0)

        oak_defaults = self.registry.get_entity_defaults("oak_log")
        self.assertGreater(oak_defaults["task"], 0)

    def test_lookup_value(self):
        """测试价值查询（带继承）"""
        safety = self.registry.lookup_value("zombie", "safety")
        self.assertLess(safety, 0)

        task = self.registry.lookup_value("oak_log", "task")
        self.assertGreater(task, 0)

    def test_get_default_values(self):
        """测试获取三维默认价值"""
        zombie_values = self.registry.get_default_values("zombie")
        self.assertIn("safety", zombie_values)
        self.assertIn("task", zombie_values)
        self.assertIn("exploration", zombie_values)

        self.assertEqual(len(zombie_values), 3)

    def test_get_all_features_for_entity(self):
        """测试获取实体所有特征"""
        features = self.registry.get_all_features_for_entity("zombie")
        self.assertIsInstance(features, dict)
        self.assertIn("attack_damage", features)
        self.assertIn("hostile", features)

    def test_entity_exists(self):
        """测试实体存在检查"""
        self.assertTrue(self.registry.entity_exists("zombie"))
        self.assertTrue(self.registry.entity_exists("oak_log"))
        self.assertTrue(self.registry.entity_exists("creeper"))
        self.assertFalse(self.registry.entity_exists("nonexistent_entity"))

    def test_get_all_entities(self):
        """测试获取所有实体"""
        entities = self.registry.get_all_entities()
        self.assertIsInstance(entities, list)
        self.assertIn("zombie", entities)
        self.assertIn("oak_log", entities)
        self.assertGreater(len(entities), 10)

    def test_get_entities_by_category(self):
        """测试按类别获取实体"""
        hostile_mobs = self.registry.get_entities_by_category("hostile_mob")
        self.assertIsInstance(hostile_mobs, list)

    def test_category_hierarchy(self):
        """测试类别继承链"""
        hierarchy = self.registry.get_category_hierarchy("hostile_mob")
        self.assertIn("hostile_mob", hierarchy)
        self.assertIn("entity", hierarchy)

    def test_compute_value_vector(self):
        """测试价值向量计算"""
        vector = self.registry.compute_value_vector("zombie")
        self.assertIn("safety", vector)
        self.assertIn("task", vector)
        self.assertIn("exploration", vector)


class TestConstraintEngine(unittest.TestCase):
    """ConstraintEngine 单元测试"""

    @classmethod
    def setUpClass(cls):
        cls.engine = load_constraint_engine()

    def test_load_constraint_engine(self):
        """测试约束引擎加载"""
        self.assertIsNotNone(self.engine)
        self.assertIsInstance(self.engine, ConstraintEngine)

    def test_get_all_templates(self):
        """测试获取所有约束模板"""
        templates = self.engine.get_all_templates()
        self.assertIsInstance(templates, list)
        self.assertGreaterEqual(len(templates), 10)

    def test_get_templates_by_level(self):
        """测试按级别获取约束模板"""
        l0_templates = self.engine.get_l0_templates()
        l1_templates = self.engine.get_l1_templates()
        l2_templates = self.engine.get_l2_templates()

        self.assertIsInstance(l0_templates, list)
        self.assertIsInstance(l1_templates, list)
        self.assertIsInstance(l2_templates, list)

        for t in l0_templates:
            self.assertEqual(t["level"], "L0")

    def test_evaluate_no_constraint(self):
        """测试无约束触发情况"""
        action = StructuredAction(type="mine", target="oak_log")
        observation = {
            "health_percent": 1.0,
            "nearby_entities": [],
            "nearby_blocks": [],
            "target_block": "oak_log",
            "position": {"y": 64},
            "equipped_item": "wooden_pickaxe",
            "inventory": {},
            "hunger_percent": 1.0,
        }

        result = self.engine.evaluate(action, observation)
        self.assertTrue(result.allowed)

    def test_evaluate_l0_health_constraint(self):
        """测试 L0 健康度硬约束"""
        action = StructuredAction(type="attack", target="zombie")
        observation = {
            "health_percent": 0.1,
            "nearby_entities": [],
            "nearby_blocks": [],
            "target_block": "zombie",
            "position": {"y": 64},
            "hunger_percent": 1.0,
        }

        result = self.engine.evaluate(action, observation)
        self.assertFalse(result.allowed)
        self.assertEqual(result.level, "L0")
        self.assertIsNotNone(result.required_action)

    def test_evaluate_l0_creeper_constraint(self):
        """测试 L0 苦力怕 proximity 约束"""
        action = StructuredAction(type="approach", target="zombie")
        observation = {
            "health_percent": 1.0,
            "nearby_entities": [
                {"name": "creeper", "distance": 4}
            ],
            "nearby_blocks": [],
            "target_block": "zombie",
            "position": {"y": 64},
            "hunger_percent": 1.0,
        }

        result = self.engine.evaluate(action, observation)
        self.assertFalse(result.allowed)
        self.assertEqual(result.level, "L0")
        self.assertEqual(result.required_action, "flee")

    def test_evaluate_l1_constraint(self):
        """测试 L1 软约束（惩罚）- 使用低饥饿度触发 T07"""
        action = StructuredAction(type="sprint", target="")
        observation = {
            "health_percent": 1.0,
            "nearby_entities": [],
            "nearby_blocks": [],
            "target_block": "",
            "position": {"y": 64},
            "hunger_percent": 0.2,
        }

        result = self.engine.evaluate(action, observation)
        self.assertTrue(result.allowed)
        self.assertNotEqual(result.penalty, 0)

    def test_filter_actions_l0_only(self):
        """测试动作过滤 - L0 硬约束"""
        actions = [
            StructuredAction(type="mine", target="oak_log"),
            StructuredAction(type="attack", target="zombie"),
        ]
        observation = {
            "health_percent": 0.1,
            "nearby_entities": [],
            "nearby_blocks": [],
            "target_block": "oak_log",
            "position": {"y": 64},
            "hunger_percent": 1.0,
        }

        allowed, results = self.engine.filter_actions(actions, observation)
        self.assertLessEqual(len(allowed), 2)

    def test_filter_actions_with_penalty(self):
        """测试带惩罚的动作过滤"""
        actions = [
            StructuredAction(type="mine", target="oak_log"),
            StructuredAction(type="collect", target="dirt"),
        ]
        observation = {
            "health_percent": 1.0,
            "nearby_entities": [],
            "nearby_blocks": [],
            "equipped_item": "wooden_pickaxe",
            "target_block": "oak_log",
            "position": {"y": 64},
            "hunger_percent": 1.0,
            "inventory": {},
        }

        scored_actions, results = self.engine.filter_actions_with_penalty(actions, observation)
        self.assertIsInstance(scored_actions, list)
        self.assertIsInstance(results, dict)

    def test_get_required_action(self):
        """测试获取必需动作"""
        observation_low_health = {
            "health_percent": 0.1,
            "nearby_entities": [],
            "nearby_blocks": [],
            "position": {"y": 64},
        }

        required = self.engine.get_required_action(observation_low_health)
        self.assertIsNotNone(required)

        observation_creeper = {
            "health_percent": 1.0,
            "nearby_entities": [{"name": "creeper", "distance": 3}],
            "nearby_blocks": [],
            "position": {"y": 64},
        }

        required = self.engine.get_required_action(observation_creeper)
        self.assertEqual(required, "flee")

    def test_get_violations(self):
        """测试获取违规信息"""
        action = StructuredAction(type="attack", target="zombie")
        observation = {
            "health_percent": 0.1,
            "nearby_entities": [],
            "nearby_blocks": [],
            "target_block": "zombie",
            "position": {"y": 64},
            "hunger_percent": 1.0,
        }

        violations = self.engine.get_violations(action, observation)
        self.assertIsInstance(violations, list)
        self.assertGreater(len(violations), 0)


class TestValueMatrix(unittest.TestCase):
    """ValueMatrix 单元测试"""

    @classmethod
    def setUpClass(cls):
        cls.matrix = load_value_matrix()

    def test_load_value_matrix(self):
        """测试价值矩阵加载"""
        self.assertIsNotNone(self.matrix)
        self.assertIsInstance(self.matrix, ValueMatrix)

    def test_compute_dynamic_weights(self):
        """测试动态权重计算"""
        weights = self.matrix.compute_dynamic_weights(
            health_percent=0.8,
            threat_density=0.2,
            has_active_goal=True
        )

        self.assertIn("safety", weights)
        self.assertIn("task", weights)
        self.assertIn("exploration", weights)

        self.assertGreater(weights["task"], weights["safety"])

        total = weights["safety"] + weights["task"] + weights["exploration"]
        self.assertAlmostEqual(total, 1.0, places=5)

    def test_compute_dynamic_weights_low_health(self):
        """测试低健康度时安全权重增加"""
        weights_normal = self.matrix.compute_dynamic_weights(
            health_percent=1.0,
            threat_density=0.0,
            has_active_goal=True
        )

        weights_low = self.matrix.compute_dynamic_weights(
            health_percent=0.2,
            threat_density=0.0,
            has_active_goal=True
        )

        self.assertGreater(weights_low["safety"], weights_normal["safety"])

    def test_compute_dynamic_weights_high_threat(self):
        """测试高威胁时安全权重增加"""
        weights_normal = self.matrix.compute_dynamic_weights(
            health_percent=1.0,
            threat_density=0.0,
            has_active_goal=True
        )

        weights_dangerous = self.matrix.compute_dynamic_weights(
            health_percent=1.0,
            threat_density=1.0,
            has_active_goal=True
        )

        self.assertGreater(weights_dangerous["safety"], weights_normal["safety"])

    def test_compute_entity_value(self):
        """测试实体基础价值计算"""
        zombie_value = self.matrix.compute_entity_value("zombie")
        self.assertIn("safety", zombie_value)
        self.assertLess(zombie_value["safety"], 0)

        oak_value = self.matrix.compute_entity_value("oak_log")
        self.assertGreater(oak_value["task"], 0)

    def test_compute_feature_based_value(self):
        """测试基于特征的价值计算"""
        dynamic_weights = {"safety": 0.3, "task": 0.5, "exploration": 0.2}

        zombie_feature_value = self.matrix.compute_feature_based_value(
            "zombie", dynamic_weights
        )
        self.assertIn("safety", zombie_feature_value)
        self.assertIn("task", zombie_feature_value)
        self.assertIn("exploration", zombie_feature_value)

        oak_feature_value = self.matrix.compute_feature_based_value(
            "oak_log", dynamic_weights
        )
        self.assertIn("safety", oak_feature_value)

    def test_score_basic(self):
        """测试动作评分"""
        action = StructuredAction(type="mine", target="oak_log")
        observation = {
            "health_percent": 1.0,
            "nearby_entities": [],
            "target_block": "oak_log",
        }

        score = self.matrix.score(action, observation)
        self.assertIsNotNone(score)
        self.assertGreater(score.total, 0)

    def test_score_with_goal_progress(self):
        """测试带目标进度的评分"""
        action = StructuredAction(type="mine", target="oak_log")
        observation = {
            "health_percent": 1.0,
            "nearby_entities": [],
            "target_block": "oak_log",
        }

        score_no_goal = self.matrix.score(action, observation, goal_progress=0.0)
        score_with_goal = self.matrix.score(action, observation, goal_progress=0.5)

        self.assertGreater(score_with_goal.total, score_no_goal.total)

    def test_score_with_threat(self):
        """测试有威胁时的评分降低"""
        safe_action = StructuredAction(type="mine", target="oak_log")
        observation_safe = {
            "health_percent": 1.0,
            "nearby_entities": [],
            "target_block": "oak_log",
        }

        observation_threat = {
            "health_percent": 1.0,
            "nearby_entities": [
                {"name": "zombie", "distance": 5}
            ],
            "target_block": "oak_log",
        }

        score_safe = self.matrix.score(safe_action, observation_safe)
        score_threat = self.matrix.score(safe_action, observation_threat)

        self.assertGreater(score_safe.total, score_threat.total)

    def test_compute_action_scores(self):
        """测试批量动作评分"""
        actions = [
            StructuredAction(type="mine", target="oak_log"),
            StructuredAction(type="attack", target="zombie"),
            StructuredAction(type="equip", target="sword"),
        ]
        observation = {
            "health_percent": 1.0,
            "nearby_entities": [],
            "target_block": "oak_log",
        }

        scores = self.matrix.compute_action_scores(actions, observation)
        self.assertEqual(len(scores), 3)

        for score in scores:
            self.assertIsNotNone(score.total)

    def test_get_best_action(self):
        """测试获取最佳动作"""
        actions = [
            StructuredAction(type="mine", target="oak_log"),
            StructuredAction(type="attack", target="zombie"),
        ]
        observation = {
            "health_percent": 1.0,
            "nearby_entities": [],
            "target_block": "oak_log",
        }

        best_action, best_score = self.matrix.get_best_action(actions, observation)
        self.assertIsNotNone(best_action)
        self.assertIsNotNone(best_score)

    def test_get_best_action_empty(self):
        """测试空动作列表"""
        best_action, best_score = self.matrix.get_best_action([], {})
        self.assertIsNone(best_action)
        self.assertIsNone(best_score)

    def test_update_feature_weights(self):
        """测试更新特征权重"""
        new_weights = {
            "attack_damage": {"safety": -1.0, "task": 0.0, "exploration": 0.0},
        }

        self.matrix.update_feature_weights(new_weights)

        fw = self.matrix.get_feature_weights()
        self.assertEqual(fw.get("attack_damage", "safety"), -1.0)

    def test_apply_patch(self):
        """测试应用增量补丁"""
        patch = {
            "feature_weights": {
                "minable": {"safety": 0.0, "task": 2.0, "exploration": 1.0},
            }
        }

        self.matrix.apply_patch(patch)

        fw = self.matrix.get_feature_weights()
        self.assertEqual(fw.get("minable", "task"), 2.0)

    def test_compute_threat_density(self):
        """测试威胁密度计算"""
        observation_no_threat = {
            "nearby_entities": [],
        }
        density = self.matrix._compute_threat_density(observation_no_threat)
        self.assertEqual(density, 0.0)

        observation_threat = {
            "nearby_entities": [
                {"name": "zombie", "distance": 10},
                {"name": "skeleton", "distance": 15},
            ],
        }
        density = self.matrix._compute_threat_density(observation_threat)
        self.assertGreater(density, 0.0)
        self.assertLessEqual(density, 1.0)


class TestIntegration(unittest.TestCase):
    """集成测试 - 测试组件间协作"""

    @classmethod
    def setUpClass(cls):
        cls.registry = load_entity_registry()
        cls.engine = load_constraint_engine()
        cls.matrix = load_value_matrix()

    def test_constraint_then_value(self):
        """测试先约束后评分流程"""
        actions = [
            StructuredAction(type="mine", target="oak_log"),
            StructuredAction(type="attack", target="zombie"),
            StructuredAction(type="approach", target="creeper"),
        ]

        observation = {
            "health_percent": 1.0,
            "nearby_entities": [
                {"name": "creeper", "distance": 4}
            ],
            "nearby_blocks": [],
            "target_block": "oak_log",
            "position": {"y": 64},
            "hunger_percent": 1.0,
        }

        allowed, rejections = self.engine.filter_actions(actions, observation)
        self.assertIsInstance(allowed, list)
        self.assertIsInstance(rejections, list)

        if allowed:
            scores = self.matrix.compute_action_scores(allowed, observation)
            best_action, best_score = self.matrix.get_best_action(allowed, observation)
            self.assertIsNotNone(best_action)
            self.assertIsNotNone(best_score)

    def test_full_workflow(self):
        """测试完整工作流"""
        observation = {
            "health_percent": 0.5,
            "nearby_entities": [
                {"name": "zombie", "distance": 8}
            ],
            "nearby_blocks": [],
            "target_block": "oak_log",
            "position": {"y": 64},
            "hunger_percent": 1.0,
        }

        action = StructuredAction(type="mine", target="oak_log")

        constraint_result = self.engine.evaluate(action, observation)
        self.assertTrue(constraint_result.allowed)

        value_score = self.matrix.score(action, observation, goal_progress=0.3)
        self.assertIsNotNone(value_score)


if __name__ == "__main__":
    unittest.main(verbosity=2)

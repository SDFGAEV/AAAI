"""
Phase 2 测试 - EvaluativeLLMWorker模拟数据测试

测试目标：
1. 验证四个核心组件的生成方法接口
2. 测试JSON解析和验证逻辑
3. 确保模拟数据符合schema要求
4. 测试fallback机制

注意：此测试不调用真实LLM API，仅测试本地逻辑
"""

import json
import unittest
from pathlib import Path
import sys

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from voyager.evaluative.llm_worker import EvaluativeLLMWorker, ConstraintInstance, DiagnosisPatch
from voyager.evaluative.value_matrix import FeatureWeights
from voyager.evaluative.goal_graph import GoalGraphManager


class MockOpenAI:
    """模拟OpenAI客户端，返回预设的响应"""

    def __init__(self, api_key=None, base_url=None, responses=None):
        self.api_key = api_key
        self.base_url = base_url
        self.responses = responses or {}
        self.chat = self.Chat(self)  # 传递self作为parent

    class Chat:
        def __init__(self, parent=None):
            self.parent = parent
            self.completions = self.Completions(self)  # 传递self作为parent

        class Completions:
            def __init__(self, parent=None):
                self.parent = parent  # parent现在是Chat实例

            def create(self, model=None, messages=None, temperature=None):
                # 根据prompt内容返回预设响应
                if not messages:
                    return MockOpenAI.ChatCompletion("{}")

                system_content = messages[0]["content"] if messages else ""

                # 通过parent.parent访问MockOpenAI实例的responses
                mock_openai_instance = self.parent.parent if self.parent else None
                if not mock_openai_instance:
                    return MockOpenAI.ChatCompletion("{}")

                responses = mock_openai_instance.responses

                # 检查system_content是否包含关键词
                system_lower = system_content.lower()
                if "value function" in system_lower or "value_matrix" in system_lower:
                    return MockOpenAI.ChatCompletion(responses.get("value_matrix", "{}"))
                elif "hard constraints" in system_lower or "hard_constraints" in system_lower:
                    return MockOpenAI.ChatCompletion(responses.get("constraints", "{}"))
                elif "goal graph" in system_lower or "goal_graph" in system_lower:
                    return MockOpenAI.ChatCompletion(responses.get("goal_graph", "{}"))
                elif "feedback data" in system_lower or "incremental patches" in system_lower or "diagnosis" in system_lower:
                    return MockOpenAI.ChatCompletion(responses.get("diagnosis", "{}"))
                else:
                    return MockOpenAI.ChatCompletion("{}")

    class ChatCompletion:
        def __init__(self, content):
            class Choice:
                def __init__(self, content):
                    class Message:
                        def __init__(self, content):
                            self.content = content
                    self.message = Message(content)
            self.choices = [Choice(content)]


class TestEvaluativeLLMWorker(unittest.TestCase):
    """EvaluativeLLMWorker单元测试"""

    def setUp(self):
        # 创建模拟响应
        self.mock_responses = {
            "value_matrix": json.dumps({
                "weights": {
                    "attack_damage": {"safety": -0.5, "task": 0.0, "exploration": 0.0},
                    "move_speed": {"safety": -0.3, "task": 0.0, "exploration": 0.0},
                    "hostile": {"safety": -0.9, "task": 0.0, "exploration": 0.0},  # 修正为-0.9
                    "minable": {"safety": 0.0, "task": 1.0, "exploration": 0.5},
                    "drop_value": {"safety": 0.0, "task": 0.5, "exploration": 0.3}
                },
                "confidence": 0.85,
                "timestamp": 1746129600.0
            }),
            "constraints": json.dumps({
                "constraints": [
                    {
                        "template_id": "avoid_dangerous_entity",
                        "parameters": {"entity_type": "zombie", "min_distance": 8},
                        "priority": 2,
                        "description": "Stay away from zombies"
                    },
                    {
                        "template_id": "maintain_safe_health",
                        "parameters": {"min_health_percent": 0.4},
                        "priority": 2,
                        "description": "Keep health above 40%"
                    }
                ]
            }),
            "goal_graph": json.dumps({
                "root": "craft_1_crafting_table",
                "nodes": {
                    "mine_1_oak_log": {
                        "action_type": "mine",
                        "target_item": "oak_log",
                        "target_count": 1,
                        "action_count": 1,
                        "needs": []
                    },
                    "craft_4_oak_planks": {
                        "action_type": "craft",
                        "target_item": "oak_planks",
                        "target_count": 4,
                        "action_count": 1,
                        "needs": ["mine_1_oak_log"]
                    },
                    "craft_1_crafting_table": {
                        "action_type": "craft",
                        "target_item": "crafting_table",
                        "target_count": 1,
                        "action_count": 1,
                        "needs": ["craft_4_oak_planks"]
                    }
                }
            }),
            "diagnosis": json.dumps({
                "patches": [
                    {
                        "component": "value",
                        "operation": "update",
                        "target": "hostile",
                        "data": {"weights": {"safety": -3.0, "task": 0.0, "exploration": 0.0}},
                        "confidence": 0.9,
                        "evidence": ["Agent attacked by zombie 5 times"]
                    },
                    {
                        "component": "constraint",
                        "operation": "add",
                        "target": "zombie_avoidance",
                        "data": {"template_id": "avoid_dangerous_entity", "parameters": {"entity_type": "zombie", "min_distance": 12}, "priority": 2},
                        "confidence": 0.8,
                        "evidence": ["Zombie attacks caused health loss"]
                    }
                ]
            })
        }

        # 创建模拟客户端
        self.mock_client = MockOpenAI(
            api_key="mock_key",
            base_url="http://mock.api",
            responses=self.mock_responses
        )

        # 创建worker实例
        self.worker = EvaluativeLLMWorker(
            goal_manager=GoalGraphManager(),
            model="gpt-4.1",
            api_key="mock_key",
            base_url="http://mock.api"
        )

        # 直接替换OpenAI客户端
        import voyager.evaluative.llm_worker as llm_module
        self.original_openai = llm_module.OpenAI
        llm_module.OpenAI = lambda api_key=None, base_url=None: self.mock_client

    def tearDown(self):
        # 恢复原始OpenAI
        import voyager.evaluative.llm_worker as llm_module
        llm_module.OpenAI = self.original_openai

    def test_generate_value_matrix(self):
        """测试价值矩阵生成"""
        context = {
            "task": "Craft stone pickaxe",
            "environment": {"time_of_day": "day", "biome": "forest"},
            "entities": ["zombie", "oak_log", "cobblestone"]
        }

        result = self.worker.generate_value_matrix(context)

        self.assertIsInstance(result, FeatureWeights)
        self.assertIn("weights", result.to_dict())
        self.assertIn("confidence", result.to_dict())
        self.assertIn("timestamp", result.to_dict())

        weights = result.to_dict()["weights"]
        self.assertIn("attack_damage", weights)
        self.assertIn("hostile", weights)
        self.assertIn("minable", weights)

        # 验证权重范围
        for feature, dim_weights in weights.items():
            for dim, weight in dim_weights.items():
                if dim == "safety":
                    self.assertTrue(-1.0 <= weight <= 1.0, f"{feature}.{dim}={weight}")
                else:
                    self.assertTrue(0.0 <= weight <= 1.0, f"{feature}.{dim}={weight}")

    def test_generate_constraints(self):
        """测试约束生成"""
        context = {
            "task": "Explore forest",
            "threats": ["zombie", "skeleton"],
            "player_state": {"health": 0.8, "hunger": 0.6}
        }

        constraints = self.worker.generate_constraints(context)

        self.assertIsInstance(constraints, list)
        self.assertEqual(len(constraints), 2)

        for constraint in constraints:
            self.assertIsInstance(constraint, ConstraintInstance)
            self.assertIsInstance(constraint.template_id, str)
            self.assertIsInstance(constraint.parameters, dict)
            self.assertIsInstance(constraint.priority, int)
            self.assertIsInstance(constraint.description, str)

            # 验证模板ID
            self.assertIn(constraint.template_id, ["avoid_dangerous_entity", "maintain_safe_health"])

            # 验证参数
            if constraint.template_id == "avoid_dangerous_entity":
                self.assertIn("entity_type", constraint.parameters)
                self.assertIn("min_distance", constraint.parameters)
            elif constraint.template_id == "maintain_safe_health":
                self.assertIn("min_health_percent", constraint.parameters)

    def test_generate_goal_graph(self):
        """测试目标图生成"""
        task = "Craft 1 crafting_table"

        graph = self.worker.generate_goal_graph(task)

        # GoalGraph的验证由goal_manager完成
        self.assertIsNotNone(graph)

        # 转换为字典验证结构
        graph_dict = self.worker.goal_manager.to_dict(graph)
        self.assertIn("root", graph_dict)
        self.assertIn("nodes", graph_dict)

        nodes = graph_dict["nodes"]
        self.assertIn("mine_1_oak_log", nodes)
        self.assertIn("craft_4_oak_planks", nodes)
        self.assertIn("craft_1_crafting_table", nodes)

        # 验证依赖关系
        self.assertEqual(nodes["craft_4_oak_planks"]["needs"], ["mine_1_oak_log"])
        self.assertEqual(nodes["craft_1_crafting_table"]["needs"], ["craft_4_oak_planks"])

    def test_generate_diagnosis(self):
        """测试诊断补丁生成"""
        feedback = {
            "task": "Kill zombie",
            "success": False,
            "failures": [
                "Agent attacked by zombie 5 times",
                "Health dropped from 80% to 30%"
            ],
            "evidence": [
                "zombie_damage_times: 5",
                "health_loss: 50%"
            ]
        }

        patches = self.worker.generate_diagnosis(feedback)

        self.assertIsInstance(patches, list)
        self.assertEqual(len(patches), 2)

        for patch in patches:
            self.assertIsInstance(patch, DiagnosisPatch)
            self.assertIn(patch.component, ["value", "constraint", "goal"])
            self.assertIn(patch.operation, ["add", "update", "remove"])
            self.assertIsInstance(patch.target, str)
            self.assertIsInstance(patch.data, dict)
            self.assertTrue(0.0 <= patch.confidence <= 1.0)
            self.assertIsInstance(patch.evidence, list)

            # 验证证据列表
            for evidence in patch.evidence:
                self.assertIsInstance(evidence, str)

    def test_json_parsing_with_code_blocks(self):
        """测试带代码块的JSON解析"""
        test_cases = [
            # 标准JSON
            ('{"key": "value"}', {"key": "value"}),

            # 带json代码块
            ('```json\n{"key": "value"}\n```', {"key": "value"}),

            # 带通用代码块
            ('```\n{"key": "value"}\n```', {"key": "value"}),

            # 单引号（应被修复）
            ("{'key': 'value'}", {"key": "value"}),

            # 带尾随逗号（应被修复）
            ('{"key": "value",}', {"key": "value"}),

            # 带注释（应被移除）
            ('{"key": "value"} // comment', {"key": "value"}),

            # Python布尔值（应被转换）
            ('{"success": True, "data": None}', {"success": True, "data": None}),
        ]

        for input_str, expected in test_cases:
            with self.subTest(input=input_str):
                result = self.worker._parse_json(input_str)
                # 注意：Python布尔值转换后True变为true，None变为null
                if expected.get("success") is True:
                    self.assertEqual(result["success"], True)
                else:
                    self.assertEqual(result, expected)

    def test_fallback_mechanism(self):
        """测试fallback机制"""
        # 模拟API失败
        self.mock_client.responses = {}

        # 应该回退到本地解析器
        task = "Craft 1 crafting_table"
        graph, source, error = self.worker.generate_or_fallback_with_source(task)

        self.assertIsNotNone(graph)
        self.assertEqual(source, "fallback")
        self.assertIsInstance(error, str)

    def test_schema_validation_errors(self):
        """测试schema验证错误"""
        # 测试无效的价值矩阵
        invalid_value_matrix = {
            "weights": {
                "attack_damage": {"safety": 2.0, "task": 0.0},  # 缺少exploration，safety超出范围
            },
            "confidence": 1.5,  # 超出范围
            "timestamp": "not_a_number"
        }

        with self.assertRaises(ValueError):
            self.worker._validate_value_matrix(invalid_value_matrix)

        # 测试无效的约束
        invalid_constraints = {
            "constraints": [
                {
                    "template_id": 123,  # 不是字符串
                    "parameters": "not_a_dict"
                }
            ]
        }

        with self.assertRaises(ValueError):
            self.worker._validate_constraints(invalid_constraints)

        # 测试无效的诊断
        invalid_diagnosis = {
            "patches": [
                {
                    "component": "invalid_component",
                    "operation": "invalid_operation",
                    "target": 123,
                    "data": "not_a_dict",
                    "confidence": 1.5
                }
            ]
        }

        with self.assertRaises(ValueError):
            self.worker._validate_diagnosis(invalid_diagnosis)


if __name__ == "__main__":
    unittest.main()
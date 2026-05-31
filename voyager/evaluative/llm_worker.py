from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI

from .goal_graph import GoalGraphManager
from .schemas import GoalGraph
from .snapshot_store import SnapshotStore
from .value_matrix import FeatureWeights


@dataclass
class ConstraintInstance:
    """约束实例 - 从模板生成的参数化约束"""
    template_id: str
    parameters: Dict[str, Any]
    priority: int = 0
    description: str = ""


@dataclass
class DiagnosisPatch:
    """诊断补丁 - 基于反馈的增量更新"""
    component: str  # "value", "constraint", "goal"
    operation: str  # "add", "update", "remove"
    target: str  # 目标标识符
    data: Dict[str, Any]  # 更新数据
    confidence: float = 0.0
    evidence: List[str] = None

    def __post_init__(self):
        if self.evidence is None:
            self.evidence = []


class EvaluativeLLMWorker:
    """Low-frequency worker that generates the four core evaluative components:
    1. V (Value Matrix) - Multi-dimensional value weights for features
    2. C (Constraints) - Formalized hard constraints with parameters
    3. G (Goal Graph) - Task decomposition into goal DAG
    4. Diagnosis & Patches - Incremental patches based on feedback

    The worker calls OpenAI-compatible APIs but Controller never does.
    If API is unavailable, callers can fall back to local parsers.
    """

    def __init__(
        self,
        goal_manager: GoalGraphManager | None = None,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        snapshot_dir: str | Path = "ckpt_evaluative/goal_graphs",
        snapshot_store: SnapshotStore | None = None,
    ):
        self.goal_manager = goal_manager or GoalGraphManager()
        self.model = model or os.environ.get("VOYAGER_DAG_MODEL") or os.environ.get(
            "VOYAGER_MODEL", "gpt-4.1"
        )
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = (
            base_url
            or os.environ.get("OPENAI_API_BASE")
            or os.environ.get("OPENAI_BASE_URL")
        )
        self.snapshot_store = snapshot_store or SnapshotStore(
            goal_manager=self.goal_manager,
            goal_graph_dir=snapshot_dir,
        )

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def generate_value_matrix(self, context: Dict[str, Any]) -> FeatureWeights:
        """生成价值矩阵V：特征空间 × 多维 × 继承 × 置信度

        Args:
            context: 包含任务、环境、实体信息的上下文

        Returns:
            FeatureWeights: 特征权重对象
        """
        if not self.configured:
            raise RuntimeError("OPENAI_API_KEY is not configured for LLM value matrix generation.")

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self._load_prompt("value_matrix")},
                {"role": "user", "content": json.dumps(context, indent=2)},
            ],
            temperature=0.1,  # 稍微有点创造性
        )
        content = response.choices[0].message.content or ""
        data = self._parse_json(content)
        self._validate_value_matrix(data)  # 验证schema
        return FeatureWeights.from_dict(data)

    def generate_constraints(self, context: Dict[str, Any]) -> List[ConstraintInstance]:
        """生成形式化硬约束C：模板选择 + 参数化

        Args:
            context: 包含任务、环境、威胁信息的上下文

        Returns:
            List[ConstraintInstance]: 约束实例列表
        """
        if not self.configured:
            raise RuntimeError("OPENAI_API_KEY is not configured for LLM constraint generation.")

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self._load_prompt("hard_constraints")},
                {"role": "user", "content": json.dumps(context, indent=2)},
            ],
            temperature=0,
        )
        content = response.choices[0].message.content or ""
        data = self._parse_json(content)
        self._validate_constraints(data)  # 验证schema

        constraints = []
        for item in data.get("constraints", []):
            constraints.append(ConstraintInstance(
                template_id=item["template_id"],
                parameters=item["parameters"],
                priority=item.get("priority", 0),
                description=item.get("description", "")
            ))
        return constraints

    def generate_goal_graph(self, task: str) -> GoalGraph:
        """生成任务目标G：目标DAG（已有实现，保持兼容）"""
        if not self.configured:
            raise RuntimeError("OPENAI_API_KEY is not configured for LLM DAG generation.")

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self._load_prompt("goal_graph")},
                {"role": "user", "content": f"Task: {task}"},
            ],
            temperature=0,
        )
        content = response.choices[0].message.content or ""
        data = self._parse_json(content)
        # GoalGraph的验证由goal_manager.from_dict()完成
        return self.goal_manager.from_dict(task, data)

    def generate_diagnosis(self, feedback: Dict[str, Any]) -> List[DiagnosisPatch]:
        """生成诊断和增量补丁：基于反馈的增量更新

        Args:
            feedback: 包含追踪数据、失败原因、证据的反馈信息

        Returns:
            List[DiagnosisPatch]: 诊断补丁列表
        """
        if not self.configured:
            raise RuntimeError("OPENAI_API_KEY is not configured for LLM diagnosis generation.")

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self._load_prompt("diagnosis")},
                {"role": "user", "content": json.dumps(feedback, indent=2)},
            ],
            temperature=0.2,  # 需要一些创造性来诊断问题
        )
        content = response.choices[0].message.content or ""
        data = self._parse_json(content)
        self._validate_diagnosis(data)  # 验证schema

        patches = []
        for item in data.get("patches", []):
            patches.append(DiagnosisPatch(
                component=item["component"],
                operation=item["operation"],
                target=item["target"],
                data=item["data"],
                confidence=item.get("confidence", 0.0),
                evidence=item.get("evidence", [])
            ))
        return patches

    def generate_or_fallback(self, task: str) -> GoalGraph:
        graph, _, _ = self.generate_or_fallback_with_source(task)
        return graph

    def generate_or_fallback_with_source(
        self,
        task: str,
    ) -> tuple[GoalGraph, str, str | None]:
        try:
            graph = self.generate_goal_graph(task)
            print("LLM Worker generated and validated goal graph.")
            return graph, "llm", None
        except Exception as exc:
            print(f"LLM Worker fallback to local parser: {exc}")
            return self.goal_manager.parse_task(task), "fallback", str(exc)

    def save_snapshot(
        self,
        *,
        task: str,
        graph: GoalGraph,
        source: str,
        error: str | None = None,
    ) -> Path:
        return self.snapshot_store.save_goal_graph(
            task=task,
            graph=graph,
            source=source,
            model=self.model,
            base_url=self.base_url,
            error=error,
        )

    def _load_prompt(self, prompt_name: str = "goal_graph") -> str:
        """加载指定名称的prompt文件

        Args:
            prompt_name: prompt文件名（不带.txt后缀）

        Returns:
            str: prompt内容
        """
        prompt_path = Path(__file__).with_name("prompts") / f"{prompt_name}.txt"
        if not prompt_path.exists():
            # 如果指定prompt不存在，回退到默认的goal_graph
            if prompt_name != "goal_graph":
                print(f"Warning: Prompt '{prompt_name}' not found, falling back to 'goal_graph'")
                prompt_path = Path(__file__).with_name("prompts") / "goal_graph.txt"

        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

        return prompt_path.read_text(encoding="utf-8")

    def _parse_json(self, content: str) -> dict:
        """解析JSON内容，支持代码块格式

        Args:
            content: 包含JSON的字符串，可能被```json```包围

        Returns:
            dict: 解析后的字典

        Raises:
            json.JSONDecodeError: 如果JSON格式无效
            ValueError: 如果内容为空
        """
        content = content.strip()
        if not content:
            raise ValueError("Empty JSON content")

        # 提取代码块中的JSON
        if content.startswith("```"):
            match = re.search(r"```(?:json)?\s*(.*?)```", content, re.DOTALL)
            if match:
                content = match.group(1).strip()
            else:
                # 如果开始有```但没有结束，尝试移除开头的```
                content = re.sub(r"^```(?:json)?\s*", "", content)

        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            # 尝试修复常见的JSON格式问题
            content = self._fix_common_json_issues(content)
            return json.loads(content)

    def _fix_common_json_issues(self, content: str) -> str:
        """修复常见的JSON格式问题

        Args:
            content: 可能有格式问题的JSON字符串

        Returns:
            str: 修复后的JSON字符串
        """
        # 1. 修复单引号
        content = re.sub(r"(?<!\\)'", '"', content)

        # 2. 修复未转义的双引号
        content = re.sub(r'(?<!\\)"', r'\"', content)

        # 3. 修复尾随逗号
        content = re.sub(r',\s*}', '}', content)
        content = re.sub(r',\s*]', ']', content)

        # 4. 修复注释（移除单行和多行注释）
        content = re.sub(r'//.*', '', content)
        content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)

        # 5. 修复布尔值（Python风格转JSON风格）
        content = re.sub(r'\bTrue\b', 'true', content)
        content = re.sub(r'\bFalse\b', 'false', content)
        content = re.sub(r'\bNone\b', 'null', content)

        return content.strip()

    def _validate_value_matrix(self, data: dict) -> None:
        """验证价值矩阵JSON的schema

        Args:
            data: 解析后的价值矩阵数据

        Raises:
            ValueError: 如果schema验证失败
        """
        required_keys = {"weights", "confidence", "timestamp"}
        if not required_keys.issubset(data.keys()):
            missing = required_keys - set(data.keys())
            raise ValueError(f"Value matrix missing required keys: {missing}")

        weights = data["weights"]
        if not isinstance(weights, dict):
            raise ValueError("Value matrix 'weights' must be a dictionary")

        for feature, dim_weights in weights.items():
            if not isinstance(feature, str):
                raise ValueError(f"Feature name must be string, got {type(feature)}")

            if not isinstance(dim_weights, dict):
                raise ValueError(f"Feature '{feature}' weights must be a dictionary")

            required_dims = {"safety", "task", "exploration"}
            if not required_dims.issubset(dim_weights.keys()):
                missing = required_dims - set(dim_weights.keys())
                raise ValueError(f"Feature '{feature}' missing dimensions: {missing}")

            for dim, weight in dim_weights.items():
                if not isinstance(weight, (int, float)):
                    raise ValueError(f"Feature '{feature}' dimension '{dim}' weight must be number, got {type(weight)}")
                # 检查值范围
                if dim == "safety" and not (-1.0 <= weight <= 1.0):
                    raise ValueError(f"Feature '{feature}' safety weight {weight} out of range [-1.0, 1.0]")
                elif dim in ["task", "exploration"] and not (0.0 <= weight <= 1.0):
                    raise ValueError(f"Feature '{feature}' {dim} weight {weight} out of range [0.0, 1.0]")

        confidence = data["confidence"]
        if not isinstance(confidence, (int, float)) or not (0.0 <= confidence <= 1.0):
            raise ValueError(f"Confidence {confidence} out of range [0.0, 1.0]")

        if not isinstance(data["timestamp"], (int, float)):
            raise ValueError("Timestamp must be number")

    def _validate_constraints(self, data: dict) -> None:
        """验证约束JSON的schema

        Args:
            data: 解析后的约束数据

        Raises:
            ValueError: 如果schema验证失败
        """
        if "constraints" not in data:
            raise ValueError("Constraints data must contain 'constraints' key")

        constraints = data["constraints"]
        if not isinstance(constraints, list):
            raise ValueError("Constraints must be a list")

        for i, constraint in enumerate(constraints):
            if not isinstance(constraint, dict):
                raise ValueError(f"Constraint at index {i} must be a dictionary")

            required_keys = {"template_id", "parameters"}
            if not required_keys.issubset(constraint.keys()):
                missing = required_keys - set(constraint.keys())
                raise ValueError(f"Constraint at index {i} missing keys: {missing}")

            if not isinstance(constraint["template_id"], str):
                raise ValueError(f"Constraint at index {i} template_id must be string")

            if not isinstance(constraint["parameters"], dict):
                raise ValueError(f"Constraint at index {i} parameters must be dictionary")

            # 可选字段验证
            if "priority" in constraint and not isinstance(constraint["priority"], int):
                raise ValueError(f"Constraint at index {i} priority must be integer")

            if "description" in constraint and not isinstance(constraint["description"], str):
                raise ValueError(f"Constraint at index {i} description must be string")

    def _validate_diagnosis(self, data: dict) -> None:
        """验证诊断JSON的schema

        Args:
            data: 解析后的诊断数据

        Raises:
            ValueError: 如果schema验证失败
        """
        if "patches" not in data:
            raise ValueError("Diagnosis data must contain 'patches' key")

        patches = data["patches"]
        if not isinstance(patches, list):
            raise ValueError("Patches must be a list")

        for i, patch in enumerate(patches):
            if not isinstance(patch, dict):
                raise ValueError(f"Patch at index {i} must be a dictionary")

            required_keys = {"component", "operation", "target", "data"}
            if not required_keys.issubset(patch.keys()):
                missing = required_keys - set(patch.keys())
                raise ValueError(f"Patch at index {i} missing keys: {missing}")

            # 验证component
            component = patch["component"]
            if component not in ["value", "constraint", "goal"]:
                raise ValueError(f"Patch at index {i} invalid component: {component}")

            # 验证operation
            operation = patch["operation"]
            if operation not in ["add", "update", "remove"]:
                raise ValueError(f"Patch at index {i} invalid operation: {operation}")

            # 验证target
            if not isinstance(patch["target"], str):
                raise ValueError(f"Patch at index {i} target must be string")

            # 验证data
            if not isinstance(patch["data"], dict):
                raise ValueError(f"Patch at index {i} data must be dictionary")

            # 验证confidence
            if "confidence" in patch:
                confidence = patch["confidence"]
                if not isinstance(confidence, (int, float)) or not (0.0 <= confidence <= 1.0):
                    raise ValueError(f"Patch at index {i} confidence {confidence} out of range [0.0, 1.0]")

            # 验证evidence
            if "evidence" in patch:
                evidence = patch["evidence"]
                if not isinstance(evidence, list):
                    raise ValueError(f"Patch at index {i} evidence must be list")
                for j, item in enumerate(evidence):
                    if not isinstance(item, str):
                        raise ValueError(f"Patch at index {i} evidence item at index {j} must be string")

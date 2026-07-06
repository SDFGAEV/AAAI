"""
ContextBucket: 将高维上下文编码为可泛化的 bucket 字符串。

Key 设计原则：不直接用完整 context（样本太稀疏），
而是泛化成 bucket，同类场景共享统计。
"""

from typing import Dict, Optional


class ContextBucket:
    """
    上下文编码器。

    实际实现中可自定义编码函数。默认使用分层回退：
    Level 0: global（全局统计）
    Level 1: knowledge type
    Level 2: subgoal_type + failure_type
    Level 3: specific knowledge
    Level 4: specific knowledge + context
    """

    def __init__(self, cfg: Optional[Dict] = None):
        self.cfg = cfg or {}

    def encode(self, knowledge_type: str,
               subgoal_type: str = "",
               failure_type: str = "",
               task_tier: str = "",
               inventory_signature: str = "") -> str:
        """
        将上下文编码为 bucket 字符串。

        Args:
            knowledge_type: "skill" / "remedy"
            subgoal_type: "craft" / "mine" / "smelt" / "fight" / "navigate"
            failure_type: "missing_tool" / "missing_prerequisite" / "gui_failure" / ...
            task_tier: "wood" / "stone" / "iron" / "diamond"
            inventory_signature: "has_required_tool" / "missing_required_tool"

        Returns:
            context bucket string for TrustStore key
        """
        parts = [knowledge_type]
        if subgoal_type:
            parts.append(subgoal_type)
        if failure_type:
            parts.append(failure_type)
        if task_tier:
            parts.append(task_tier)
        if inventory_signature:
            parts.append(inventory_signature)
        return "/".join(parts)

    @staticmethod
    def encode_from_env(env_status: Dict, knowledge_type: str) -> str:
        """
        直接从 env_status 编码上下文。
        """
        subgoal = env_status.get("current_subgoal", "")
        failure = env_status.get("failure_type", "")
        tier = env_status.get("task_tier", "")
        inv = "has_tool" if env_status.get("has_required_tool", False) else "no_tool"
        bucket = ContextBucket()
        return bucket.encode(knowledge_type, subgoal, failure, tier, inv)

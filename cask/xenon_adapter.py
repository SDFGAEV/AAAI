"""
XENON Adapter: CASK Trust 集成到 XENON 的 DecomposedMemory 和 HRG 中。

不改 XENON 源码，通过适配层替换关键判断逻辑。
"""

from typing import Any, Dict, List, Optional, Tuple

from .trust_store import TrustStore
from .trust_gate import TrustGate
from .context_bucket import ContextBucket


class CaskDecomposedMemoryAdapter:
    """
    替换 XENON DecomposedMemory 的 is_succeeded_waypoint / save_success_failure
    使用 Beta LCB 替代 success-failure 阈值。
    """

    def __init__(self, trust_store: TrustStore, trust_gate: TrustGate):
        self.store = trust_store
        self.gate = trust_gate
        self.bucket = ContextBucket()

    def record_outcome(self, waypoint: str, language_action_str: str,
                       is_success: bool, context_info: Optional[Dict] = None):
        """
        替代 DecomposedMemory.save_success_failure()。
        用 TrustStore 的 Beta 后验记录，支持软标签。
        """
        knowledge_type = "skill"
        subgoal_type = self._infer_subgoal_type(waypoint, language_action_str)

        ctx = self.bucket.encode(
            knowledge_type=knowledge_type,
            subgoal_type=subgoal_type,
            task_tier=self._infer_tier(waypoint),
        )

        success = 1.0 if is_success else 0.0
        weight = 1.0  # postcondition 检查 = 精确标签
        self.store.record_outcome(
            f"{knowledge_type}:{waypoint}", ctx, success, weight
        )

    def is_trustworthy(self, waypoint: str,
                       context: Optional[str] = None) -> bool:
        """
        替代 DecomposedMemory.is_succeeded_waypoint()。
        检查该 waypoint 的 LCB 是否通过 skill 门控。
        """
        if context is None:
            context = self.bucket.encode(knowledge_type="skill")
        return self.gate.check_skill(
            self.store, f"skill:{waypoint}", context
        )

    def get_success_lcb(self, waypoint: str) -> float:
        """返回 skill success 的 LCB 分数"""
        ctx = self.bucket.encode(knowledge_type="skill",
                                 subgoal_type="",
                                 task_tier=self._infer_tier(waypoint))
        return self.store.lcb(f"skill:{waypoint}", ctx)

    def _infer_subgoal_type(self, waypoint: str, action_str: str) -> str:
        action_lower = action_str.lower()
        if "craft" in action_lower or any(t in waypoint for t in
                                         ["pickaxe", "axe", "sword", "hoe", "shovel",
                                          "chest", "furnace", "crafting_table"]):
            return "craft"
        if "mine" in action_lower or "smelt" in action_lower or "ore" in waypoint:
            return "mine"
        if "fight" in action_lower or "kill" in action_lower or "sword" in waypoint:
            return "fight"
        if any(t in waypoint for t in ["log", "wood", "plank", "stick"]):
            return "collect"
        return "craft"

    def _infer_tier(self, item: str) -> str:
        if any(t in item for t in ["diamond", "netherite"]):
            return "diamond"
        if any(t in item for t in ["iron", "gold", "redstone"]):
            return "iron"
        if any(t in item for t in ["stone", "copper", "coal", "furnace"]):
            return "stone"
        if any(t in item for t in ["wood", "log", "plank", "stick", "crafting"]):
            return "wood"
        return "stone"


class CaskHrgAdapter:
    """
    CASK Trust 集成到 HypothesizedRecipeGraph（ADG）中。
    在使用修正配方前检查其可信度。
    """

    def __init__(self, trust_store: TrustStore, trust_gate: TrustGate):
        self.store = trust_store
        self.gate = trust_gate
        self.bucket = ContextBucket()

    def record_recipe_outcome(self, item_name: str,
                              hypothesis_used: bool,
                              is_success: bool):
        """记录修正配方使用结果"""
        ctx = self.bucket.encode(
            knowledge_type="remedy",
            subgoal_type="craft",
            task_tier=self._infer_tier(item_name),
        )
        success = 1.0 if is_success else 0.0
        weight = 1.0
        kid = f"hypothesis:{item_name}"
        self.store.record_outcome(kid, ctx, success, weight)

    def is_hypothesis_trustworthy(self, item_name: str) -> bool:
        """修正配方是否可信"""
        ctx = self.bucket.encode(
            knowledge_type="remedy",
            subgoal_type="craft",
            task_tier=self._infer_tier(item_name),
        )
        lcb = self.store.lcb(f"hypothesis:{item_name}", ctx)
        return lcb >= 0.3  # 修正配方至少 30% LCB

    def _infer_tier(self, item: str) -> str:
        if any(t in item for t in ["diamond", "netherite"]):
            return "diamond"
        if any(t in item for t in ["iron", "gold", "redstone"]):
            return "iron"
        if any(t in item for t in ["stone", "copper", "coal", "furnace"]):
            return "stone"
        if any(t in item for t in ["wood", "log", "plank", "stick", "crafting"]):
            return "wood"
        return "stone"

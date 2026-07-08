"""
XENON Adapter — unified interface between C-ACT and XENON memory.

Provides a standard API for extracting knowledge contracts from
XENON's knowledge structures and feeding decisions back.
"""

from typing import Dict, List, Optional
from .adg_parser import ADGParser
from .fam_parser import FAMParser


class XenonAdapter:
    """Unified adapter between C-ACT and XENON memory system."""

    def __init__(self):
        self.adg = ADGParser()
        self.fam = FAMParser()

    def extract_knowledge(self, xenon_memory) -> List[Dict]:
        """Extract all contract-ready knowledge from XENON memory.

        Returns list of knowledge dicts ready for ContractExtractor.
        """
        knowledge_list = []

        # Extract from Adaptive Dependency Graph
        dep_knowledge = self.adg.extract_corrections(xenon_memory)
        knowledge_list.extend(dep_knowledge)

        # Extract from Failure-aware Action Memory
        action_knowledge = self.fam.extract_corrections(xenon_memory)
        knowledge_list.extend(action_knowledge)

        return knowledge_list

    def get_context(self, waypoint: str, task: Dict,
                    xenon_memory) -> Dict:
        """Build context dict for C-ACT decision."""
        return {
            "waypoint": waypoint,
            "subgoal_type": self._infer_subgoal_type(waypoint),
            "failure_type": self._infer_failure_type(waypoint, xenon_memory),
            "task_tier": task.get("tier", "stone"),
            "risk_level": self._infer_risk(waypoint),
        }

    @staticmethod
    def _infer_subgoal_type(waypoint: str) -> str:
        w = waypoint.lower()
        if any(t in w for t in ["craft", "make", "build"]): return "craft"
        if any(t in w for t in ["mine", "collect", "gather", "dig"]): return "mine"
        if any(t in w for t in ["smelt"]): return "smelt"
        if any(t in w for t in ["equip", "wear", "wield"]): return "equip"
        if any(t in w for t in ["explore", "find", "locate"]): return "explore"
        if any(t in w for t in ["kill", "fight", "defeat"]): return "combat"
        return "craft"

    @staticmethod
    def _infer_failure_type(waypoint: str, xenon_memory) -> str:
        w = waypoint.lower()
        if any(t in w for t in ["wrong_tool", "wrong pick"]): return "wrong_tool"
        if any(t in w for t in ["missing", "not found", "no "]):
            return "missing_resource"
        if any(t in w for t in ["failed", "cannot", "unable"]): return "execution_failure"
        return "none"

    @staticmethod
    def _infer_risk(waypoint: str) -> str:
        w = waypoint.lower()
        if any(t in w for t in ["diamond", "lava", "ender", "nether"]): return "high"
        if any(t in w for t in ["iron", "obsidian", "enchant"]): return "medium"
        return "low"

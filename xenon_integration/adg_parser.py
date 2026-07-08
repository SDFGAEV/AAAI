"""
Adaptive Dependency Graph (ADG) Parser.

Parses XENON's Adaptive Dependency Graph for dependency
corrections that can be turned into Knowledge Contracts.

XENON ADG stores:
  - Item-to-item dependency relationships
  - Corrected prerequisites
  - Planning dependency edges
"""

from typing import Dict, List, Optional


class ADGParser:
    """Parse XENON's Adaptive Dependency Graph for contract extraction."""

    def extract_corrections(self, xenon_memory) -> List[Dict]:
        """Extract dependency corrections from XENON's ADG.

        Args:
            xenon_memory: XENON DecomposedMemory instance

        Returns:
            List of knowledge dicts with keys:
            {source, type, subgoal_type, correction, preconditions,
             postconditions, episode_id, task_tier}
        """
        corrections = []

        # Access XENON's hypothesized recipe graph if available
        if hasattr(xenon_memory, 'recipe_graph'):
            graph = xenon_memory.recipe_graph
            if hasattr(graph, 'get_corrected_edges'):
                for edge in graph.get_corrected_edges():
                    corrections.append(self._parse_edge(edge))

        # Access XENON's relative graph if available
        if hasattr(xenon_memory, 'knowledge_graph'):
            kg = xenon_memory.knowledge_graph
            if hasattr(kg, 'get_dependencies'):
                for dep in kg.get_dependencies():
                    corrections.append(self._parse_dependency(dep))

        return corrections

    def _parse_edge(self, edge: Dict) -> Dict:
        """Parse a corrected ADG edge into contract-ready format."""
        return {
            "source": "XENON_ADG",
            "type": "dependency_correction",
            "subgoal_type": "craft",
            "correction": edge.get("correction",
                f"Crafting {edge.get('target', 'item')} requires "
                f"{edge.get('source', 'prerequisite')}"),
            "preconditions": edge.get("preconditions", []),
            "postconditions": ["dependency_satisfied"],
            "episode_id": edge.get("episode", ""),
            "task_tier": edge.get("tier", "stone"),
        }

    def _parse_dependency(self, dep: Dict) -> Dict:
        """Parse a graph dependency into contract-ready format."""
        return {
            "source": "XENON_ADG",
            "type": "dependency_correction",
            "subgoal_type": dep.get("type", "craft"),
            "correction": dep.get("description",
                f"{dep.get('target', 'target')} requires "
                f"{dep.get('source', 'source')}"),
            "preconditions": [f"has_{dep.get('source', 'prerequisite')}"],
            "postconditions": ["dependency_satisfied"],
            "episode_id": dep.get("episode", ""),
            "task_tier": dep.get("tier", "stone"),
        }

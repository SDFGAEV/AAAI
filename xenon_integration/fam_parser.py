"""
Failure-aware Action Memory (FAM) Parser.

Parses XENON's Failure-aware Action Memory for action
corrections and remedies that can be turned into
Knowledge Contracts.

XENON FAM stores:
  - Failed actions and their corrections
  - Typed failure feedback
  - Successful remedies
"""

from typing import Dict, List, Optional


class FAMParser:
    """Parse XENON's Failure-aware Action Memory for contract extraction."""

    def extract_corrections(self, xenon_memory) -> List[Dict]:
        """Extract action corrections from XENON's FAM.

        Args:
            xenon_memory: XENON DecomposedMemory instance

        Returns:
            List of knowledge dicts with keys:
            {source, type, subgoal_type, failure_type, correction,
             preconditions, postconditions, episode_id, task_tier}
        """
        corrections = []

        # Extract from XENON's failed subgoal memory
        if hasattr(xenon_memory, 'retrieve_failed_subgoals'):
            # Iterate over known waypoints
            for wp in getattr(xenon_memory, 'succeeded_waypoints', []):
                failed = xenon_memory.retrieve_failed_subgoals(wp)
                if failed:
                    for f in failed:
                        corrections.append(self._parse_failure(wp, f))

        # Extract from XENON's decomposed memory success/failure pairs
        if hasattr(xenon_memory, '_success_history') and \
           hasattr(xenon_memory, '_failure_history'):
            for wp in xenon_memory._success_history:
                if wp in xenon_memory._failure_history:
                    fail_count = xenon_memory._failure_history[wp]
                    if fail_count > 0:
                        corrections.append(self._parse_correction_pair(
                            wp, fail_count))

        return corrections

    def _parse_failure(self, waypoint: str, failure_info) -> Dict:
        """Parse a failure entry into contract-ready format."""
        fail_dict = failure_info if isinstance(failure_info, dict) else \
                    {"description": str(failure_info)}

        return {
            "source": "XENON_FAM",
            "type": "action_correction",
            "subgoal_type": self._infer_subgoal(waypoint),
            "failure_type": fail_dict.get("type", "execution_failure"),
            "correction": fail_dict.get("correction",
                f"Do not use wrong tool for {waypoint}; use correct tool instead."),
            "preconditions": fail_dict.get("preconditions", []),
            "postconditions": ["failure_resolved"],
            "episode_id": fail_dict.get("episode", waypoint),
            "task_tier": fail_dict.get("tier", "stone"),
        }

    def _parse_correction_pair(self, waypoint: str,
                                fail_count: int) -> Dict:
        """Parse a success-after-failure pair into contract-ready format."""
        return {
            "source": "XENON_FAM",
            "type": "action_correction",
            "subgoal_type": self._infer_subgoal(waypoint),
            "failure_type": "execution_failure",
            "correction": f"After {int(fail_count)} failure(s), "
                          f"corrected approach for {waypoint}.",
            "preconditions": [],
            "postconditions": ["failure_resolved"],
            "episode_id": waypoint,
            "task_tier": "stone",
        }

    @staticmethod
    def _infer_subgoal(waypoint: str) -> str:
        w = waypoint.lower()
        if any(t in w for t in ["craft", "make"]): return "craft"
        if any(t in w for t in ["mine", "collect"]): return "mine"
        if any(t in w for t in ["smelt"]): return "smelt"
        if any(t in w for t in ["equip"]): return "equip"
        return "craft"

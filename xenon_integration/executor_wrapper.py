"""
XENON Executor Wrapper — safe execution with contract checking.

Wraps XENON controller execution to verify contract conditions
before and after knowledge reuse, logging results.
"""

from typing import Dict, Tuple, Optional


class ExecutorWrapper:
    """Wrap XENON controller execution with contract verification."""

    def __init__(self, contract_checker=None):
        from cact.contract import ContractChecker
        self.checker = contract_checker or ContractChecker()

    def execute_with_contract_check(self, controller_func,
                                     contract: Dict,
                                     state_before: Dict,
                                     *args, **kwargs) -> Tuple[any, Dict]:
        """Execute controller function with pre/post contract verification.

        Args:
            controller_func: XENON controller execution function
            contract: KnowledgeContract dict
            state_before: State snapshot before execution
            *args, **kwargs: Passed to controller_func

        Returns:
            (result, verification_dict) with keys:
            {pre_ok, post_ok, violations_before, violations_after,
             progress_delta, contract_violation}
        """
        from cact.contract import KnowledgeContract

        # Pre-check
        contract_obj = KnowledgeContract.from_dict(contract)
        pre_ok, pre_violations = self.checker.check_preconditions(
            contract_obj, state_before)

        verification = {
            "pre_ok": pre_ok,
            "violations_before": pre_violations,
        }

        if not pre_ok:
            verification["post_ok"] = False
            verification["violations_after"] = ["precondition_not_met"]
            verification["progress_delta"] = 0.0
            verification["contract_violation"] = True
            return None, verification

        # Execute
        result = controller_func(*args, **kwargs)

        # Post-check (simplified — in practice needs state_after)
        state_after = kwargs.get("state_after", {})
        post_ok, post_violations = self.checker.check_postconditions(
            contract_obj, state_before, state_after)

        verification.update({
            "post_ok": post_ok,
            "violations_after": post_violations,
            "progress_delta": self._estimate_progress(state_before, state_after),
            "contract_violation": not post_ok,
        })

        return result, verification

    @staticmethod
    def _estimate_progress(state_before: Dict, state_after: Dict) -> float:
        """Estimate progress delta between states (0-1)."""
        if not state_after:
            return 0.0
        # Simple heuristic: inventory change
        inv_before = state_before.get("inventory_count", 0)
        inv_after = state_after.get("inventory_count", 0)
        if inv_before == 0:
            return min(1.0, inv_after / 10.0)
        return min(1.0, max(-0.5, (inv_after - inv_before) / max(inv_before, 1)))

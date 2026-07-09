"""
Attribution-Aware Outcome Classification for C-ACT Lifecycle.

Minecraft failures have many causes besides bad knowledge:
  - Navigation failure (got lost, couldn't reach target)
  - Environment randomness (mob interference, block placement RNG)
  - Executor failure (VLM hallucination, Steve-1 action error)
  - Resource conflict (competing knowledge for same resource)
  - Contract violation (postcondition not met)
  - Base planner failure (would have failed anyway)

C-ACT must distinguish these before updating lifecycle.
Otherwise, good knowledge is penalized for failures it didn't cause.

Design doc §14: "好知识因为导航失败被误杀" must be prevented.
"""

from typing import Dict, List, Optional
from enum import Enum


class AttributionLabel(str, Enum):
    """Outcome attribution for lifecycle updates."""
    KNOWLEDGE_CAUSED_SUCCESS = "knowledge_caused_success"
    BASE_WOULD_SUCCEED      = "base_would_succeed"       # base succeeded without knowledge
    BASE_ALSO_FAILED        = "base_also_failed"         # base also failed (knowledge not the diff)
    ENVIRONMENT_FAILURE     = "environment_failure"
    NAVIGATION_FAILURE      = "navigation_failure"
    EXECUTION_FAILURE       = "execution_failure"
    RESOURCE_CONFLICT       = "resource_conflict"
    CONTRACT_VIOLATION      = "contract_violation"
    CHAIN_FAILURE           = "chain_failure"
    HARMFUL_REUSE           = "harmful_reuse"
    UNCERTAIN               = "uncertain"


class OutcomeAttributor:
    """Attribute episode outcomes to knowledge vs. external factors."""

    def attribute(self, success: bool, is_harmful: bool,
                  contract_violated: bool,
                  interaction_conflict: bool,
                  used_knowledge: bool,
                  progress_delta: float = 0.0,
                  postcondition_satisfied: Optional[bool] = None,
                  base_would_have_succeeded: Optional[bool] = None,
                  execution_errors: int = 0,
                  navigation_errors: int = 0,
                  ) -> AttributionLabel:
        """Attribute outcome to the most likely cause.

        Called after each knowledge use or fallback decision.
        """

        # ── Success cases ──
        if success:
            if not used_knowledge:
                # Base/fallback succeeded — counterfactual signal
                return AttributionLabel.BASE_WOULD_SUCCEED
            # Knowledge was used and succeeded
            if is_harmful:
                return AttributionLabel.HARMFUL_REUSE  # Succeeded but caused harm
            if contract_violated:
                return AttributionLabel.CONTRACT_VIOLATION
            return AttributionLabel.KNOWLEDGE_CAUSED_SUCCESS

        # ── Failure cases ──
        if not used_knowledge:
            # Base planner failed on its own — knowledge wasn't the differentiator
            return AttributionLabel.BASE_ALSO_FAILED

        # Knowledge was used and task failed
        if is_harmful:
            return AttributionLabel.HARMFUL_REUSE

        if contract_violated:
            return AttributionLabel.CONTRACT_VIOLATION

        if interaction_conflict:
            return AttributionLabel.CHAIN_FAILURE

        # Postcondition check: if knowledge's promises were kept
        # but overall task still failed, blame environment/execution
        if postcondition_satisfied is True:
            if progress_delta > 0:
                # Knowledge helped but something else failed
                if navigation_errors > execution_errors:
                    return AttributionLabel.NAVIGATION_FAILURE
                return AttributionLabel.EXECUTION_FAILURE
            return AttributionLabel.ENVIRONMENT_FAILURE

        # Postcondition violated — knowledge didn't deliver
        if postcondition_satisfied is False:
            # Check if base would have succeeded
            if base_would_have_succeeded is True:
                return AttributionLabel.CONTRACT_VIOLATION
            return AttributionLabel.KNOWLEDGE_CAUSED_SUCCESS  # knowledge was the diff

        # Ambiguous: cannot determine cause
        return AttributionLabel.UNCERTAIN

    def should_penalize(self, label: AttributionLabel) -> bool:
        """Whether this attribution should increment harm/cvr counts."""
        PENALIZE = {
            AttributionLabel.HARMFUL_REUSE,
            AttributionLabel.CONTRACT_VIOLATION,
            AttributionLabel.CHAIN_FAILURE,
        }
        return label in PENALIZE

    def should_reward(self, label: AttributionLabel) -> bool:
        """Whether this attribution should increment use/success counts."""
        REWARD = {
            AttributionLabel.KNOWLEDGE_CAUSED_SUCCESS,
        }
        return label in REWARD

    def should_count_base(self, label: AttributionLabel) -> bool:
        """Whether this attribution provides positive base (counterfactual) signal.
        Only BASE_WOULD_SUCCEED (base succeeded without knowledge) counts.
        BASE_ALSO_FAILED means knowledge wasn't the differentiator — not counted."""
        COUNT_BASE = {
            AttributionLabel.BASE_WOULD_SUCCEED,
        }
        return label in COUNT_BASE

    def is_no_fault(self, label: AttributionLabel) -> bool:
        """Whether the knowledge is blameless for this outcome."""
        NO_FAULT = {
            AttributionLabel.ENVIRONMENT_FAILURE,
            AttributionLabel.NAVIGATION_FAILURE,
            AttributionLabel.EXECUTION_FAILURE,
            AttributionLabel.RESOURCE_CONFLICT,
            AttributionLabel.BASE_ALSO_FAILED,
        }
        return label in NO_FAULT


def apply_attribution_to_lifecycle(
    kid: str,
    attribution: AttributionLabel,
    success: float,
    is_harmful: float,
    trust_store,        # TrustStore instance
    context: str,
    attributor: OutcomeAttributor = None,
) -> Dict:
    """Apply attribution-aware update to lifecycle and trust store.

    Records data and triggers lifecycle transitions, then persists.
    Returns dict with update summary for logging.
    """
    if attributor is None:
        attributor = OutcomeAttributor()

    result = {
        "knowledge_id": kid,
        "attribution": attribution.value,
        "action": "none",
    }

    used = False
    if attributor.should_reward(attribution):
        trust_store.record_use(kid, context, success, save=False)
        trust_store.record_harm(kid, context, is_harmful, save=False)
        used = True
        result["action"] = "reward"

    elif attributor.should_penalize(attribution):
        trust_store.record_use(kid, context, success, save=False)
        trust_store.record_harm(kid, context, is_harmful, save=False)
        used = True
        result["action"] = "penalize"

    elif attributor.should_count_base(attribution):
        trust_store.record_base(kid, context, success, save=False)
        result["action"] = "count_base"

    elif attributor.is_no_fault(attribution):
        result["action"] = "no_fault"

    else:
        result["action"] = "uncertain"

    # Trigger lifecycle transitions (same logic as record_episode)
    if used and getattr(trust_store, 'abl_lifecycle', True):
        n = trust_store.total_count(kid, context, "use")
        pi = trust_store.uplift_probability(kid, context)
        hu = trust_store.harm_upper_bound(kid, context)

        tau, h_star = 0.88, 0.10
        if hasattr(trust_store, '_synced_thresholds') and trust_store._synced_thresholds:
            contract = trust_store.get_contract(kid)
            group = contract.get("group", "crafting") if contract else "crafting"
            thresholds = trust_store._synced_thresholds
            tau = thresholds.get(group, {}).get("tau", 0.88)
            h_star = thresholds.get(group, {}).get("harm", 0.10)

        new_state = trust_store.lifecycle.evaluate_auto_transition(
            kid, pi, hu, tau, h_star, int(n))
        if new_state:
            trust_store.lifecycle.transition(kid, new_state, "auto_after_observation")

    # Persist all changes
    trust_store._save()

    return result

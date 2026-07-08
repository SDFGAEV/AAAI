"""
Knowledge Lifecycle Manager — 6-state state machine for C-ACT.

State transitions:
  Candidate → Quarantined → Probation → Certified → Deprecated → Disabled

Transitions are triggered by:
  - π_uplift   (Bayesian counterfactual uplift probability)
  - harm_ucb   (harmful reuse upper confidence bound)
  - contract violations (CVR)
  - interaction conflicts
  - recent temporal performance

Only knowledge in {Probation, Certified} can be reused.
Disabled knowledge is permanently blocked (manual review only).
"""

import json
from typing import Dict, List, Tuple, Optional
from enum import Enum


class LifecycleState(Enum):
    CANDIDATE   = "candidate"
    QUARANTINED = "quarantined"
    PROBATION   = "probation"
    CERTIFIED   = "certified"
    DEPRECATED  = "deprecated"
    DISABLED    = "disabled"


# ── String constants for external import ──
CANDIDATE   = "candidate"
QUARANTINED = "quarantined"
PROBATION   = "probation"
CERTIFIED   = "certified"
DEPRECATED  = "deprecated"
DISABLED    = "disabled"

# Ordering for comparison (later states >= earlier)
STATE_ORDER = {
    LifecycleState.CANDIDATE: 0,
    LifecycleState.QUARANTINED: 1,
    LifecycleState.PROBATION: 2,
    LifecycleState.CERTIFIED: 3,
    LifecycleState.DEPRECATED: 4,
    LifecycleState.DISABLED: 5,
}

# States that allow free reuse (no extra verification needed)
REUSABLE_STATES = {LifecycleState.CERTIFIED}

# States that allow supervised reuse (reuse + double-verification)
# Probation knowledge has 3-5 observations — posterior is still wide.
# It CAN be used, but every step must pass an immediate reflection check.
# If the check fails, fallback to base policy immediately.
SUPERVISED_STATES = {LifecycleState.PROBATION}

# States treated as "active" (counted in knowledge growth)
ACTIVE_STATES = {LifecycleState.QUARANTINED, LifecycleState.PROBATION,
                 LifecycleState.CERTIFIED}


class LifecycleManager:
    """6-state knowledge lifecycle governance."""

    def __init__(self, store_path: str = None):
        self._states: Dict[str, str] = {}   # {knowledge_id: state_string}
        self._history: Dict[str, List[Dict]] = {}  # {knowledge_id: [transition_event]}
        self._store_path = store_path
        self._lifecycle_file = None
        if store_path:
            import os
            os.makedirs(store_path, exist_ok=True)
            self._lifecycle_file = os.path.join(store_path, "lifecycle.json")
            self._load()

    # ── I/O ──
    def _load(self):
        if self._lifecycle_file:
            import os
            if os.path.exists(self._lifecycle_file):
                try:
                    with open(self._lifecycle_file) as f:
                        data = json.load(f)
                    self._states = data.get("states", {})
                    self._history = data.get("history", {})
                except Exception:
                    pass

    def _save(self):
        if self._lifecycle_file:
            with open(self._lifecycle_file, "w") as f:
                json.dump({"states": self._states, "history": self._history},
                          f, indent=2)

    # ── State queries ──
    def get_state(self, kid: str) -> LifecycleState:
        s = self._states.get(kid, "candidate")
        try:
            return LifecycleState(s)
        except ValueError:
            return LifecycleState.CANDIDATE

    def is_reusable(self, kid: str) -> bool:
        """Knowledge can be freely reused (Certified)."""
        return self.get_state(kid) in REUSABLE_STATES

    def is_supervised(self, kid: str) -> bool:
        """Knowledge can be reused but needs double verification (Probation)."""
        return self.get_state(kid) in SUPERVISED_STATES

    def can_reuse(self, kid: str) -> bool:
        """Knowledge can be reused in any mode (free or supervised)."""
        return self.get_state(kid) in REUSABLE_STATES | SUPERVISED_STATES

    def is_active(self, kid: str) -> bool:
        return self.get_state(kid) in ACTIVE_STATES

    # ── State transitions ──
    def transition(self, kid: str, new_state, reason: str = "",
                   metadata: Dict = None) -> Dict:
        """Execute a lifecycle state transition and log it.

        Accepts LifecycleState enum or string (e.g. "probation", "certified").
        """
        # Coerce string to LifecycleState
        if isinstance(new_state, str):
            try:
                new_state = LifecycleState(new_state)
            except ValueError:
                return {"transitioned": False, "reason": "invalid_state",
                        "state": new_state}
        old_state = self.get_state(kid)

        # Guard: can't go backwards (except to disable)
        if (new_state != LifecycleState.DISABLED and
            STATE_ORDER.get(new_state, 0) <= STATE_ORDER.get(old_state, 0)):
            return {"transitioned": False, "reason": "backward_not_allowed",
                    "old": old_state.value, "new": new_state.value}

        self._states[kid] = new_state.value
        event = {
            "knowledge_id": kid,
            "old_status": old_state.value,
            "new_status": new_state.value,
            "reason": reason,
            "metadata": metadata or {},
        }
        if kid not in self._history:
            self._history[kid] = []
        self._history[kid].append(event)
        self._save()
        return {"transitioned": True, "event": event}

    def evaluate_auto_transition(self, kid: str,
                                  pi_uplift: float,
                                  harm_ucb: float,
                                  tau: float, h_star: float,
                                  n_observations: int,
                                  contract_violations_recent: int = 0,
                                  interaction_conflicts_recent: int = 0) -> Optional[LifecycleState]:
        """Auto-evaluate whether a knowledge item should transition state.

        This is called after each observation update, using the
        calibrated thresholds τ* and h* for the knowledge's group/level.
        """
        current = self.get_state(kid)
        C = LifecycleState

        # CANDIDATE → QUARANTINED: after contract extraction & 1 observation
        if current == C.CANDIDATE and n_observations >= 1:
            return C.QUARANTINED

        # QUARANTINED → PROBATION: initial uplift check passed + ESS >= 3
        if current == C.QUARANTINED and n_observations >= 3 and pi_uplift > 0.3:
            return C.PROBATION

        # PROBATION → CERTIFIED: full gate passed with sufficient evidence
        if current == C.PROBATION:
            if n_observations >= 5 and pi_uplift >= tau and harm_ucb <= h_star:
                return C.CERTIFIED

        # CERTIFIED → DEPRECATED: contract violations or drift detected
        if current == C.CERTIFIED:
            if contract_violations_recent >= 2 or interaction_conflicts_recent >= 2:
                return C.DEPRECATED

        # DEPRECATED → DISABLED: repeated violations / harm confirmed
        if current == C.DEPRECATED:
            if contract_violations_recent >= 5 or harm_ucb > h_star * 2:
                return C.DISABLED

        # PROBATION → DEPRECATED: persistent failure to certify
        if current == C.PROBATION and n_observations >= 10 and pi_uplift < 0.5:
            return C.DEPRECATED

        # CERTIFIED → DEPRECATED: harm spike
        if current == C.CERTIFIED and harm_ucb > h_star * 1.5:
            return C.DEPRECATED

        return None

    def force_disable(self, kid: str, reason: str = "manual") -> Dict:
        """Force-disable a knowledge item (irreversible without manual review)."""
        return self.transition(kid, LifecycleState.DISABLED, reason)

    # ── Statistics ──
    def stats(self) -> Dict[str, int]:
        counts = {s.value: 0 for s in LifecycleState}
        for st in self._states.values():
            if st in counts:
                counts[st] += 1
        return counts

    def active_knowledge_ids(self) -> List[str]:
        return [kid for kid in self._states if self.is_active(kid)]

    def certified_knowledge_ids(self) -> List[str]:
        return [kid for kid in self._states
                if self.get_state(kid) == LifecycleState.CERTIFIED]

    def get_history(self, kid: str) -> List[Dict]:
        return self._history.get(kid, [])

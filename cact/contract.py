"""
Knowledge Contract v2: falsifiable behavioral claims for self-evolved knowledge.

C-ACT transforms raw self-evolved knowledge into structured contracts with:
  - scope          — structured applicability domain
  - preconditions  — must hold before reuse
  - postconditions — must hold after reuse
  - hard_non_applicable_contexts — safety boundaries (never reuse)
  - recovery_rule  — fallback strategy when preconditions unmet
  - termination_condition — when to stop applying
  - evidence_requirement — minimum evidence for lifecycle promotion
  - expected_uplift / risk_bound — declared expectations

Contract verification: pre-admission (scope + pre + boundary + risk)
and post-execution (postconditions + progress + no new harm).
"""

import json, uuid
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict


# ── Knowledge type & level enums (v2 expanded) ──
KNOWLEDGE_TYPES = ["skill", "remedy", "dependency_correction",
                   "action_correction", "failure_memory",
                   "planning_rule", "guardrail", "interaction_pattern"]
KNOWLEDGE_LEVELS = ["strategy", "functional", "atomic_correction",
                    "dependency", "failure_memory", "interaction"]

# ── Lifecycle states (imported from lifecycle_manager) ──
from .lifecycle_manager import (CANDIDATE, QUARANTINED, PROBATION,
                                 CERTIFIED, DEPRECATED, DISABLED)

# ── Hard safety boundaries: contexts where reuse is NEVER allowed ──
HARD_SAFETY_CONTEXTS = [
    "lava_nearby", "low_health", "combat_active",
    "near_cliff", "irreversible_resource_constraint"
]

# ── Risk level definitions ──
RISK_LEVELS = ["low", "medium", "high"]


@dataclass
class KnowledgeContract:
    """A self-evolved knowledge item turned into a falsifiable behavioral claim (v2)."""
    knowledge_id: str = field(default_factory=lambda: f"kc_{uuid.uuid4().hex[:12]}")
    source: str = ""              # "XENON_ADG" | "XENON_FAM"
    type: str = ""                # skill|remedy|dependency_correction|action_correction|failure_memory|planning_rule|guardrail|interaction_pattern
    level: str = ""               # strategy|functional|atomic_correction|dependency|failure_memory|interaction

    # Core knowledge
    gene: str = ""                # One-sentence core
    full_text: str = ""           # Complete behavioral description

    # Structured scope (v2): replaces flat claimed_context
    scope: Dict[str, str] = field(default_factory=dict)
    # Backward-compat: claimed_context maps to scope
    claimed_context: Dict[str, str] = field(default_factory=dict)

    # Contract claims
    preconditions: List[str] = field(default_factory=list)
    postconditions: List[str] = field(default_factory=list)
    expected_uplift: float = 0.05
    risk_bound: float = 0.10

    # Hard safety boundaries (v2: renamed from non_applicable_contexts)
    hard_non_applicable_contexts: List[str] = field(default_factory=list)
    # Backward-compat alias
    non_applicable_contexts: List[str] = field(default_factory=list)

    # v2 new fields
    recovery_rule: str = ""               # Fallback strategy when preconditions unmet
    termination_condition: str = ""       # When to stop applying this knowledge
    evidence_requirement: Dict[str, float] = field(default_factory=dict)
    # evidence_requirement keys: min_use, min_base, max_harm_ucb

    # Provenance
    source_episode: str = ""
    status: str = CANDIDATE

    def to_dict(self) -> Dict:
        d = asdict(self)
        # Remove backward-compat alias from serialization
        d.pop("non_applicable_contexts", None)
        d.pop("claimed_context", None)
        return d

    def get_scope(self) -> Dict[str, str]:
        """Return effective scope, preferring v2 scope field over legacy claimed_context."""
        if self.scope:
            return self.scope
        return self.claimed_context

    def get_safety_boundaries(self) -> List[str]:
        """Return effective hard safety boundaries."""
        if self.hard_non_applicable_contexts:
            return self.hard_non_applicable_contexts
        return self.non_applicable_contexts

    @classmethod
    def from_dict(cls, d: Dict) -> "KnowledgeContract":
        # Normalize legacy fields
        d = dict(d)
        if "claimed_context" in d and not d.get("scope"):
            d["scope"] = d["claimed_context"]
        if "non_applicable_contexts" in d and not d.get("hard_non_applicable_contexts"):
            d["hard_non_applicable_contexts"] = d["non_applicable_contexts"]
        d = {k: v for k, v in d.items()
             if k in cls.__dataclass_fields__}
        return cls(**d)


# ── Contract types inferred from XENON knowledge source ──

def infer_type_from_xenon(knowledge_source: str) -> str:
    """Infer contract type from XENON knowledge source."""
    if "dependency" in knowledge_source.lower():
        return "dependency_correction"
    if "action" in knowledge_source.lower() or "fam" in knowledge_source.lower():
        return "action_correction"
    if "failure" in knowledge_source.lower():
        return "failure_memory"
    if "remedy" in knowledge_source.lower():
        return "remedy"
    return "skill"


def infer_level_from_xenon(content: str, task_tier: str = "stone") -> str:
    """Infer contract level from knowledge content and task context."""
    tier_order = {"wooden": 0, "stone": 1, "iron": 2, "golden": 3, "diamond": 4}
    tier = tier_order.get(task_tier, 1)

    # Strategy: long-horizon planning rules
    if any(kw in content.lower() for kw in ["nether", "before entering", "portal",
                                              "long chain", "multi-step"]):
        return "strategy"

    # Functional: skill or remedy involving multiple sub-steps
    if any(kw in content.lower() for kw in ["craft", "smelt", "equip", "build"]):
        if len(content.split()) > 30:
            return "functional"
        return "atomic"

    # Dependency: item-to-item requirement corrections
    if any(kw in content.lower() for kw in ["requires", "prerequisite", "depends on",
                                              "needs", "obtain first"]):
        return "dependency"

    # Failure memory: retry-based
    if any(kw in content.lower() for kw in ["retry", "failed", "do not", "avoid"]):
        return "failure_memory"

    # Default by tier
    return "atomic_correction" if tier <= 2 else "functional"


# ── Contract verification ──

class ContractChecker:
    """Verify contract conditions before and after knowledge reuse (v2)."""

    # ── Pre-admission checks (4 conditions) ──

    @staticmethod
    def check_scope_match(contract: KnowledgeContract,
                          context: Dict) -> bool:
        """Check if current context falls within contract scope.

        Scope fields (task_group, subgoal_type, failure_type, task_tier)
        are compared against context bucket. A scope field with value "*"
        matches any context value.
        """
        scope = contract.get_scope()
        if not scope:
            return True  # No scope = universally applicable
        for key, val in scope.items():
            if val == "*":
                continue
            ctx_val = context.get(key, "")
            if str(ctx_val) != str(val):
                return False
        return True

    @staticmethod
    def check_context_match(contract: KnowledgeContract,
                            context: Dict) -> bool:
        """Backward-compat alias for check_scope_match."""
        return ContractChecker.check_scope_match(contract, context)

    @staticmethod
    def check_preconditions(contract: KnowledgeContract,
                            state: Dict) -> Tuple[bool, List[str]]:
        """Check all preconditions against current state.
        Returns (all_satisfied, [violated_precondition_descriptions]).
        """
        violations = []
        for cond in contract.preconditions:
            if not ContractChecker._eval_condition(cond, state):
                violations.append(cond)
        return len(violations) == 0, violations

    @staticmethod
    def check_hard_boundary(contract: KnowledgeContract,
                            state: Dict) -> Tuple[bool, List[str]]:
        """Check if any hard_non_applicable_context is triggered.
        Returns (safe, [triggered_contexts]).
        True = safe (no hard boundary triggered).
        """
        triggered = []
        for ctx in contract.get_safety_boundaries():
            if ContractChecker._eval_context_flag(ctx, state):
                triggered.append(ctx)
        return len(triggered) == 0, triggered

    @staticmethod
    def check_non_applicable(contract: KnowledgeContract,
                             state: Dict) -> Tuple[bool, List[str]]:
        """Backward-compat alias for check_hard_boundary."""
        return ContractChecker.check_hard_boundary(contract, state)

    # ── Post-execution checks ──

    @staticmethod
    def check_postconditions(contract: KnowledgeContract,
                             state_before: Dict,
                             state_after: Dict) -> Tuple[bool, List[str]]:
        """Check all postconditions after reuse.
        Returns (all_satisfied, [violated_postcondition_descriptions]).
        """
        violations = []
        for cond in contract.postconditions:
            if not ContractChecker._eval_condition(cond, state_after):
                violations.append(cond)
        return len(violations) == 0, violations

    @staticmethod
    def _eval_condition(condition: str, state: Dict) -> bool:
        """Evaluate a contract condition string against state dict.
        Supports simple patterns like:
          - "target_block == diamond_ore"
          - "current_tool not in {stone_pickaxe, wooden_pickaxe}"
          - "has_iron_pickaxe"
        """
        cond = condition.strip()
        # Pattern: "X not in {A, B, C}"
        if "not in" in cond:
            attr, vals_str = cond.split("not in", 1)
            attr = attr.strip()
            vals = {v.strip() for v in vals_str.strip(" {}").split(",")}
            return state.get(attr, "") not in vals
        # Pattern: "X in {A, B, C}"
        if " in " in cond and "not" not in cond.split("in")[0]:
            attr, vals_str = cond.split(" in ", 1)
            attr = attr.strip()
            vals = {v.strip() for v in vals_str.strip(" {}").split(",")}
            return state.get(attr, "") in vals
        # Pattern: "X == Y"
        if "==" in cond:
            attr, val = cond.split("==", 1)
            return str(state.get(attr.strip(), "")).lower() == val.strip().lower()
        # Pattern: "X" (binary check — attribute exists and is truthy)
        if cond in state:
            return bool(state[cond])
        # Pattern: just a check for existence
        return state.get(cond, None) is not None

    @staticmethod
    def _eval_context_flag(flag: str, state: Dict) -> bool:
        """Evaluate a safety context flag."""
        flag_map = {
            "lava_nearby": lambda s: s.get("near_lava", False),
            "low_health": lambda s: s.get("low_health",
                                          s.get("health", 20) < 5),
            "combat_active": lambda s: s.get("in_combat", False),
            "near_cliff": lambda s: s.get("near_cliff", False),
            "irreversible_resource_constraint": lambda s:
                s.get("resource_critical", False),
        }
        if flag in flag_map:
            return flag_map[flag](state)
        return False


# ── Contract extraction from XENON knowledge ──

class ContractExtractor:
    """Extract KnowledgeContract from raw XENON knowledge correction."""

    def extract(self, knowledge: Dict) -> KnowledgeContract:
        """Main extraction method.

        Args:
            knowledge: Raw dict from XENON with keys like:
                'type', 'subgoal', 'failure_type', 'correction',
                'preconditions', 'postconditions', 'context', 'episode_id'
        Returns:
            KnowledgeContract with filled fields.
        """
        ktype = knowledge.get("type", infer_type_from_xenon(
            knowledge.get("source", "")))
        content = knowledge.get("correction", knowledge.get("text", ""))

        # Build preconditions from XENON knowledge
        preconds = list(knowledge.get("preconditions", []))
        if not preconds and "requires" in content.lower():
            # Heuristic: extract "X requires Y" patterns
            preconds = self._extract_preconditions_heuristic(content)

        # Build postconditions from XENON knowledge
        postconds = list(knowledge.get("postconditions", []))
        if not postconds:
            postconds = self._extract_postconditions_heuristic(content, ktype)

        # Infer non-applicable contexts
        non_app = list(knowledge.get("non_applicable_contexts", []))
        non_app = self._infer_safety_contexts(content, non_app)

        contract = KnowledgeContract(
            source=knowledge.get("source", "XENON_FAM"),
            type=ktype,
            level=knowledge.get("level",
                infer_level_from_xenon(content, knowledge.get("task_tier", "stone"))),
            gene=knowledge.get("gene", self._extract_gene(content)),
            full_text=content,
            scope={
                "task_group": knowledge.get("task_group", knowledge.get("group", "")),
                "subgoal_type": knowledge.get("subgoal_type", ""),
                "failure_type": knowledge.get("failure_type", ""),
                "task_tier": knowledge.get("task_tier", "stone"),
            },
            claimed_context={
                "subgoal_type": knowledge.get("subgoal_type", ""),
                "failure_type": knowledge.get("failure_type", ""),
                "task_tier": knowledge.get("task_tier", "stone"),
            },
            preconditions=preconds,
            postconditions=postconds,
            expected_uplift=knowledge.get("expected_uplift", 0.05),
            risk_bound=knowledge.get("risk_bound", 0.10),
            hard_non_applicable_contexts=non_app,
            recovery_rule=knowledge.get("recovery_rule", ""),
            termination_condition=knowledge.get("termination_condition", ""),
            evidence_requirement={
                "min_use": knowledge.get("min_use", 5),
                "min_base": knowledge.get("min_base", 3),
                "max_harm_ucb": knowledge.get("max_harm_ucb", 0.10),
            },
            source_episode=knowledge.get("episode_id", ""),
            status=CANDIDATE,
        )
        return contract

    def extract_batch(self, knowledge_list: List[Dict]) -> List[KnowledgeContract]:
        return [self.extract(k) for k in knowledge_list]

    @staticmethod
    def _extract_gene(content: str) -> str:
        """Extract one-sentence core from knowledge text."""
        sentences = content.replace("\n", " ").split(". ")
        # Take the first substantive sentence (skip very short ones)
        for s in sentences:
            s = s.strip()
            if len(s) > 20:
                return s.rstrip(".")
        return content[:100]

    @staticmethod
    def _extract_preconditions_heuristic(content: str) -> List[str]:
        """Heuristic precondition extraction."""
        preconds = []
        content_lower = content.lower()
        if "iron_pickaxe" in content_lower:
            preconds.append("has_iron_pickaxe")
        if "diamond_pickaxe" in content_lower:
            preconds.append("has_diamond_pickaxe")
        if "furnace" in content_lower:
            preconds.append("has_furnace")
        if "crafting_table" in content_lower:
            preconds.append("has_crafting_table")
        if "fuel" in content_lower or "coal" in content_lower:
            preconds.append("has_fuel")
        return preconds

    @staticmethod
    def _extract_postconditions_heuristic(content: str, ktype: str) -> List[str]:
        """Heuristic postcondition extraction."""
        postconds = []
        if ktype in ("action_correction", "remedy"):
            postconds.append("failure_resolved")
        if ktype == "dependency_correction":
            postconds.append("dependency_satisfied")
        if "craft" in content.lower():
            postconds.append("craft_completed")
        if "mine" in content.lower():
            postconds.append("block_mined")
        return postconds

    @staticmethod
    def _infer_safety_contexts(content: str,
                               existing: List[str]) -> List[str]:
        """Infer safety boundaries from knowledge content."""
        safety = list(existing)
        content_lower = content.lower()
        if any(w in content_lower for w in ["lava", "fire"]):
            if "lava_nearby" not in safety:
                safety.append("lava_nearby")
        if any(w in content_lower for w in ["combat", "mob", "zombie", "skeleton"]):
            if "combat" not in safety:
                safety.append("combat")
        if any(w in content_lower for w in ["diamond", "rare", "precious"]):
            if "irreversible_resource_constraint" not in safety:
                safety.append("irreversible_resource_constraint")
        return safety

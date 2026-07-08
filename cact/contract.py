"""
Knowledge Contract: falsifiable behavioral claims for self-evolved knowledge.

C-ACT transforms raw XENON knowledge (ADG dependency corrections,
FAM action corrections) into structured contracts with:
  - preconditions  (must hold before reuse)
  - postconditions (must hold after reuse)
  - non_applicable_contexts (hard safety boundaries)
  - expected_uplift / risk_bound (declared expectations)

Contract verification happens before AND after each reuse decision.
"""

import json, uuid
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict


# ── Knowledge type & level enums ──
KNOWLEDGE_TYPES = ["skill", "remedy", "dependency_correction",
                   "action_correction", "failure_memory"]
KNOWLEDGE_LEVELS = ["strategy", "functional", "atomic",
                    "dependency", "failure_memory"]

# ── Lifecycle states (shared with lifecycle_manager) ──
CANDIDATE    = "candidate"
QUARANTINED  = "quarantined"
PROBATION    = "probation"
CERTIFIED    = "certified"
DEPRECATED   = "deprecated"
DISABLED     = "disabled"

# ── Hard safety boundaries: contexts where reuse is NEVER allowed ──
HARD_SAFETY_CONTEXTS = [
    "lava_nearby", "low_health", "combat",
    "near_cliff", "irreversible_resource_constraint"
]

# ── Risk level definitions ──
RISK_LEVELS = ["low", "medium", "high"]


@dataclass
class KnowledgeContract:
    """A self-evolved knowledge item turned into a falsifiable behavioral claim."""
    knowledge_id: str = field(default_factory=lambda: f"kc_{uuid.uuid4().hex[:12]}")
    source: str = ""              # "XENON_ADG" | "XENON_FAM"
    type: str = ""                # skill | remedy | dependency_correction | action_correction | failure_memory
    level: str = ""               # strategy | functional | atomic | dependency | failure_memory

    # Core knowledge
    gene: str = ""                # One-sentence core
    full_text: str = ""           # Complete behavioral description

    # Contract claims
    claimed_context: Dict[str, str] = field(default_factory=dict)
    preconditions: List[str] = field(default_factory=list)
    postconditions: List[str] = field(default_factory=list)
    expected_uplift: float = 0.05
    risk_bound: float = 0.10

    # Hard safety boundaries
    non_applicable_contexts: List[str] = field(default_factory=list)

    # Provenance
    source_episode: str = ""
    status: str = CANDIDATE

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "KnowledgeContract":
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
    return "atomic" if tier <= 2 else "functional"


# ── Contract verification ──

class ContractChecker:
    """Verify contract conditions before and after knowledge reuse."""

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
    def check_non_applicable(contract: KnowledgeContract,
                             state: Dict) -> Tuple[bool, List[str]]:
        """Check if any non_applicable_context is triggered.
        Returns (safe, [triggered_contexts]).
        True = safe (no non-applicable context triggered).
        """
        triggered = []
        for ctx in contract.non_applicable_contexts:
            if ContractChecker._eval_context_flag(ctx, state):
                triggered.append(ctx)
        return len(triggered) == 0, triggered

    @staticmethod
    def check_context_match(contract: KnowledgeContract,
                            context: Dict) -> bool:
        """Check if current context matches claimed_context."""
        claimed = contract.claimed_context
        if not claimed:
            return True  # No context claim = always applicable
        for key, val in claimed.items():
            if context.get(key) != val and val != "*":
                return False
        return True

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
            "low_health": lambda s: s.get("health", 20) < 5,
            "combat": lambda s: s.get("in_combat", False),
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
            claimed_context={
                "subgoal_type": knowledge.get("subgoal_type", ""),
                "failure_type": knowledge.get("failure_type", ""),
                "task_tier": knowledge.get("task_tier", "stone"),
            },
            preconditions=preconds,
            postconditions=postconds,
            expected_uplift=knowledge.get("expected_uplift", 0.05),
            risk_bound=knowledge.get("risk_bound", 0.10),
            non_applicable_contexts=non_app,
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

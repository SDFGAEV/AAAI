"""
Knowledge Bank Sanitizer — bank-level hygiene for C-ACT.

Before a knowledge item enters the candidate pool, the sanitizer:
  1. Deduplicates near-identical items (same type + task_group, similar embedding)
  2. Merges same-type corrections into generalized rules
  3. Quarantines items missing critical contract fields

This is NOT an admission decision — it only determines what remains available.

Design doc §4: "Bank Sanitizer decides what remains available; C-ACT decides
what is admitted now."
"""

import json
import hashlib
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field


# ── Dedup constants ──
EMBED_SIM_THRESHOLD = 0.85
MAX_DEDUP_GROUP_SIZE = 5

# ── Quarantine: fields that must be present ──
REQUIRED_CONTRACT_FIELDS = [
    "gene", "type", "claimed_context", "preconditions", "non_applicable_contexts"
]

# ── Merge: knowledge types eligible for merging ──
MERGEABLE_TYPES = {"action_correction", "dependency_correction", "remedy"}


@dataclass
class SanitizerAction:
    """Record of what the sanitizer did to a knowledge item."""
    knowledge_id: str
    action: str  # "keep" | "merge" | "quarantine" | "dedup_drop"
    reason: str = ""
    merge_group: str = ""
    merged_into: str = ""


class BankSanitizer:
    """Deduplicate, merge, and quarantine knowledge before admission."""

    def __init__(self, embed_sim_threshold: float = EMBED_SIM_THRESHOLD):
        self.embed_threshold = embed_sim_threshold
        self._action_log: List[SanitizerAction] = []

    def sanitize(self, candidates: List[Dict],
                 existing_ids: set = None) -> Tuple[List[Dict], List[SanitizerAction]]:
        """Run sanitization pipeline on candidate knowledge.

        Args:
            candidates: Raw knowledge dicts from upstream self-evolution pipeline.
                Each must have at minimum: knowledge_id, type, gene, task_group,
                preconditions, non_applicable_contexts, claimed_context.
            existing_ids: Set of already-registered knowledge IDs to avoid
                re-processing previously sanitized items.

        Returns:
            (clean_candidates, actions): Sanitized candidates + action log.
        """
        self._action_log = []
        existing = existing_ids or set()

        # Step 1: Quarantine incomplete items
        complete, quarantined = self._partition_by_completeness(candidates)
        for q in quarantined:
            self._action_log.append(SanitizerAction(
                q.get("knowledge_id", "?"), "quarantine",
                reason="missing required fields"))

        # Step 2: Deduplicate within type + task_group
        deduped = self._deduplicate(complete)

        # Step 3: Merge same-type corrections
        merged = self._merge_same_type(deduped)

        # Step 4: Filter already-registered
        clean = [m for m in merged if m.get("knowledge_id") not in existing]

        # Quarantined items: mark status but do NOT include in clean candidates.
        # They are logged and can be manually reviewed, but are NOT passed to
        # ContractExtractor for active admission.
        for q in quarantined:
            q["_sanitizer_action"] = "quarantine"
            q["status"] = "quarantined"

        return clean, self._action_log

    # ── Completeness check ──
    def _partition_by_completeness(self, candidates: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """Split candidates into complete and quarantined."""
        complete = []
        quarantined = []
        for c in candidates:
            if self._is_complete(c):
                complete.append(c)
            else:
                quarantined.append(c)
        return complete, quarantined

    @staticmethod
    def _is_complete(item: Dict) -> bool:
        """A knowledge item is complete if all required contract fields are present."""
        for field in REQUIRED_CONTRACT_FIELDS:
            val = item.get(field)
            if val is None or (isinstance(val, (list, dict, str)) and len(val) == 0):
                return False
        return True

    # ── Deduplication ──
    def _deduplicate(self, candidates: List[Dict]) -> List[Dict]:
        """Group by (type, task_group) and deduplicate similar items."""
        # Group candidates by (type, task_group)
        by_group: Dict[str, List[Dict]] = {}
        for c in candidates:
            key = f"{c.get('type', '')}|{c.get('task_group', c.get('group', ''))}"
            if key not in by_group:
                by_group[key] = []
            by_group[key].append(c)

        kept = []
        for group_key, items in by_group.items():
            if len(items) <= 1:
                kept.extend(items)
                continue

            # Within each group, find near-duplicate pairs
            dedup_groups = self._cluster_similar(items)
            for dg in dedup_groups:
                # Keep the best one per duplicate cluster
                best = self._select_best(dg)
                kept.append(best)
                for other in dg:
                    if other is not best:
                        self._action_log.append(SanitizerAction(
                            other.get("knowledge_id", "?"), "dedup_drop",
                            reason=f"near_duplicate_of_{best.get('knowledge_id', '?')}"))

        return kept

    def _cluster_similar(self, items: List[Dict]) -> List[List[Dict]]:
        """Cluster items by textual similarity of gene/instruction.

        Uses a simple Jaccard-like token overlap since full embedding comparison
        requires an embedding model. Falls back to exact gene match if no embeddings
        are available.
        """
        if not items:
            return []

        # Build clusters
        clusters = []
        used = set()

        for i, item_a in enumerate(items):
            if i in used:
                continue
            cluster = [item_a]
            used.add(i)

            for j, item_b in enumerate(items):
                if j in used:
                    continue
                if self._is_similar(item_a, item_b):
                    cluster.append(item_b)
                    used.add(j)

            clusters.append(cluster)

        return clusters

    def _is_similar(self, a: Dict, b: Dict) -> bool:
        """Check if two knowledge items are near-duplicates.

        Uses token overlap on gene + instruction text as a lightweight proxy
        for embedding similarity.
        """
        text_a = f"{a.get('gene', '')} {a.get('instruction', '')} {a.get('correction', a.get('full_text', ''))}"
        text_b = f"{b.get('gene', '')} {b.get('instruction', '')} {b.get('correction', b.get('full_text', ''))}"

        tokens_a = set(text_a.lower().split())
        tokens_b = set(text_b.lower().split())

        if not tokens_a or not tokens_b:
            return False

        # Jaccard similarity
        intersection = tokens_a & tokens_b
        union = tokens_a | tokens_b
        sim = len(intersection) / len(union) if union else 0.0
        return sim > self.embed_threshold

    @staticmethod
    def _select_best(candidates: List[Dict]) -> Dict:
        """Select the best candidate from a duplicate cluster.

        Priority: contract completeness > observation count > lower harm > newer.
        """
        def score(c: Dict) -> Tuple[int, int, int, float]:
            preconds = len(c.get("preconditions", []))
            postconds = len(c.get("postconditions", []))
            completeness = 1 if preconds > 0 and postconds > 0 else 0
            obs = c.get("observation_count", c.get("n_observations", 0))
            harm = c.get("harm_count", c.get("harmful_count", 0))
            ts = c.get("timestamp", 0)
            # Higher score = better
            return (completeness, obs, -harm, ts)

        return max(candidates, key=score)

    # ── Merge ──
    def _merge_same_type(self, candidates: List[Dict]) -> List[Dict]:
        """Merge same-type corrections into generalized rules.

        Example: "stone_pickaxe cannot mine diamond" + "wooden_pickaxe cannot mine diamond"
        → "tool_tier < iron cannot mine diamond_ore".
        """
        # Group by (type, task_group)
        by_key: Dict[str, List[Dict]] = {}
        for c in candidates:
            kt = c.get("type", "")
            if kt not in MERGEABLE_TYPES:
                continue
            key = f"{kt}|{c.get('task_group', c.get('group', ''))}"
            if key not in by_key:
                by_key[key] = []
            by_key[key].append(c)

        merged = list(candidates)
        singletons = [c for c in candidates
                      if c.get("type", "") not in MERGEABLE_TYPES]

        for key, items in by_key.items():
            if len(items) < 2:
                continue
            # Check if all items are same type with similar structure
            if self._can_merge(items):
                generalized = self._generalize(items)
                if generalized:
                    # Remove originals, add generalized version
                    for item in items:
                        if item in merged:
                            merged.remove(item)
                    merged.append(generalized)
                    for item in items:
                        self._action_log.append(SanitizerAction(
                            item.get("knowledge_id", "?"), "merge",
                            merge_group=key,
                            merged_into=generalized.get("knowledge_id", "?")))

        return merged

    @staticmethod
    def _can_merge(items: List[Dict]) -> bool:
        """Check if items are eligible for merging: same type, same failure pattern."""
        if len(items) < 2:
            return False
        types = {item.get("type") for item in items}
        if len(types) > 1:
            return False
        # Check they target similar preconditions
        precond_patterns = set()
        for item in items:
            for pc in item.get("preconditions", []):
                # Extract the structure: e.g., "target_block == diamond_ore" → "target_block"
                if "==" in pc:
                    precond_patterns.add(pc.split("==")[0].strip())
                elif "not in" in pc:
                    precond_patterns.add(pc.split("not in")[0].strip())
                else:
                    precond_patterns.add(pc.strip())
        return len(precond_patterns) <= 3  # Reasonably similar

    @staticmethod
    def _generalize(items: List[Dict]) -> Optional[Dict]:
        """Create a generalized rule from a group of similar corrections.

        This is a heuristic: take the item with the most evidence as the base,
        widen its preconditions to cover all merged items.
        """
        if not items:
            return None

        base = max(items, key=lambda c: c.get("observation_count",
                                               c.get("n_observations", 0)))
        generalized = dict(base)
        generalized["knowledge_id"] = f"{base.get('knowledge_id', 'merge')}_generalized"
        generalized["gene"] = f"[GENERALIZED] {base.get('gene', '')} (+{len(items)-1} similar)"
        generalized["_merge_sources"] = [i.get("knowledge_id") for i in items]

        # Widen the scope to cover all merged items
        groups = {i.get("task_group", i.get("group", "")) for i in items if i.get("task_group")}
        if len(groups) > 1:
            generalized["task_group"] = "|".join(sorted(groups))

        return generalized

    @property
    def action_log(self) -> List[SanitizerAction]:
        return self._action_log

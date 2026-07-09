"""
ContextBucket: adaptive hierarchical context encoding (C-ACT)

Design: high-dimensional context → generalized bucket string for statistical sharing.
Supports adaptive split/merge to balance granularity vs sample efficiency.

Split criterion: within-bucket ECE > threshold + n >= n_min
Merge criterion: KL divergence between buckets < threshold
"""

import math
from typing import Dict, List, Tuple, Optional


class ContextBucket:
    def __init__(self, ece_threshold: float = 0.15, merge_threshold: float = 0.05,
                 n_split_min: int = 10, n_merge_min: int = 3):
        self.ece_threshold = ece_threshold
        self.merge_threshold = merge_threshold
        self.n_split_min = n_split_min
        self.n_merge_min = n_merge_min
        self._bucket_stats: Dict[str, Dict] = {}
        self._split_fields = ["subgoal_type", "failure_type", "inventory_sig",
                              "risk_level", "task_tier", "biome"]

    def encode(self, knowledge_type: str = "skill", subgoal_type: str = "",
               failure_type: str = "", inventory_sig: str = "",
               risk_level: str = "medium", task_tier: str = "stone",
               biome: str = "forest") -> str:
        """Encode context into bucket key."""
        fields = [knowledge_type]
        if "subgoal_type" in self._split_fields:
            fields.append(subgoal_type or "craft")
        if "failure_type" in self._split_fields:
            fields.append(failure_type or "none")
        if "inventory_sig" in self._split_fields:
            fields.append(inventory_sig or "basic")
        if "risk_level" in self._split_fields:
            fields.append(risk_level)
        if "task_tier" in self._split_fields:
            fields.append(task_tier)
        return "|".join(fields)

    def maybe_split(self, bucket_key: str, confidence_list: List[float],
                    outcome_list: List[float]) -> Optional[str]:
        """Split a bucket if within-bucket ECE is too high. Returns new split field or None."""
        if len(confidence_list) < self.n_split_min:
            return None
        ece = self._compute_ece(confidence_list, outcome_list)
        if ece < self.ece_threshold:
            return None
        best_field = None; best_reduction = 0.0
        parts = bucket_key.split("|")
        for i, field in enumerate(self._split_fields):
            if i >= len(parts): continue
            val_high = [o for j, o in enumerate(outcome_list)
                        if self._field_high(parts, i, j, confidence_list)]
            val_low = [o for j, o in enumerate(outcome_list)
                       if not self._field_high(parts, i, j, confidence_list)]
            if len(val_high) < 3 or len(val_low) < 3: continue
            ece_split = (self._compute_ece(
                [c for j, c in enumerate(confidence_list) if self._field_high(parts, i, j, confidence_list)], val_high) +
                         self._compute_ece(
                [c for j, c in enumerate(confidence_list) if not self._field_high(parts, i, j, confidence_list)], val_low)) / 2
            reduction = ece - ece_split
            if reduction > best_reduction:
                best_reduction = reduction; best_field = field
        return best_field

    def _field_high(self, parts, idx, j, conf_list):
        """Check if the j-th outcome belongs to the 'high' subgroup for field idx.
        The field value is extracted from the bucket key parts list.
        Falls back to confidence check if field value can't be parsed."""
        if idx >= len(parts):
            return j < len(conf_list) and conf_list[j] > 0.5
        field_val = parts[idx]
        try:
            num = float(field_val.split("_")[-1]) if field_val.split("_")[-1].replace('.','',1).replace('-','',1).isdigit() else None
            if num is not None:
                return num > 0.5
        except (ValueError, IndexError):
            pass
        return j < len(conf_list) and conf_list[j] > 0.5

    def maybe_merge(self, buckets: List[str],
                    bucket_stats: Dict[str, Tuple[float, float, int]]) -> List[List[str]]:
        """Merge low-sample similar buckets. Returns list of merged bucket groups."""
        if len(buckets) < 2: return []
        merges = []; used = set()
        for i, b1 in enumerate(buckets):
            if b1 in used: continue
            group = [b1]; used.add(b1)
            a1, b1_s, n1 = bucket_stats.get(b1, (1.0, 1.0, 0))
            if n1 >= self.n_merge_min * 2: continue
            m1 = a1 / max(a1 + b1_s, 1e-8) if a1 + b1_s > 0 else 0.5
            for b2 in buckets[i+1:]:
                if b2 in used: continue
                a2, b2_s, n2 = bucket_stats.get(b2, (1.0, 1.0, 0))
                if n2 >= self.n_merge_min * 2: continue
                m2 = a2 / max(a2 + b2_s, 1e-8) if a2 + b2_s > 0 else 0.5
                kl = self._approx_kl(m1, m2)
                if kl < self.merge_threshold:
                    group.append(b2); used.add(b2)
            if len(group) > 1: merges.append(group)
        return merges

    @staticmethod
    def _compute_ece(conf: List[float], out: List[float], n_bins: int = 5) -> float:
        if len(conf) < n_bins: return 0.0
        pairs = sorted(zip(conf, out)); n = len(pairs)
        e = 0.0
        for i in range(n_bins):
            lo, hi = i * n // n_bins, (i + 1) * n // n_bins
            if hi > lo:
                c_bin = [p[0] for p in pairs[lo:hi]]
                o_bin = [p[1] for p in pairs[lo:hi]]
                e += abs(sum(c_bin) / len(c_bin) - sum(o_bin) / len(o_bin)) * (hi - lo) / n
        return e

    @staticmethod
    def _approx_kl(m1: float, m2: float) -> float:
        m1 = max(0.001, min(0.999, m1)); m2 = max(0.001, min(0.999, m2))
        return m1 * math.log(m1 / m2) + (1 - m1) * math.log((1 - m1) / (1 - m2))

    # ── Adaptive maintenance (doc §6.3) ──

    def accumulate(self, bucket_key: str, confidence: float, outcome: float):
        """Record one observation for a bucket. Used to build stats for split/merge."""
        if bucket_key not in self._bucket_stats:
            self._bucket_stats[bucket_key] = {"conf": [], "out": []}
        self._bucket_stats[bucket_key]["conf"].append(confidence)
        self._bucket_stats[bucket_key]["out"].append(outcome)

    def maintain(self, allowed: bool = True) -> Dict:
        """Run adaptive split/merge on all buckets.

        Args:
            allowed: If False (E3 frozen), no maintenance is performed.

        Returns:
            {"splits": [...], "merges": [[...]], ...}
        """
        result = {"splits": [], "merges": [], "actions": []}
        if not allowed or not self._bucket_stats:
            return result

        # Split: check each bucket for high ECE
        for key, stats in list(self._bucket_stats.items()):
            conf = stats["conf"]
            out = stats["out"]
            if len(conf) < self.n_split_min:
                continue
            ece = self._compute_ece(conf, out)
            if ece > self.ece_threshold:
                new_field = self.maybe_split(key, conf, out)
                if new_field:
                    result["splits"].append({"bucket": key, "new_field": new_field,
                                              "ece": round(ece, 3), "n": len(conf)})
                    result["actions"].append(f"split:{key}→+{new_field}")

        # Merge: check for low-support similar buckets
        buckets = list(self._bucket_stats.keys())
        if len(buckets) >= 2:
            bucket_stats_tuples = {}
            for b in buckets:
                s = self._bucket_stats[b]
                total = max(len(s["conf"]), 1)
                succ = sum(s["out"])
                alpha = 1.0 + succ
                beta_param = 1.0 + total - succ
                bucket_stats_tuples[b] = (alpha, beta_param, total)
            merge_groups = self.maybe_merge(buckets, bucket_stats_tuples)
            for group in merge_groups:
                result["merges"].append(group)
                result["actions"].append(f"merge:{'|'.join(group)}")

        return result

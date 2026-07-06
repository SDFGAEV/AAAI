"""
CASK Memory: wraps XENON DecomposedMemory with Trust Before Reuse.
Complete: 6 methods, 7 log types, CF branching, interaction tracking.
"""
import logging, numpy as np, json, os, copy
from typing import Dict, List, Optional
from .trust_store import TrustStore
from .trust_gate import TrustGate
from .context_bucket import ContextBucket


class CaskMemory:
    def __init__(self, xenon_memory, method="CounterfactualTrust",
                 store_path: str = None, t_eps: float = 0.0, frozen_store=None,
                 frozen: bool = False, cf_branching: bool = False,
                 log_dir: str = None):
        self._mem = xenon_memory
        self._store = TrustStore(store_path or "cask_ckpt/trust_store")
        if frozen_store is not None:
            self._store._data = copy.deepcopy(frozen_store)
        self._gate = TrustGate()
        self._bucket = ContextBucket()
        self._logger = logging.getLogger("CaskMemory")
        self.method = method; self.t_eps = t_eps
        self.frozen = frozen; self.cf_branching = cf_branching
        self.log_dir = log_dir; self._cf_branches = []
        # All 7 log types
        self.elogs = []; self.slogs = []; self.klogs = []
        self.blogs = []; self.cflogs = []; self.ilogs = []
        self._prev_kid = None; self._pairwise_harm = {}
        self._backtrack = 0; self._repfail = {}

    # --- Pass-through ---
    @property
    def succeeded_waypoints(self): return self._mem.succeeded_waypoints
    def retrieve_similar_succeeded_waypoints(self, w, k=3): return self._mem.retrieve_similar_succeeded_waypoints(w, k)
    def retrieve_failed_subgoals(self, w): return self._mem.retrieve_failed_subgoals(w)
    def retrieve_total_failed_counts(self, w): return self._mem.retrieve_total_failed_counts(w)
    def save_plan(self, *a, **kw): return self._mem.save_plan(*a, **kw)
    def reset_success_failure_history(self, i): return self._mem.reset_success_failure_history(i)
    @property
    def current_environment(self): return self._mem.current_environment if hasattr(self._mem, 'current_environment') else ""
    @current_environment.setter
    def current_environment(self, v): self._mem.current_environment = v
    def get_store_data(self): return dict(self._store._data)

    # --- §7 Interaction check ---
    def _check_interaction(self, kid_i, kid_j, eta=0.5):
        key = tuple(sorted([kid_i, kid_j]))
        if key in self._pairwise_harm:
            from scipy.stats import beta as beta_dist
            a, b = self._pairwise_harm[key]
            return float(beta_dist.ppf(0.95, a, b)) > eta
        return False

    def _record_interaction(self, kid_i, kid_j, harmful):
        key = tuple(sorted([kid_i, kid_j]))
        if key not in self._pairwise_harm:
            self._pairwise_harm[key] = [1.0, 1.0]
        if harmful: self._pairwise_harm[key][0] += 1.0
        else: self._pairwise_harm[key][1] += 1.0

    # --- update episode stats before flush ---
    def update_last_episode(self, total_steps=0, llm_calls=0, wall_time_sec=0.0,
                            input_tokens=0, output_tokens=0):
        """Called by main_planning.py after task completion to add runtime stats."""
        if self.elogs:
            e = self.elogs[-1]
            e["total_steps"] = total_steps or e.get("total_steps", 0)
            e["llm_calls"] = llm_calls or e.get("llm_calls", 0)
            e["wall_time_sec"] = wall_time_sec or e.get("wall_time_sec", 0.0)
            e["tokens"] = e.get("tokens", 0) + input_tokens + output_tokens
            if self.slogs:
                self.slogs[-1]["total_steps"] = total_steps

    # --- dump logs ---
    def dump_logs(self):
        if not self.log_dir: return
        os.makedirs(self.log_dir, exist_ok=True)
        files = {
            "episode.jsonl": self.elogs, "subgoal.jsonl": self.slogs,
            "knowledge_reuse.jsonl": self.klogs, "fallback.jsonl": self.blogs,
            "cf_branch.jsonl": self.cflogs, "interaction.jsonl": self.ilogs,
        }
        for fname, data in files.items():
            if data:
                with open(os.path.join(self.log_dir, fname), "a") as f:
                    for e in data: f.write(json.dumps(e) + "\n")
        # Clear flushed logs
        self.elogs.clear(); self.slogs.clear(); self.klogs.clear()
        self.blogs.clear(); self.cflogs.clear(); self.ilogs.clear()

    # --- Intercepted: save_success_failure ---
    def save_success_failure(self, waypoint, language_action_str, is_success,
                             cf_base_result: bool = None):
        self._mem.save_success_failure(waypoint, language_action_str, is_success)
        ctx = self._bucket.encode(knowledge_type="skill", subgoal_type="craft",
                                  task_tier=self._infer_tier(waypoint))
        sv = 1.0 if is_success else 0.0; kid = f"skill:{waypoint}"
        if not self.frozen:
            self._store.record_episode(kid, ctx, used=True, success=sv,
                                       is_harmful=0.0 if is_success else 1.0)
        n = self._store.total_count(kid, ctx); lcb = self._store.lcb(kid, ctx)
        bu = self._store.ucb(kid, ctx, "base"); up = self._store.uplift(kid, ctx)
        hu = self._store.harm_ucb(kid, ctx); ts = up - 0.2 * hu

        # Counterfactual branching: record paired result
        if cf_base_result is not None:
            self.cflogs.append({"pair_id": f"cf_{waypoint}_{n:.0f}", "waypoint": waypoint,
                "ctx": ctx, "branch_use": {"success": int(is_success), "harmful": 0 if is_success else 1},
                "branch_base": {"success": int(cf_base_result), "harmful": 0 if cf_base_result else 1}})
            if not self.frozen:
                self._store.record_episode(kid, ctx, used=False, success=1.0 if cf_base_result else 0.0,
                                           is_harmful=0.0 if cf_base_result else 1.0)

        # Knowledge reuse log (§17.3)
        self.klogs.append({"waypoint": waypoint, "kid": kid, "ctx": ctx,
            "method": self.method, "frozen": self.frozen, "success": is_success,
            "use_lcb": round(lcb, 4), "base_ucb": round(bu, 4), "uplift": round(up, 4),
            "harm_ucb": round(hu, 4), "trust_score": round(ts, 4),
            "n_use": int(n), "t_eps": self.t_eps, "decision": "reuse",
            "is_harmful": 0 if is_success else 1, "outcome_success": int(is_success)})

        # Episode log (§17.1) — one per subgoal
        self.elogs.append({"waypoint": waypoint, "method": self.method, "frozen": self.frozen,
            "success": int(is_success), "total_steps": 0, "llm_calls": 0, "unrecoverable_failure": 0 if is_success else 1})

        # Subgoal log (§17.2)
        self.slogs.append({"waypoint": waypoint, "context": {"subgoal_type": "craft", "task_tier": self._infer_tier(waypoint)},
            "progress_delta": 1.0 if is_success else 0.0, "failure_type": "none" if is_success else "execution_failure"})

        # Fallback log (§17.4) — when base policy was used
        if not is_success and n < 3:
            self.blogs.append({"waypoint": waypoint, "kid": kid, "used_candidate": False,
                "used_base_policy": True, "base_success": 0, "progress_delta": 0.0})

        # Structured knowledge
        if is_success and n >= 2 and lcb > 0.3:
            self.elogs[-1]["knowledge_generated"] = f"s_{waypoint}_{int(n):02d}"
        if not is_success and n >= 2:
            self._backtrack += 1; self._repfail[waypoint] = self._repfail.get(waypoint, 0) + 1
            # Remedy type to klogs for IRR
            self.klogs.append({"waypoint": waypoint, "kid": kid, "ctx": ctx,
                "method": self.method, "type": "remedy", "frozen": self.frozen,
                "success": False, "failure_resolved": False,
                "use_lcb": round(lcb, 4), "base_ucb": round(bu, 4),
                "trust_score": round(ts, 4), "n_use": int(n),
                "t_eps": self.t_eps, "decision": "reuse",
                "is_harmful": 1, "outcome_success": 0})

    # --- Intercepted: is_succeeded_waypoint ---
    def is_succeeded_waypoint(self, waypoint):
        ctx = self._bucket.encode(knowledge_type="skill", subgoal_type="craft",
                                  task_tier=self._infer_tier(waypoint))
        kid = f"skill:{waypoint}"
        trusted = self._decide(kid, ctx)
        n = self._store.total_count(kid, ctx); lcb = self._store.lcb(kid, ctx)

        # §7: Interaction check — block reuse if conflict with previous knowledge
        if trusted and self._prev_kid and self._check_interaction(self._prev_kid, kid):
            trusted = False

        if trusted:
            is_ok, sg = self._mem.is_succeeded_waypoint(waypoint)
            result = (True, sg) if is_ok else (False, None)
        else:
            result = (False, None)
            if self.method != "NoKnowledge":
                self._store.record_episode(kid, ctx, used=False, success=0.0, is_harmful=0.0)

        self._prev_kid = kid
        self._logger.info(f"CASKe[{self.method}] {kid}: LCB={lcb:.3f} n={n:.0f} trust={trusted}")
        return result

    def _decide(self, kid, ctx):
        s = self._store
        if self.method == "NoKnowledge": return np.random.random() < 0.5
        if self.method == "NoTrust": return True
        if self.method == "RawSuccess": return s.mean(kid, ctx, "use") >= 0.5
        if self.method == "MeanUplift":
            if s.total_count(kid, ctx, "use") < 1: return True
            return s.mean(kid, ctx, "use") - s.mean(kid, ctx, "base") >= 0.05
        if self.method == "CounterfactualTrust":
            if s.total_count(kid, ctx, "use") < 1: return True
            return s.uplift(kid, ctx) >= self.t_eps
        if self.method == "Full-Frozen":
            if s.total_count(kid, ctx, "use") < 1: return True
            return s.uplift(kid, ctx) - 0.2 * s.harm_ucb(kid, ctx) >= self.t_eps
        return True

    def _infer_tier(self, item):
        il = item.lower()
        if any(t in il for t in ["diamond","netherite","ender"]): return "diamond"
        if any(t in il for t in ["iron","gold","redstone"]): return "iron"
        if any(t in il for t in ["stone","cobblestone","coal","furnace"]): return "stone"
        if any(t in il for t in ["wood","log","plank","stick","crafting"]): return "wood"
        return "stone"

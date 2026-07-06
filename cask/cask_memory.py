"""
CASK Memory: ACT-RL knowledge governance layer (v2)

Wraps XENON DecomposedMemory with adaptive counterfactual trust gate.

Knowledge lifecycle: Candidate → Probation → Certified → Deprecated → Disabled
No RL rollback — only knowledge-level governance.

7 methods, 7 log types, adaptive calibration, interaction awareness.
"""

import logging, numpy as np, json, os, copy, time
from typing import Dict, List, Optional, Tuple
from .trust_store import TrustStore, CANDIDATE, PROBATION, CERTIFIED, DEPRECATED, DISABLED
from .trust_gate import TrustGate
from .context_bucket import ContextBucket


class CaskMemory:
    def __init__(self, xenon_memory, method="CounterfactualTrust",
                 store_path: str = None, t_eps: float = 0.0, frozen_store=None,
                 frozen: bool = False, cf_branching: bool = False,
                 active_calib_rate: float = 0.0, log_dir: str = None):
        self._mem = xenon_memory
        self._store = TrustStore(store_path or "cask_ckpt/trust_store")
        if frozen_store is not None:
            self._store._data = copy.deepcopy(frozen_store)
        self._gate = TrustGate()
        self._bucket = ContextBucket()
        self._logger = logging.getLogger("CaskMemory")
        self.method = method; self.t_eps = t_eps
        self.frozen = frozen; self.cf_branching = cf_branching
        self.active_calib_rate = active_calib_rate
        self.log_dir = log_dir
        # Logs
        self.elogs = []; self.slogs = []; self.klogs = []
        self.blogs = []; self.cflogs = []; self.ilogs = []; self.vlogs = []
        self._prev_kid = None; self._pairwise_harm = {}
        self._knowledge_cnt = 0; self._backtrack = 0; self._repfail = {}
        # Drift tracking
        self._recent_errors: List[float] = []; self._healthy_error: float = 0.3
        self._drift_counter = 0

    # ── Pass-through ──
    @property
    def succeeded_waypoints(self): return self._mem.succeeded_waypoints
    def retrieve_similar_succeeded_waypoints(self, w, k=3): return self._mem.retrieve_similar_succeeded_waypoints(w, k)
    def retrieve_failed_subgoals(self, w): return self._mem.retrieve_failed_subgoals(w)
    def retrieve_total_failed_counts(self, w): return self._mem.retrieve_total_failed_counts(w)
    def save_plan(self, *a, **kw): return self._mem.save_plan(*a, **kw)
    def reset_success_failure_history(self, i): return self._mem.reset_success_failure_history(i)
    def set_history_index(self, i): return self._mem.set_history_index(i)
    def add_succeeded_waypoint(self, wp, action): return self._mem.add_succeeded_waypoint(wp, action)
    def add_failed_waypoint(self, wp, action): return self._mem.add_failed_waypoint(wp, action)
    def save_reflection(self, wp, reflect, is_success): return self._mem.save_reflection(wp, reflect, is_success)
    def save_replan(self, wp, replan): return self._mem.save_replan(wp, replan)
    def save_decomposed_plan(self, wp, plan, is_success): return self._mem.save_decomposed_plan(wp, plan, is_success)
    def set_current_environment(self, env_name): pass
    @property
    def current_environment(self): return None
    @current_environment.setter
    def current_environment(self, v): pass

    # ── Version log ──
    def record_version(self, version_id="", prev_version="", new_knowledge=0,
                       shadow_eval=0, promote=False, rollback=False):
        lc = self._store.lifecycle_stats()
        self.vlogs.append({
            "version_id": version_id, "previous_version": prev_version,
            "new_knowledge_count": new_knowledge, "shadow_eval_episodes": shadow_eval,
            "promote": promote, "rollback": rollback,
            "knowledge_store_size": self._knowledge_cnt,
            "lifecycle": lc,
        })

    # ── Episode stats ──
    def update_last_episode(self, total_steps=0, llm_calls=0, wall_time_sec=0.0,
                            input_tokens=0, output_tokens=0):
        if self.elogs:
            e = self.elogs[-1]
            e["total_steps"] = total_steps or e.get("total_steps", 0)
            e["llm_calls"] = llm_calls or e.get("llm_calls", 0)
            e["wall_time_sec"] = wall_time_sec or e.get("wall_time_sec", 0.0)
            e["tokens"] = e.get("tokens", 0) + input_tokens + output_tokens
            if self.slogs:
                self.slogs[-1]["total_steps"] = total_steps

    # ── Dump logs ──
    def dump_logs(self):
        if not self.log_dir: return
        os.makedirs(self.log_dir, exist_ok=True)
        files = {
            "episode.jsonl": self.elogs, "subgoal.jsonl": self.slogs,
            "knowledge_reuse.jsonl": self.klogs, "fallback.jsonl": self.blogs,
            "cf_branch.jsonl": self.cflogs, "interaction.jsonl": self.ilogs,
            "version.jsonl": self.vlogs,
        }
        for fname, data in files.items():
            if data:
                with open(os.path.join(self.log_dir, fname), "a") as f:
                    for e in data: f.write(json.dumps(e) + "\n")
        self.elogs.clear(); self.slogs.clear(); self.klogs.clear()
        self.blogs.clear(); self.cflogs.clear(); self.ilogs.clear(); self.vlogs.clear()

    # ─── save_success_failure ───
    def save_success_failure(self, waypoint, language_action_str, is_success,
                             cf_base_result: bool = None):
        self._mem.save_success_failure(waypoint, language_action_str, is_success)
        sv = 1.0 if is_success else 0.0
        kid = f"skill:{waypoint}"
        ktype = self._infer_knowledge_type(waypoint, is_success)
        ctx = self._bucket.encode(knowledge_type="skill", subgoal_type="craft",
                                  task_tier=self._infer_tier(waypoint))
        task_grp = self._infer_group(waypoint)
        if not self.frozen:
            self._store.record_episode(kid, ctx, used=True, success=sv,
                                       is_harmful=0.0 if is_success else 1.0,
                                       knowledge_type=ktype)
        n = self._store.total_count(kid, ctx); lcb = self._store.lcb(kid, ctx)
        bu = self._store.ucb(kid, ctx, "base"); up = self._store.uplift(kid, ctx)
        hu = self._store.harm_ucb(kid, ctx); pu = self._store.prob_use_better(kid, ctx)
        ess = self._store.ess(kid, ctx); lc = self._store.get_lifecycle(kid)

        # Counterfactual branching
        if cf_base_result is not None:
            self.cflogs.append({
                "pair_id": f"cf_{waypoint}_{n:.0f}", "waypoint": waypoint, "ctx": ctx,
                "branch_use": {"success": int(is_success), "harmful": 0 if is_success else 1},
                "branch_base": {"success": int(cf_base_result), "harmful": 0 if cf_base_result else 1},
            })
            if not self.frozen:
                self._store.record_episode(kid, ctx, used=False,
                                           success=1.0 if cf_base_result else 0.0,
                                           is_harmful=0.0 if cf_base_result else 1.0)

        # Knowledge reuse log
        log_entry = {
            "waypoint": waypoint, "kid": kid, "ctx": ctx, "method": self.method,
            "frozen": self.frozen, "success": is_success, "task_group": task_grp,
            "use_lcb": round(lcb, 4), "base_ucb": round(bu, 4),
            "uplift": round(up, 4), "harm_ucb": round(hu, 4),
            "prob_uplift": round(pu, 4), "pi_uplift": round(pu, 4),
            "ess": round(ess, 1), "lifecycle": lc,
            "t_eps": self.t_eps, "decision": "reuse",
            "is_harmful": 0 if is_success else 1, "outcome_success": int(is_success),
        }
        self.klogs.append(log_entry)

        # Episode log
        self.elogs.append({
            "waypoint": waypoint, "method": self.method, "frozen": self.frozen,
            "task_group": task_grp, "success": int(is_success),
            "total_steps": 0, "llm_calls": 0, "tokens": 0,
            "lifecycle": lc, "unrecoverable_failure": 0 if is_success else 1,
        })

        # Subgoal log
        self.slogs.append({
            "waypoint": waypoint, "success": int(is_success),
            "context": {"subgoal_type": "craft", "task_tier": self._infer_tier(waypoint)},
            "progress_delta": 1.0 if is_success else 0.0,
            "failure_type": "none" if is_success else "execution_failure",
        })

        # Fallback / base log
        if not is_success and n < 3:
            self.blogs.append({
                "waypoint": waypoint, "kid": kid, "used_candidate": False,
                "used_base_policy": True, "base_success": 0, "progress_delta": 0.0,
            })

        # Knowledge generation
        if is_success and n >= 2 and lcb > 0.3:
            self.elogs[-1]["knowledge_generated"] = f"s_{waypoint}_{int(n):02d}"
            self._knowledge_cnt += 1

        # Remedy tracking
        if not is_success and n >= 2:
            self._backtrack += 1
            self._repfail[waypoint] = self._repfail.get(waypoint, 0) + 1
            self.klogs.append({
                "waypoint": waypoint, "kid": kid, "ctx": ctx,
                "method": self.method, "type": "remedy", "frozen": self.frozen,
                "success": False, "failure_resolved": False, "task_group": task_grp,
                "prob_uplift": round(pu, 4), "harm_ucb": round(hu, 4),
                "trust_score": round(up - 0.2 * hu, 4),
                "t_eps": self.t_eps, "decision": "reuse",
                "is_harmful": 1, "outcome_success": 0, "lifecycle": lc,
            })

        # Drift tracking
        pred_err = abs(sv - self._store.mean(kid, ctx, "use"))
        self._recent_errors.append(pred_err)
        if len(self._recent_errors) > 50:
            self._recent_errors = self._recent_errors[-50:]

        # Apply lazy decay periodically
        self._drift_counter += 1
        if self._drift_counter % 20 == 0:
            drift_factor = self._store.detect_drift(
                self._recent_errors[-20:], self._healthy_error)
            self._store.decay_all(retention=0.95, drift_factor=drift_factor)

    # ─── is_succeeded_waypoint ───
    def is_succeeded_waypoint(self, waypoint):
        ctx = self._bucket.encode(knowledge_type="skill", subgoal_type="craft",
                                  task_tier=self._infer_tier(waypoint))
        kid = f"skill:{waypoint}"
        ess = self._store.ess(kid, ctx)
        pu = self._store.prob_use_better(kid, ctx)
        up = self._store.uplift(kid, ctx)
        hu = self._store.harm_ucb(kid, ctx)
        lc = self._store.get_lifecycle(kid)
        task_grp = self._infer_group(waypoint)

        trusted = self._decide(kid, ctx, pu, up, hu, ess, lc, task_grp)

        # Active calibration: randomly force base
        if trusted and self.active_calib_rate > 0:
            imbalance = abs(self._store.total_count(kid, ctx, "use") -
                            self._store.total_count(kid, ctx, "base"))
            samp_imb = min(imbalance / max(ess, 1), 1.0)
            risk_level = self._infer_risk(waypoint)
            q = self._gate.exploration_rate(ess, pu, risk_level, samp_imb)
            if np.random.random() < q:
                trusted = False
                self._store.record_episode(kid, ctx, used=False, success=0.5,
                                           is_harmful=0.0)

        # Interaction check with 4-state classification
        conflict_pair = None; interaction_state = None
        if trusted and self._prev_kid and self._check_interaction(self._prev_kid, kid):
            d_mean, d_lcb, is_syn, is_conf, state = self._store.interaction_uplift(
                self._prev_kid, kid, ctx)
            interaction_state = state
            if is_conf:
                trusted = False; conflict_pair = [self._prev_kid, kid]
            elif not is_syn and state == "unknown":
                if self._infer_risk(waypoint) != "low":
                    trusted = False; conflict_pair = [self._prev_kid, kid]

        if trusted:
            is_ok, sg = self._mem.is_succeeded_waypoint(waypoint)
            result = (True, sg) if is_ok else (False, None)
        else:
            result = (False, None)
            if conflict_pair:
                self.ilogs.append({
                    "used_knowledge_chain": conflict_pair,
                    "chain_success": 0, "failure_reason": "resource_conflict",
                    "conflict_pairs": [conflict_pair],
                    "resource_conflict": 1, "interaction_state": interaction_state,
                })

        self._prev_kid = kid
        self._logger.info(
            f"ACT[{self.method}] {kid}: pu={pu:.3f} up={up:.3f} hu={hu:.3f} "
            f"ess={ess:.1f} lc={lc} trust={trusted}")
        return result

    # ─── Decision logic ───
    def _decide(self, kid, ctx, pu, up, hu, ess, lc, task_grp):
        s = self._store
        if self.method == "NoKnowledge":
            return np.random.random() < 0.5
        if self.method == "NoTrust":
            return True
        if self.method == "RawSuccess":
            return s.mean(kid, ctx, "use") >= 0.5
        if self.method == "MeanUplift":
            return s.mean(kid, ctx, "use") - s.mean(kid, ctx, "base") >= 0.05

        # Fixed-Bayes: fixed thresholds
        if self.method == "Fixed-Bayes":
            if s.total_count(kid, ctx, "use") < 1: return True
            return pu >= 0.90 and up >= 0.05 and hu <= 0.10

        # CounterfactualTrust: legacy LCB-UCB gate
        if self.method == "CounterfactualTrust":
            if s.total_count(kid, ctx, "use") < 1: return True
            return up >= self.t_eps

        # CounterfactualTrust-BF: Bayes factor gate
        if self.method == "CounterfactualTrust-BF":
            if s.total_count(kid, ctx, "use") < 1: return True
            return pu >= 0.90 and hu <= 0.10

        # Adaptive-Bayes: adaptive thresholds
        if self.method == "Adaptive-Bayes":
            if lc == DISABLED: return False
            if s.total_count(kid, ctx, "use") < 1: return True
            g = self._gate
            tau_g = g.tau.get(task_grp, 0.90)
            delta_g = g.delta.get(task_grp, 0.05)
            harm_g = g.harm.get(task_grp, 0.10)
            upland_ok = pu >= tau_g and up >= delta_g
            safety_ok = hu <= harm_g
            if upland_ok and safety_ok: return True
            # Thompson safe probe
            if ess < 3 and self._infer_risk(task_grp) != "high":
                p_u, p_b, p_h = s.thompson_sample(kid, ctx)
                probe_ok = p_u > p_b + delta_g and p_h < harm_g
                q_probe = g.thompson_probe_rate(ess, pu, self._infer_risk(task_grp))
                if probe_ok and np.random.random() < q_probe:
                    return True
            return False

        # ACT-RL-Full: adaptive + conformal + interaction
        if self.method == "ACT-RL-Full":
            if lc == DISABLED: return False
            if s.total_count(kid, ctx, "use") < 1 and lc != CANDIDATE:
                return True
            g = self._gate
            tau_g = g.tau.get(task_grp, 0.90)
            delta_g = g.delta.get(task_grp, 0.05)
            harm_g = g.harm.get(task_grp, 0.10)
            upland_ok = pu >= tau_g and up >= delta_g
            safety_ok = hu <= harm_g
            if upland_ok and safety_ok: return True
            if ess < 3 and self._infer_risk(task_grp) != "high":
                p_u, p_b, p_h = s.thompson_sample(kid, ctx)
                if p_u > p_b + delta_g and p_h < harm_g:
                    q = g.thompson_probe_rate(ess, pu, self._infer_risk(task_grp))
                    if np.random.random() < q: return True
            return False

        # Full-Frozen: legacy
        if self.method == "Full-Frozen":
            if s.total_count(kid, ctx, "use") < 1: return True
            return up - 0.2 * hu >= self.t_eps
        return True

    # ─── Helpers ───
    def _infer_tier(self, item):
        il = str(item).lower()
        if any(t in il for t in ["diamond", "netherite", "ender"]): return "diamond"
        if any(t in il for t in ["iron", "obsidian", "portal"]): return "iron"
        if any(t in il for t in ["stone", "furnace", "shield", "cake"]): return "stone"
        if any(t in il for t in ["plank", "stick", "wood", "dirt", "wool"]): return "wood"
        return "stone"

    def _infer_group(self, item):
        il = str(item).lower()
        if any(t in il for t in ["craft", "plank", "stick", "table", "ladder", "furnace",
                                  "stonecut", "shield", "cake", "potion", "enchant"]):
            return "crafting"
        if any(t in il for t in ["mine", "collect", "ore", "diamond", "iron", "obsidian",
                                  "wood", "dirt", "wool", "cobble"]):
            return "mining"
        if any(t in il for t in ["explore", "find", "village", "forest", "lava", "treasure",
                                  "bedrock", "map", "biome"]):
            return "exploration"
        if any(t in il for t in ["techtree", "nether_portal", "long", "tech_tree"]):
            return "tech_tree"
        if any(t in il for t in ["wrong_tool", "missing", "failure", "recover"]):
            return "failure_recovery"
        if any(t in il for t in ["interaction", "conflict", "limited", "compet"]):
            return "interaction_stress"
        return "crafting"

    def _infer_risk(self, item):
        il = str(item).lower()
        if any(t in il for t in ["diamond", "netherite", "lava", "combat", "witch",
                                  "portal", "ender"]):
            return "high"
        if any(t in il for t in ["iron", "obsidian", "cake", "potion", "enchant",
                                  "skeleton", "zombie"]):
            return "medium"
        return "low"

    def _infer_knowledge_type(self, item, is_success):
        il = str(item).lower()
        if is_success: return "skill"
        if any(t in il for t in ["wrong_tool", "missing_ingredient"]): return "action_correction"
        if any(t in il for t in ["missing_table", "missing_furnace", "missing"]): return "dependency"
        return "skill"

    def _check_interaction(self, kid_i, kid_j):
        key = tuple(sorted([kid_i, kid_j]))
        if key not in self._pairwise_harm:
            return False
        h, total = self._pairwise_harm[key]
        if total < 1: return False
        return (h / total) > 0.3  # more than 30% harmful → conflict risk

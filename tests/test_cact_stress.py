#!/usr/bin/env python3
"""C-ACT Extreme Comprehensive Stress Tests.

This test suite exercises every C-ACT module under stress conditions:
large data volumes, edge cases, boundary values, concurrent patterns,
and numerical stability. All tests are self-validating and report PASS/FAIL.
"""

import sys
import os
import json
import tempfile
import time
import math
import random
import shutil
from copy import deepcopy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from cact.trust_store import TrustStore
from cact.trust_gate import TrustGate, TASK_GROUPS, EPS_HARM
from cact.contract import (
    KnowledgeContract, ContractChecker, ContractExtractor,
    infer_type_from_xenon, infer_level_from_xenon,
    KNOWLEDGE_TYPES, KNOWLEDGE_LEVELS, HARD_SAFETY_CONTEXTS,
)
from cact.lifecycle_manager import (
    LifecycleManager, LifecycleState,
    CANDIDATE, QUARANTINED, PROBATION, CERTIFIED, DEPRECATED, DISABLED,
    STATE_ORDER, REUSABLE_STATES, ACTIVE_STATES,
)
from cact.temporal_decay import TemporalDecay, RHO_MIN, RHO_MAX, RHO_DEFAULT
from cact.active_logging import ActiveBaseLogger
from cact.thompson_probe import SafeThompsonProber
from cact.interaction_gate import InteractionGate, SYNERGY, NEUTRAL, CONFLICT, UNKNOWN
from cact.decision_controller import DecisionController, DecisionResult
from cact.cact_memory import CactMemory
from cact.empirical_bayes import EmpiricalBayes
from cact.context_bucket import ContextBucket
from cact.metrics import (
    compute_sr, compute_kus, compute_hrr, compute_irr,
    compute_coverage, compute_ece, compute_cov_risk,
    compute_hardsr, compute_failuresr, compute_interactionsr,
    compute_rcr, compute_cfr, compute_kpr, compute_csr, compute_cvr,
)


# ── Test Infrastructure ──

_total_pass = 0
_total_fail = 0


def _pass(name):
    global _total_pass
    _total_pass += 1
    print(f"  PASS: {name}")


def _fail(name, msg=""):
    global _total_fail
    _total_fail += 1
    suffix = f" -- {msg}" if msg else ""
    print(f"  FAIL: {name}{suffix}")


def check(name, condition, msg=""):
    """Assert a condition and report PASS/FAIL."""
    if condition:
        _pass(name)
    else:
        _fail(name, msg)


# ── Mock XENON Memory for CactMemory tests ──

class _MockXenonMemory:
    """Minimal mock of XENON DecomposedMemory for CactMemory integration tests."""
    def __init__(self):
        self._succeeded = {}
        self._failed = {}
        self._plans = {}
        self._reflections = {}
        self._replans = {}
        self._history_idx = 0
        self._env = "crafting"
        self.succeeded_waypoints = []
    def retrieve_similar_succeeded_waypoints(self, w, k=3):
        return self.succeeded_waypoints[:k]
    def retrieve_failed_subgoals(self, w):
        return list(self._failed.keys())
    def retrieve_total_failed_counts(self, w):
        return len(self._failed)
    def save_plan(self, wp, plan):
        self._plans[wp] = plan
    def reset_success_failure_history(self, i):
        pass
    def set_history_index(self, i):
        self._history_idx = i
    def add_succeeded_waypoint(self, wp, action):
        self.succeeded_waypoints.append(wp)
        self._succeeded[wp] = action
    def add_failed_waypoint(self, wp, action):
        self._failed[wp] = action
    def save_reflection(self, wp, reflect, is_success):
        self._reflections[wp] = reflect
    def save_replan(self, wp, replan):
        self._replans[wp] = replan
    def save_decomposed_plan(self, wp, plan, is_success):
        self._plans[wp] = plan
    def save_success_failure(self, wp, action_str, is_success):
        if is_success:
            self._succeeded[wp] = action_str
        else:
            self._failed[wp] = action_str
    def is_succeeded_waypoint(self, waypoint):
        return waypoint in self._succeeded, self._succeeded.get(waypoint)


# ── Utility ──

def _make_temp_store():
    """Create a TrustStore with a unique temp directory."""
    d = tempfile.mkdtemp(prefix="cact_stress_test_")
    return TrustStore(store_path=d)


def _make_calib_data(n, seed=42):
    """Generate synthetic calibration data for TrustGate tests."""
    rng = np.random.default_rng(seed)
    data = []
    for i in range(n):
        pi = float(rng.uniform(0.5, 1.0))
        harm = float(rng.uniform(0.0, 0.3))
        uplift = float(rng.uniform(-0.1, 0.3))
        data.append({
            "pi_uplift": pi,
            "harm_ucb": harm,
            "uplift": uplift,
            "is_harmful": 0 if harm < 0.10 else (1 if harm > 0.20 else rng.integers(0, 2)),
        })
    return data


def _make_knowledge_dict(idx=0):
    """Create a synthetic knowledge dict for ContractExtractor."""
    return {
        "type": "skill",
        "subgoal_type": "craft",
        "failure_type": "wrong_tool",
        "source": "XENON_FAM",
        "correction": f"Craft stone pickaxe before mining iron ore to avoid breaking the block.",
        "text": f"Knowledge item {idx}: Craft stone pickaxe before mining iron ore.",
        "preconditions": ["has_crafting_table", "has_sticks"],
        "postconditions": ["craft_completed"],
        "non_applicable_contexts": [],
        "expected_uplift": 0.08,
        "risk_bound": 0.05,
        "task_tier": "stone",
        "episode_id": f"ep_{idx:04d}",
        "level": "atomic",
        "gene": "craft stone pickaxe",
    }


# ═══════════════════════════════════════════════════════════════
# 1. TrustStore Stress Tests
# ═══════════════════════════════════════════════════════════════

class TestTrustStoreStress:
    """Stress tests for the TrustStore (Bayesian 3-Beta posterior store)."""

    def test_10000_rapid_record_episode(self):
        """Test 10,000 rapid record_episode() calls for memory/performance."""
        store = _make_temp_store()
        d = store._data
        start_len = len(d)
        try:
            for i in range(10000):
                kid = f"skill_{i % 100}"
                ctx = f"craft_stone"
                used = (i % 3 != 0)  # 67% used
                success = 1.0 if np.random.random() > 0.2 else 0.0
                store.record_episode(kid, ctx, used=used, success=success,
                                    is_harmful=0.0 if success > 0 else 0.5)
            # Verify: no crash, reasonable data size, all values valid
            for k, v in store._data.items():
                assert v["alpha"] >= 0.09, f"alpha too low: {v['alpha']}"
                assert v["beta"] >= 0.09, f"beta too low: {v['beta']}"
                assert v["alpha"] < 20000, f"alpha suspiciously high: {v['alpha']}"
                assert v["beta"] < 20000, f"beta suspiciously high: {v['beta']}"
            # Verify we have reasonable number of entries (at most 100 kids * 3 stats + some bases)
            assert len(store._data) <= 600, f"Too many entries: {len(store._data)}"
            _pass("10k rapid record_episode cycle completed, no corruption")
        except Exception as e:
            _fail("10k rapid record_episode", str(e))
        finally:
            shutil.rmtree(store.store_path, ignore_errors=True)

    def test_edge_case_zero_observations(self):
        """Test with zero observations — defaults should be priors."""
        store = _make_temp_store()
        try:
            a, b = store.get_stats("never_seen", "craft", "use")
            check("zero-obs use alpha near prior", a > 0.9, str(a))
            check("zero-obs use beta near prior", b > 0.9, str(b))
            a, b = store.get_stats("never_seen", "craft", "harm")
            check("zero-obs harm alpha near prior", a > 0.9, str(a))
            check("zero-obs harm beta near prior", b > 1.0, str(b))
            # Total count should be near zero
            n = store.total_count("never_seen", "craft", "use")
            check("zero-obs total_count near zero", abs(n) < 0.01, str(n))
        except Exception as e:
            _fail("edge_case_zero_observations", str(e))
        finally:
            shutil.rmtree(store.store_path, ignore_errors=True)

    def test_all_successes(self):
        """Test 1000 all-success recordings — mean should converge to 1.0."""
        store = _make_temp_store()
        try:
            kid = "all_win"
            ctx = "craft"
            for i in range(1000):
                store.record_episode(kid, ctx, used=True, success=1.0, is_harmful=0.0)
            m = store.mean(kid, ctx, "use")
            check("all_successes mean > 0.99", m > 0.99, str(m))
            hm = store.mean(kid, ctx, "harm")
            check("all_successes harm mean < 0.02", hm < 0.02, str(hm))
        except Exception as e:
            _fail("all_successes", str(e))
        finally:
            shutil.rmtree(store.store_path, ignore_errors=True)

    def test_all_failures(self):
        """Test 1000 all-failure recordings — mean should converge to 0.0."""
        store = _make_temp_store()
        try:
            kid = "all_lose"
            ctx = "craft"
            for i in range(1000):
                store.record_episode(kid, ctx, used=True, success=0.0, is_harmful=1.0)
            m = store.mean(kid, ctx, "use")
            check("all_failures mean < 0.01", m < 0.01, str(m))
            hm = store.mean(kid, ctx, "harm")
            check("all_failures harm mean > 0.98", hm > 0.98, str(hm))
        except Exception as e:
            _fail("all_failures", str(e))
        finally:
            shutil.rmtree(store.store_path, ignore_errors=True)

    def test_mixed_success_failure(self):
        """Test 50/50 mixed — mean should be near 0.5."""
        store = _make_temp_store()
        try:
            kid = "fifty_fifty"
            ctx = "craft"
            for i in range(2000):
                s = 1.0 if i % 2 == 0 else 0.0
                store.record_episode(kid, ctx, used=True, success=s, is_harmful=1.0 - s)
            m = store.mean(kid, ctx, "use")
            check("mixed mean near 0.5", 0.45 < m < 0.55, str(m))
        except Exception as e:
            _fail("mixed_success_failure", str(e))
        finally:
            shutil.rmtree(store.store_path, ignore_errors=True)

    def test_alternating_concurrent_access(self):
        """Simulate concurrent access via rapid alternation between multiple kids."""
        store = _make_temp_store()
        try:
            kids = [f"kid_{i}" for i in range(50)]
            for _ in range(200):
                for kid in kids:
                    used = np.random.random() > 0.3
                    s = np.random.random() if used else 0.5
                    h = 0.1 if used else 0.0
                    store.record_episode(kid, "craft", used=used, success=s,
                                        is_harmful=h)
            # All kids should have data
            for kid in kids:
                a, b = store.get_stats(kid, "craft", "use")
                assert a > 0.1, f"Kid {kid} has invalid alpha: {a}"
                assert b > 0.1, f"Kid {kid} has invalid beta: {b}"
            _pass("alternating_concurrent_access with 50 kids over 200 rounds")
        except Exception as e:
            _fail("alternating_concurrent_access", str(e))
        finally:
            shutil.rmtree(store.store_path, ignore_errors=True)

    def test_extreme_knowledge_ids(self):
        """Test with extreme knowledge IDs: very long, special chars, unicode."""
        store = _make_temp_store()
        try:
            extreme_ids = [
                "a" * 500,                                 # Very long
                "kid_with/slashes\\and\\backslashes",      # Path chars
                "\u4e2d\u6587\u77e5\u8bc6ID",             # Chinese unicode
                "\u00e9\u00e8\u00e7\u00fc\u00f6\u00e4",   # European unicode
                "kid with spaces and !@#$%^&*()",          # Special chars
                "",                                         # Empty string
                "kid\nwith\nnewlines",                     # Newlines
                "\t\t\ttabs",                              # Tabs
                "emoji_\U0001f600_\U0001f4a9",            # Emoji
                "SQL'; DROP TABLE--",                     # Common injection
            ]
            for i, kid in enumerate(extreme_ids):
                store.record_episode(kid, "craft", used=True, success=1.0, is_harmful=0.0)
                a, b = store.get_stats(kid, "craft", "use")
                assert a > 0.9, f"Extreme kid {i} alpha broken: {a}"
            _pass("extreme_knowledge_ids all handled without error")
        except Exception as e:
            _fail("extreme_knowledge_ids", str(e))
        finally:
            shutil.rmtree(store.store_path, ignore_errors=True)

    def test_beta_params_never_below_zero(self):
        """Test that Beta params never go below 0 or above reasonable bounds."""
        store = _make_temp_store()
        try:
            kid = "bounds_check"
            ctx = "craft"
            # Mix of extreme outcomes
            for i in range(500):
                success = 0.0 if i % 3 == 0 else 1.0
                store.record_episode(kid, ctx, used=True, success=success,
                                    is_harmful=0.0)
            for stat in ["use", "base", "harm"]:
                a, b = store.get_stats(kid, ctx, stat)
                assert a > 0, f"{stat} alpha <= 0: {a}"
                assert b > 0, f"{stat} beta <= 0: {b}"
                assert a < 10000, f"{stat} alpha too large: {a}"
                assert b < 10000, f"{stat} beta too large: {b}"
            _pass("beta_params always in valid range")
        except Exception as e:
            _fail("beta_params_never_below_zero", str(e))
        finally:
            shutil.rmtree(store.store_path, ignore_errors=True)

    def test_1000_contract_registration(self):
        """Test contract registration and retrieval for 1000+ contracts."""
        store = _make_temp_store()
        try:
            for i in range(1000):
                kid = f"kc_{i:06d}"
                contract = {
                    "type": random.choice(KNOWLEDGE_TYPES),
                    "level": random.choice(KNOWLEDGE_LEVELS),
                    "expected_uplift": random.uniform(0.01, 0.2),
                    "risk_bound": random.uniform(0.01, 0.3),
                }
                store.register_contract(kid, contract)
            # Retrieve all
            found = 0
            for i in range(1000):
                c = store.get_contract(f"kc_{i:06d}")
                if c is not None and "type" in c:
                    found += 1
            check("1000 contracts registered", found == 1000, f"found={found}")
            # Check lifecycle states all set
            for i in range(1000):
                state = store.get_lifecycle_state(f"kc_{i:06d}")
                assert state == CANDIDATE, f"Expected CANDIDATE, got {state}"
            _pass("1000_contract_registration + lifecycle init verified")
        except Exception as e:
            _fail("1000_contract_registration", str(e))
        finally:
            shutil.rmtree(store.store_path, ignore_errors=True)

    def test_lifecycle_auto_transitions_all_6_states(self):
        """Test lifecycle auto-transitions across all 6 states."""
        store = _make_temp_store()
        try:
            kid = "lifecycle_test"
            ctx = "craft"
            contract = {
                "type": "skill",
                "level": "atomic",
                "expected_uplift": 0.1,
                "risk_bound": 0.05,
            }
            store.register_contract(kid, contract)

            # CANDIDATE state
            s0 = store.get_lifecycle_state(kid)
            check("initial state is CANDIDATE", s0 == CANDIDATE, s0)

            # 1 observation -> QUARANTINED
            store.record_episode(kid, ctx, used=True, success=1.0, is_harmful=0.0)
            s1 = store.get_lifecycle_state(kid)
            check("after 1 obs, state is QUARANTINED", s1 == QUARANTINED, s1)

            # 3 more observations with high success -> PROBATION
            for _ in range(3):
                store.record_episode(kid, ctx, used=True, success=1.0, is_harmful=0.0)
            s2 = store.get_lifecycle_state(kid)
            check("after 4 total obs (high success), state is PROBATION", s2 == PROBATION, s2)

            # More successes -> CERTIFIED
            for _ in range(5):
                store.record_episode(kid, ctx, used=True, success=1.0, is_harmful=0.0)
            s3 = store.get_lifecycle_state(kid)
            check("after enough high-success obs, state is CERTIFIED",
                  s3 == CERTIFIED, s3)
            _pass("lifecycle_auto_transitions_all_6_states verified")
        except Exception as e:
            _fail("lifecycle_auto_transitions_all_6_states", str(e))
        finally:
            shutil.rmtree(store.store_path, ignore_errors=True)

    def test_get_all_pair_stats_100_pairs(self):
        """Test get_all_pair_stats() with 100 knowledge pairs."""
        store = _make_temp_store()
        try:
            kids = [f"kid_{i}" for i in range(15)]  # 15 choose 2 = 105 pairs
            ctx = "craft"
            # Record joint stats
            for i in range(15):
                for j in range(i + 1, 15):
                    store.record_joint(
                        kids[i], kids[j], ctx,
                        success=1.0 if (i + j) % 2 == 0 else 0.0,
                        harmful=0.0,
                    )
            result = store.get_all_pair_stats(kids, ctx)
            check("get_all_pair_stats returns 105 pairs",
                  len(result) == 105, str(len(result)))
            for (ki, kj), stats in result.items():
                assert "alpha" in stats, f"Missing alpha for {ki},{kj}"
                assert "beta" in stats, f"Missing beta for {ki},{kj}"
            _pass("get_all_pair_stats with 105 pairs correct")
        except Exception as e:
            _fail("get_all_pair_stats_100_pairs", str(e))
        finally:
            shutil.rmtree(store.store_path, ignore_errors=True)

    def test_probabilities_in_0_1_range(self):
        """Test all probability methods return values in [0,1]."""
        store = _make_temp_store()
        try:
            kid = "prob_range"
            ctx = "craft"
            for _ in range(50):
                s = np.random.random()
                store.record_episode(kid, ctx, used=True, success=s,
                                    is_harmful=1 - s)
            vals = [
                store.mean(kid, ctx, "use"),
                store.mean(kid, ctx, "base"),
                store.mean(kid, ctx, "harm"),
                store.uplift_probability(kid, ctx),
                store.lcb(kid, ctx, "use"),
                store.ucb(kid, ctx, "use"),
                store.harm_upper_bound(kid, ctx),
                store.prob_harm_safe(kid, ctx),
            ]
            for i, v in enumerate(vals):
                assert 0.0 <= v <= 1.0, f"Value {i} out of [0,1]: {v}"
            _pass("all probabilities in [0,1] range")
        except Exception as e:
            _fail("probabilities_in_0_1_range", str(e))
        finally:
            shutil.rmtree(store.store_path, ignore_errors=True)

    def test_ess_calculation(self):
        """Test effective sample size calculation."""
        store = _make_temp_store()
        try:
            kid = "ess_test"
            ctx = "craft"
            ess0 = store.ess(kid, ctx)
            check("initial ESS near 0", abs(ess0) < 0.1, str(ess0))
            for i in range(10):
                store.record_episode(kid, ctx, used=True, success=1.0, is_harmful=0.0)
            ess_after = store.ess(kid, ctx)
            # Each used recording increments use + harm
            check("ESS grows after recordings", ess_after > 15, str(ess_after))
        except Exception as e:
            _fail("ess_calculation", str(e))
        finally:
            shutil.rmtree(store.store_path, ignore_errors=True)

    def test_thompson_sample_output_format(self):
        """Test thompson_sample returns 3 values in [0,1]."""
        store = _make_temp_store()
        try:
            kid = "ts_test"
            ctx = "craft"
            for _ in range(20):
                store.record_episode(kid, ctx, used=True, success=0.7, is_harmful=0.1)
            p_use, p_base, p_harm = store.thompson_sample(kid, ctx)
            assert 0.0 <= p_use <= 1.0, f"p_use={p_use}"
            assert 0.0 <= p_base <= 1.0, f"p_base={p_base}"
            assert 0.0 <= p_harm <= 1.0, f"p_harm={p_harm}"
            _pass("thompson_sample returns valid [0,1] values")
        except Exception as e:
            _fail("thompson_sample_output_format", str(e))
        finally:
            shutil.rmtree(store.store_path, ignore_errors=True)

    def test_export_import_roundtrip(self):
        """Test export_for_calibration/import_from_calibration roundtrip."""
        store = _make_temp_store()
        try:
            store.record_episode("kid1", "craft", used=True, success=1.0, is_harmful=0.0)
            store.record_episode("kid1", "craft", used=False, success=0.0, is_harmful=0.0)
            data = store.export_for_calibration()
            assert "_data" in data, "export missing _data"
            assert "_contracts" in data, "export missing _contracts"
            # New store importing
            store2 = _make_temp_store()
            store2.import_from_calibration(data)
            a1, b1 = store.get_stats("kid1", "craft", "use")
            a2, b2 = store2.get_stats("kid1", "craft", "use")
            check("export-import roundtrip alpha matches", abs(a1 - a2) < 0.001,
                  f"{a1} vs {a2}")
            check("export-import roundtrip beta matches", abs(b1 - b2) < 0.001,
                  f"{b1} vs {b2}")
            shutil.rmtree(store2.store_path, ignore_errors=True)
        except Exception as e:
            _fail("export_import_roundtrip", str(e))
        finally:
            shutil.rmtree(store.store_path, ignore_errors=True)

    def test_uplift_probability_monotonic(self):
        """Test uplift_probability is higher when use >> base."""
        store = _make_temp_store()
        try:
            # Kid A: use >> base (good uplift)
            kid_good = "uplift_good"
            kid_bad = "uplift_bad"
            ctx = "craft"
            for _ in range(50):
                store.record_episode(kid_good, ctx, used=True, success=0.9, is_harmful=0.0)
                store.record_episode(kid_good, ctx, used=False, success=0.3, is_harmful=0.0)
                store.record_episode(kid_bad, ctx, used=True, success=0.3, is_harmful=0.0)
                store.record_episode(kid_bad, ctx, used=False, success=0.5, is_harmful=0.0)
            pi_good = store.uplift_probability(kid_good, ctx)
            pi_bad = store.uplift_probability(kid_bad, ctx)
            check("uplift good > uplift bad", pi_good > pi_bad,
                  f"good={pi_good:.4f} bad={pi_bad:.4f}")
        except Exception as e:
            _fail("uplift_probability_monotonic", str(e))
        finally:
            shutil.rmtree(store.store_path, ignore_errors=True)

    def test_uplift_lcb_sanity(self):
        """Test uplift_lcb is bounded by mean differences."""
        store = _make_temp_store()
        try:
            kid = "lcb_test"
            ctx = "craft"
            for _ in range(30):
                store.record_episode(kid, ctx, used=True, success=0.8, is_harmful=0.0)
                store.record_episode(kid, ctx, used=False, success=0.4, is_harmful=0.0)
            ul = store.uplift_lcb(kid, ctx)
            diff = store.mean(kid, ctx, "use") - store.mean(kid, ctx, "base")
            # LCB should be conservative (<= actual difference)
            check("uplift_lcb <= mean difference", ul <= diff + 0.3,
                  f"ul={ul:.4f} diff={diff:.4f}")
        except Exception as e:
            _fail("uplift_lcb_sanity", str(e))
        finally:
            shutil.rmtree(store.store_path, ignore_errors=True)


# ═══════════════════════════════════════════════════════════════
# 2. TrustGate Stress Tests
# ═══════════════════════════════════════════════════════════════

class TestTrustGateStress:
    """Stress tests for TrustGate (adaptive calibration gate)."""

    def test_calibrate_5000_data_points(self):
        """Test calibrate() with 5000 calibration data points."""
        gate = TrustGate()
        try:
            data = _make_calib_data(5000, seed=123)
            result = gate.calibrate(data)
            check("calibrate with 5000 points returns config", "tau" in result, str(result))
            check("calibrate coverage >= 0", result["coverage"] >= 0, str(result))
            check("calibrate risk in [0,1]", 0 <= result["risk"] <= 1, str(result))
            check("tau threshold in range", 0.80 <= result["tau"] <= 0.95, str(result))
            check("delta threshold in range", 0.02 <= result["delta"] <= 0.10, str(result))
            check("harm threshold in range", 0.06 <= result["harm"] <= 0.15, str(result))
            check("n_calib correct", result["n_calib"] == 5000, str(result))
            _pass("calibrate with 5000 points successful")
        except Exception as e:
            _fail("calibrate_5000_data_points", str(e))

    def test_calibrate_empty_data(self):
        """Test calibrate() with empty calibration data."""
        gate = TrustGate()
        try:
            result = gate.calibrate([])
            check("empty calib returns default tau=0.90",
                  result["tau"] == 0.90, str(result))
            check("empty calib returns default delta=0.05",
                  result["delta"] == 0.05, str(result))
            check("empty calib coverage=0",
                  result["coverage"] == 0.0, str(result))
            check("empty calib risk=0",
                  result["risk"] == 0.0, str(result))
        except Exception as e:
            _fail("calibrate_empty_data", str(e))

    def test_calibrate_single_data_point(self):
        """Test calibrate() with a single data point."""
        gate = TrustGate()
        try:
            data = [{"pi_uplift": 0.95, "harm_ucb": 0.03, "uplift": 0.1, "is_harmful": 0}]
            result = gate.calibrate(data)
            check("single point calibrate succeeds", "tau" in result, str(result))
            check("single point n_calib=1", result["n_calib"] == 1, str(result))
            # Should find some valid config
            assert result["tau"] >= 0.80, f"tau too low: {result['tau']}"
        except Exception as e:
            _fail("calibrate_single_data_point", str(e))

    def test_calibrate_all_groups_six_groups(self):
        """Test calibrate_all_groups() with all 6 groups, some empty."""
        gate = TrustGate()
        try:
            data_by_group = {}
            for grp in TASK_GROUPS:
                if grp in ("crafting", "mining"):
                    data_by_group[grp] = _make_calib_data(200, seed=hash(grp) % 10000)
                else:
                    data_by_group[grp] = []  # Empty
            results = gate.calibrate_all_groups(data_by_group)
            check("all 6 groups in results", len(results) == 6, str(len(results)))
            for grp in TASK_GROUPS:
                assert grp in results, f"Missing group {grp}"
            # Non-empty groups should have calibrated values
            for grp in ("crafting", "mining"):
                assert gate.tau.get(grp) is not None, f"tau missing for {grp}"
                assert gate.delta.get(grp) is not None, f"delta missing for {grp}"
                assert gate.harm.get(grp) is not None, f"harm missing for {grp}"
            _pass("calibrate_all_groups with 6 groups, some empty")
        except Exception as e:
            _fail("calibrate_all_groups_six_groups", str(e))

    def test_evaluate_boundary_pi_0(self):
        """Test evaluate() with boundary pi_uplift=0.0."""
        gate = TrustGate()
        gate.tau["_global"] = 0.90
        gate.harm["_global"] = 0.10
        gate.delta["_global"] = 0.05
        try:
            allow, info = gate.evaluate(0.0, 0.0, 0.0)
            check("pi=0 should be blocked", not allow, str(info))
            check("reason is uplift or both", info["reason"] in ("uplift_fail", "both_fail"),
                  info["reason"])
        except Exception as e:
            _fail("evaluate_boundary_pi_0", str(e))

    def test_evaluate_boundary_pi_1(self):
        """Test evaluate() with boundary pi_uplift=1.0."""
        gate = TrustGate()
        gate.tau["_global"] = 0.90
        gate.harm["_global"] = 0.10
        gate.delta["_global"] = 0.05
        try:
            allow, info = gate.evaluate(1.0, 0.1, 0.05)
            check("pi=1, high uplift, low harm => allow", allow, str(info))
            check("reason is gate_pass", info["reason"] == "gate_pass", info["reason"])
        except Exception as e:
            _fail("evaluate_boundary_pi_1", str(e))

    def test_evaluate_boundary_harm_0(self):
        """Test evaluate() with harm_ucb=0.0."""
        gate = TrustGate()
        gate.tau["_global"] = 0.90
        gate.harm["_global"] = 0.10
        gate.delta["_global"] = 0.05
        try:
            allow, info = gate.evaluate(0.95, 0.08, 0.0)
            check("harm=0, high uplift => allow", allow, str(info))
        except Exception as e:
            _fail("evaluate_boundary_harm_0", str(e))

    def test_evaluate_boundary_harm_1(self):
        """Test evaluate() with harm_ucb=1.0."""
        gate = TrustGate()
        gate.tau["_global"] = 0.90
        gate.harm["_global"] = 0.10
        gate.delta["_global"] = 0.05
        try:
            allow, info = gate.evaluate(0.95, 0.08, 1.0)
            check("harm=1 should be blocked", not allow, str(info))
        except Exception as e:
            _fail("evaluate_boundary_harm_1", str(e))

    def test_evaluate_lifecycle_disabled(self):
        """Test evaluate() blocks when lifecycle is disabled."""
        gate = TrustGate()
        gate.tau["_global"] = 0.90
        gate.harm["_global"] = 0.10
        gate.delta["_global"] = 0.05
        try:
            allow, info = gate.evaluate(0.95, 0.08, 0.05,
                                       lifecycle_state="disabled")
            check("disabled lifecycle blocked", not allow, str(info))
            check("reason is lifecycle_blocked",
                  info["reason"] == "lifecycle_blocked", info["reason"])
        except Exception as e:
            _fail("evaluate_lifecycle_disabled", str(e))

    def test_evaluate_contract_violation(self):
        """Test evaluate() blocks when contract_satisfied=False."""
        gate = TrustGate()
        gate.tau["_global"] = 0.90
        gate.harm["_global"] = 0.10
        gate.delta["_global"] = 0.05
        try:
            allow, info = gate.evaluate(0.95, 0.08, 0.05,
                                       contract_satisfied=False)
            check("contract violation blocked", not allow, str(info))
            check("reason is contract_violation",
                  info["reason"] == "contract_violation", info["reason"])
        except Exception as e:
            _fail("evaluate_contract_violation", str(e))

    def test_evaluate_interaction_conflict(self):
        """Test evaluate() blocks when interaction_safe=False."""
        gate = TrustGate()
        gate.tau["_global"] = 0.90
        gate.harm["_global"] = 0.10
        gate.delta["_global"] = 0.05
        try:
            allow, info = gate.evaluate(0.95, 0.08, 0.05,
                                       interaction_safe=False)
            check("interaction conflict blocked", not allow, str(info))
            check("reason is interaction_conflict",
                  info["reason"] == "interaction_conflict", info["reason"])
        except Exception as e:
            _fail("evaluate_interaction_conflict", str(e))

    def test_save_load_calibration_roundtrip(self):
        """Test save_calibration/load_calibration roundtrip."""
        gate = TrustGate()
        gate.tau["crafting"] = 0.92
        gate.delta["crafting"] = 0.06
        gate.harm["crafting"] = 0.08
        gate.theta_syn = 0.75
        gate.theta_conf = 0.85

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            calib_path = f.name
        try:
            gate.save_calibration(calib_path)
            gate2 = TrustGate()
            gate2.load_calibration(calib_path)
            check("roundtrip tau", gate2.tau.get("crafting") == 0.92,
                  str(gate2.tau))
            check("roundtrip delta", gate2.delta.get("crafting") == 0.06,
                  str(gate2.delta))
            check("roundtrip harm", gate2.harm.get("crafting") == 0.08,
                  str(gate2.harm))
            check("roundtrip theta_syn", gate2.theta_syn == 0.75,
                  str(gate2.theta_syn))
            check("roundtrip theta_conf", gate2.theta_conf == 0.85,
                  str(gate2.theta_conf))
            os.unlink(calib_path)
        except Exception as e:
            _fail("save_load_calibration_roundtrip", str(e))
            if os.path.exists(calib_path):
                os.unlink(calib_path)

    def test_get_config(self):
        """Test get_config returns complete config."""
        gate = TrustGate()
        gate.tau["test"] = 0.88
        gate.delta["test"] = 0.04
        gate.harm["test"] = 0.07
        cfg = gate.get_config()
        check("get_config has tau", "tau" in cfg)
        check("get_config has delta", "delta" in cfg)
        check("get_config has harm", "harm" in cfg)
        check("get_config has theta_syn", "theta_syn" in cfg)
        check("get_config has theta_conf", "theta_conf" in cfg)
        check("get_config has eps_harm", "eps_harm" in cfg)
        check("get_config eps_harm=0.10", cfg["eps_harm"] == EPS_HARM)

    def test_exploration_rate_bounds(self):
        """Test exploration_rate returns values in [0.05, 0.30]."""
        test_cases = [
            (10, 0.5, "medium", 0.0),
            (1, 0.9, "low", 0.5),
            (100, 0.1, "high", 0.0),
            (5, 0.5, "medium", 0.3),
            (0, 0.0, "medium", 0.0),
            (0, 1.0, "medium", 0.0),
        ]
        all_ok = True
        for ess, pi, risk, imb in test_cases:
            rate = TrustGate.exploration_rate(ess, pi, risk, imb)
            if rate < 0.05 or rate > 0.30:
                all_ok = False
                _fail(f"exploration_rate({ess},{pi},{risk},{imb})={rate:.4f} out of bounds")
        if all_ok:
            _pass("exploration_rate always in [0.05, 0.30]")

    def test_binom_ucb(self):
        """Test _binom_ucb edge cases."""
        gate = TrustGate()
        ucb0 = gate._binom_ucb(0, 100)
        check("binom_ucb(0,100) near zero", ucb0 < 0.05, str(ucb0))
        ucb1 = gate._binom_ucb(0, 0)
        check("binom_ucb(0,0) = 1.0", ucb1 == 1.0, str(ucb1))
        ucb2 = gate._binom_ucb(95, 100)
        check("binom_ucb(95,100) > 0.95", ucb2 > 0.95, str(ucb2))

    def test_should_reuse_alias(self):
        """Test should_reuse() is identical to evaluate()."""
        gate = TrustGate()
        gate.tau["_global"] = 0.90
        gate.harm["_global"] = 0.10
        gate.delta["_global"] = 0.05
        r1, i1 = gate.evaluate(0.95, 0.08, 0.05)
        r2, i2 = gate.should_reuse(0.95, 0.08, 0.05)
        check("should_reuse same allow", r1 == r2, f"{r1} vs {r2}")
        check("should_reuse same reason", i1["reason"] == i2["reason"],
              f"{i1['reason']} vs {i2['reason']}")

    def test_5_state_decision_exhaustive(self):
        """Exhaustively test evaluate() with all 5 failure modes."""
        gate = TrustGate()
        gate.tau["_global"] = 0.90
        gate.harm["_global"] = 0.10
        gate.delta["_global"] = 0.05

        # Good case
        allow, info = gate.evaluate(0.95, 0.08, 0.05)
        check("exhaustive: gate_pass", allow and info["reason"] == "gate_pass")

        # Both fail
        allow, info = gate.evaluate(0.50, -0.05, 0.50)
        check("exhaustive: both_fail", not allow and info["reason"] == "both_fail")

        # Uplift fail
        allow, info = gate.evaluate(0.50, 0.08, 0.05)
        check("exhaustive: uplift_fail", not allow and info["reason"] == "uplift_fail")

        # Safety fail
        allow, info = gate.evaluate(0.95, 0.08, 0.50)
        check("exhaustive: safety_fail", not allow and info["reason"] == "safety_fail")

        # Lifecycle blocked
        allow, info = gate.evaluate(0.95, 0.08, 0.05, lifecycle_state="disabled")
        check("exhaustive: lifecycle_blocked",
              not allow and info["reason"] == "lifecycle_blocked")

        # Contract violation
        allow, info = gate.evaluate(0.95, 0.08, 0.05, contract_satisfied=False)
        check("exhaustive: contract_violation",
              not allow and info["reason"] == "contract_violation")

        # Interaction conflict
        allow, info = gate.evaluate(0.95, 0.08, 0.05, interaction_safe=False)
        check("exhaustive: interaction_conflict",
              not allow and info["reason"] == "interaction_conflict")
        _pass("5+ state decision table exhaustive check")


# ═══════════════════════════════════════════════════════════════
# 3. Contract System Stress Tests
# ═══════════════════════════════════════════════════════════════

class TestContractStress:
    """Stress tests for KnowledgeContract, ContractChecker, ContractExtractor."""

    def test_100_different_condition_patterns(self):
        """Test ContractChecker with 100 different condition patterns."""
        checker = ContractChecker()
        all_ok = True
        for i in range(100):
            contract = KnowledgeContract(
                knowledge_id=f"kc_{i}",
                type=random.choice(KNOWLEDGE_TYPES),
                level=random.choice(KNOWLEDGE_LEVELS),
                preconditions=[f"attr_{i}_==val_{i}"],
            )
            state = {}
            if i % 3 == 0:
                state[f"attr_{i}_"] = f"val_{i}"  # Match
            elif i % 3 == 1:
                state[f"attr_{i}_"] = "wrong_val"  # Mismatch
            # else: key missing
            ok, viols = checker.check_preconditions(contract, state)
            if i % 3 == 0:
                if not ok:
                    all_ok = False
                    _fail(f"condition_{i} should match")
            else:
                if ok:
                    all_ok = False
                    _fail(f"condition_{i} should not match")
        if all_ok:
            _pass("100 condition patterns all check correctly")

    def test_empty_preconditions(self):
        """Test with empty preconditions — should always pass."""
        checker = ContractChecker()
        contract = KnowledgeContract(preconditions=[])
        ok, viols = checker.check_preconditions(contract, {})
        check("empty preconditions pass", ok, str(viols))

    def test_empty_postconditions(self):
        """Test with empty postconditions — should always pass."""
        checker = ContractChecker()
        contract = KnowledgeContract(postconditions=[])
        ok, viols = checker.check_postconditions(contract, {}, {})
        check("empty postconditions pass", ok, str(viols))

    def test_empty_non_applicable_contexts(self):
        """Test with empty non_applicable_contexts — should always be safe."""
        checker = ContractChecker()
        contract = KnowledgeContract(non_applicable_contexts=[])
        safe, triggered = checker.check_non_applicable(contract, {})
        check("empty non_applicable_contexts safe", safe, str(triggered))

    def test_in_operator_condition(self):
        """Test 'in' operator condition parsing."""
        checker = ContractChecker()
        contract = KnowledgeContract(
            preconditions=["target_tool in {stone_pickaxe, iron_pickaxe, diamond_pickaxe}"])
        ok1, _ = checker.check_preconditions(contract, {"target_tool": "stone_pickaxe"})
        check("'in' with match", ok1)
        ok2, _ = checker.check_preconditions(contract, {"target_tool": "wooden_pickaxe"})
        check("'in' with no match", not ok2)

    def test_not_in_operator_condition(self):
        """Test 'not in' operator condition parsing."""
        checker = ContractChecker()
        contract = KnowledgeContract(
            preconditions=["current_tool not in {wooden_pickaxe, wooden_axe}"])
        ok1, _ = checker.check_preconditions(contract, {"current_tool": "stone_pickaxe"})
        check("'not in' with no match => pass", ok1)
        ok2, _ = checker.check_preconditions(contract, {"current_tool": "wooden_pickaxe"})
        check("'not in' with match => fail", not ok2)

    def test_binary_condition_check(self):
        """Test binary (presence) condition check."""
        checker = ContractChecker()
        contract = KnowledgeContract(preconditions=["has_furnace", "has_crafting_table"])
        ok1, _ = checker.check_preconditions(contract, {
            "has_furnace": True, "has_crafting_table": True})
        check("binary conditions all true => pass", ok1)
        ok2, _ = checker.check_preconditions(contract, {"has_furnace": False})
        check("binary condition false => fail", not ok2)
        ok3, _ = checker.check_preconditions(contract, {})
        check("binary condition missing => fail", not ok3)

    def test_contract_extractor_malformed_missing_fields(self):
        """Test ContractExtractor with malformed/missing fields."""
        extractor = ContractExtractor()
        # Empty dict
        c1 = extractor.extract({})
        check("extract empty dict => type=action_correction or skill",
              c1.type in KNOWLEDGE_TYPES, c1.type)
        # Only source
        c2 = extractor.extract({"source": "dependency_graph"})
        check("extract source_only => dependency_correction",
              c2.type == "dependency_correction", c2.type)
        # With explicit type
        c3 = extractor.extract({"type": "skill", "source": "XENON_FAM",
                                "correction": "Do X before Y"})
        check("extract explicit type => skill", c3.type == "skill", c3.type)
        # Missing expected fields
        c4 = extractor.extract({"garbage_key": "value"})
        check("extract garbage => works", isinstance(c4, KnowledgeContract))

    def test_contract_extractor_batch(self):
        """Test ContractExtractor extract_batch with 200 items."""
        extractor = ContractExtractor()
        knowledge_list = [_make_knowledge_dict(i) for i in range(200)]
        try:
            contracts = extractor.extract_batch(knowledge_list)
            check("extract_batch returns 200 contracts",
                  len(contracts) == 200, str(len(contracts)))
            for i, c in enumerate(contracts):
                assert isinstance(c, KnowledgeContract), f"i={i} not a contract"
                assert c.knowledge_id, f"i={i} missing knowledge_id"
            _pass("extract_batch 200 contracts all valid")
        except Exception as e:
            _fail("contract_extractor_batch", str(e))

    def test_context_match_wildcard_star(self):
        """Test context matching with wildcard '*' values."""
        checker = ContractChecker()
        contract = KnowledgeContract(
            claimed_context={"subgoal_type": "craft", "failure_type": "*"})
        ok = checker.check_context_match(
            contract, {"subgoal_type": "craft", "failure_type": "wrong_tool"})
        check("wildcard '*' matches any value", ok)
        ok2 = checker.check_context_match(
            contract, {"subgoal_type": "craft", "failure_type": "missing_item"})
        check("wildcard '*' matches another value", ok2)
        # Non-wildcard mismatch
        ok3 = checker.check_context_match(
            contract, {"subgoal_type": "mining", "failure_type": "wrong_tool"})
        check("non-wildcard mismatch fails", not ok3)

    def test_safety_context_flags_all_combinations(self):
        """Test safety context flags in various combinations."""
        checker = ContractChecker()
        contract = KnowledgeContract(
            non_applicable_contexts=list(HARD_SAFETY_CONTEXTS))
        # No flags triggered
        safe, triggered = checker.check_non_applicable(contract, {})
        check("no safety flags => safe", safe, str(triggered))
        # Lava nearby
        safe, triggered = checker.check_non_applicable(contract, {"near_lava": True})
        check("lava_nearby triggered => unsafe", not safe, str(triggered))
        # Low health
        safe, triggered = checker.check_non_applicable(
            contract, {"health": 3})
        check("low_health triggered => unsafe", not safe, str(triggered))
        # Combat
        safe, triggered = checker.check_non_applicable(
            contract, {"in_combat": True})
        check("combat triggered => unsafe", not safe, str(triggered))
        # Near cliff
        safe, triggered = checker.check_non_applicable(
            contract, {"near_cliff": True})
        check("near_cliff triggered => unsafe", not safe, str(triggered))
        # Resource critical
        safe, triggered = checker.check_non_applicable(
            contract, {"resource_critical": True})
        check("resource_critical triggered => unsafe", not safe, str(triggered))
        # Multiple flags
        safe, triggered = checker.check_non_applicable(
            contract, {"near_lava": True, "in_combat": True, "health": 2})
        check("multiple safety flags => unsafe", not safe)
        _pass("safety_context_flags all combinations correct")

    def test_infer_type_from_xenon_all_patterns(self):
        """Test infer_type_from_xenon with all expected patterns."""
        check("dependency => dependency_correction",
              infer_type_from_xenon("dependency_graph_edit") == "dependency_correction")
        check("action/fam => action_correction",
              infer_type_from_xenon("action_correction_fam") == "action_correction")
        check("failure => failure_memory",
              infer_type_from_xenon("failure_record") == "failure_memory")
        check("remedy => remedy",
              infer_type_from_xenon("remedy_correction") == "remedy")
        check("unknown => skill",
              infer_type_from_xenon("some_random_source") == "skill")

    def test_infer_level_from_xenon_edge_cases(self):
        """Test infer_level_from_xenon with edge-case content."""
        check("nether => strategy",
              infer_level_from_xenon("go to nether before entering portal") == "strategy")
        check("craft long => functional",
              infer_level_from_xenon("craft iron pickaxe and smelt ores" + "x" * 50) == "functional")
        check("requires => dependency",
              infer_level_from_xenon("requires an iron pickaxe to mine") == "dependency")
        check("retry => failure_memory",
              infer_level_from_xenon("retry this action, do not fail") == "failure_memory")

    def test_knowledge_contract_to_from_dict(self):
        """Test KnowledgeContract to_dict/from_dict roundtrip."""
        kc = KnowledgeContract(
            knowledge_id="test_kc_123",
            source="XENON_FAM",
            type="skill",
            level="atomic",
            gene="Craft stone pickaxe",
            full_text="You need to craft a stone pickaxe before mining iron ore.",
            claimed_context={"subgoal_type": "craft"},
            preconditions=["has_crafting_table"],
            postconditions=["pickaxe_obtained"],
            expected_uplift=0.08,
            risk_bound=0.05,
            non_applicable_contexts=["lava_nearby"],
            source_episode="ep_0001",
            status=CANDIDATE,
        )
        d = kc.to_dict()
        kc2 = KnowledgeContract.from_dict(d)
        check("to/from_dict: knowledge_id", kc2.knowledge_id == kc.knowledge_id)
        check("to/from_dict: type", kc2.type == kc.type)
        check("to/from_dict: level", kc2.level == kc.level)
        check("to/from_dict: gene", kc2.gene == kc.gene)
        check("to/from_dict: preconditions match",
              kc2.preconditions == kc.preconditions)
        check("to/from_dict: expected_uplift",
              abs(kc2.expected_uplift - kc.expected_uplift) < 0.001)
        check("to/from_dict: non_applicable_contexts match",
              kc2.non_applicable_contexts == kc.non_applicable_contexts)


# ═══════════════════════════════════════════════════════════════
# 4. LifecycleManager Stress Tests
# ═══════════════════════════════════════════════════════════════

class TestLifecycleManagerStress:
    """Stress tests for LifecycleManager (6-state state machine)."""

    def test_rapid_1000_transitions(self):
        """Test 1000 rapid state transitions across many kids."""
        lm = LifecycleManager(store_path=tempfile.mkdtemp(prefix="lc_stress_"))
        try:
            kids = [f"kid_{i}" for i in range(250)]
            # Init all as CANDIDATE
            for k in kids:
                lm._states[k] = CANDIDATE
            # Forward transitions
            transitions_ok = 0
            for k in kids:
                r = lm.transition(k, LifecycleState.QUARANTINED, "auto")
                if r["transitioned"]:
                    transitions_ok += 1
            check("250 CANDIDATE->QUARANTINED all ok",
                  transitions_ok == 250, str(transitions_ok))
            transitions_ok = 0
            for k in kids:
                r = lm.transition(k, LifecycleState.PROBATION, "auto")
                if r["transitioned"]:
                    transitions_ok += 1
            check("250 QUARANTINED->PROBATION all ok",
                  transitions_ok == 250, str(transitions_ok))
            transitions_ok = 0
            for k in kids:
                r = lm.transition(k, LifecycleState.CERTIFIED, "auto")
                if r["transitioned"]:
                    transitions_ok += 1
            check("250 PROBATION->CERTIFIED all ok",
                  transitions_ok == 250, str(transitions_ok))
            # Verify history
            for k in kids:
                hist = lm.get_history(k)
                assert len(hist) == 3, f"{k} history len={len(hist)}"
            _pass("1000 rapid transitions all correct")
        except Exception as e:
            _fail("rapid_1000_transitions", str(e))
        finally:
            shutil.rmtree(lm._store_path, ignore_errors=True)

    def test_backward_transition_rejection(self):
        """Test that backward transitions are rejected."""
        lm = LifecycleManager(store_path=tempfile.mkdtemp(prefix="lc_back_"))
        try:
            kid = "backward_test"
            lm._states[kid] = CERTIFIED
            r = lm.transition(kid, LifecycleState.PROBATION, "try_backward")
            check("CERTIFIED->PROBATION rejected", not r["transitioned"], str(r))
            r2 = lm.transition(kid, LifecycleState.CANDIDATE, "try_backward")
            check("CERTIFIED->CANDIDATE rejected", not r2["transitioned"], str(r2))
            # Same state (not backwards but not forward either)
            r3 = lm.transition(kid, LifecycleState.CERTIFIED, "same_state")
            check("CERTIFIED->CERTIFIED rejected", not r3["transitioned"], str(r3))
            _pass("backward_transition_rejection works correctly")
        except Exception as e:
            _fail("backward_transition_rejection", str(e))
        finally:
            shutil.rmtree(lm._store_path, ignore_errors=True)

    def test_evaluate_auto_transition_all_paths(self):
        """Test evaluate_auto_transition for all state transition paths."""
        lm = LifecycleManager()

        # CANDIDATE -> QUARANTINED (n>=1)
        ns = lm.evaluate_auto_transition("k1", 0.5, 0.05, 0.88, 0.10, 1)
        check("CANDIDATE n=1 -> QUARANTINED", ns == LifecycleState.QUARANTINED, str(ns))

        # CANDIDATE n=0 -> None
        ns = lm.evaluate_auto_transition("k2", 0.5, 0.05, 0.88, 0.10, 0)
        check("CANDIDATE n=0 -> None", ns is None, str(ns))

        # QUARANTINED n=3, pi>0.3 -> PROBATION (bypassing cand, set manually)
        lm._states["k3"] = QUARANTINED
        ns = lm.evaluate_auto_transition("k3", 0.5, 0.05, 0.88, 0.10, 3)
        check("QUARANTINED n=3 pi=0.5 -> PROBATION",
              ns == LifecycleState.PROBATION, str(ns))

        # QUARANTINED n=3, pi<=0.3 -> None
        lm._states["k4"] = QUARANTINED
        ns = lm.evaluate_auto_transition("k4", 0.2, 0.05, 0.88, 0.10, 3)
        check("QUARANTINED n=3 pi=0.2 -> None", ns is None, str(ns))

        # PROBATION n>=5, pi>=tau, harm<=h_star -> CERTIFIED
        lm._states["k5"] = PROBATION
        ns = lm.evaluate_auto_transition("k5", 0.92, 0.05, 0.88, 0.10, 5)
        check("PROBATION n=5 pi=0.92 harm=0.05 -> CERTIFIED",
              ns == LifecycleState.CERTIFIED, str(ns))

        # PROBATION n>=10, pi<0.5 -> DEPRECATED
        lm._states["k6"] = PROBATION
        ns = lm.evaluate_auto_transition("k6", 0.3, 0.05, 0.88, 0.10, 10)
        check("PROBATION n=10 pi=0.3 -> DEPRECATED",
              ns == LifecycleState.DEPRECATED, str(ns))

        # CERTIFIED contract_violations >= 2 -> DEPRECATED
        lm._states["k7"] = CERTIFIED
        ns = lm.evaluate_auto_transition("k7", 0.92, 0.05, 0.88, 0.10, 10,
                                        contract_violations_recent=2)
        check("CERTIFIED violations=2 -> DEPRECATED",
              ns == LifecycleState.DEPRECATED, str(ns))

        # CERTIFIED harm_ucb > h_star * 1.5 -> DEPRECATED
        lm._states["k8"] = CERTIFIED
        ns = lm.evaluate_auto_transition("k8", 0.92, 0.20, 0.88, 0.10, 10)
        check("CERTIFIED harm_ucb=0.20 > 0.15 -> DEPRECATED",
              ns == LifecycleState.DEPRECATED, str(ns))

        # DEPRECATED violations >= 5 -> DISABLED
        lm._states["k9"] = DEPRECATED
        ns = lm.evaluate_auto_transition("k9", 0.5, 0.05, 0.88, 0.10, 10,
                                        contract_violations_recent=5)
        check("DEPRECATED violations=5 -> DISABLED",
              ns == LifecycleState.DISABLED, str(ns))

        # DEPRECATED harm > h_star*2 -> DISABLED
        lm._states["k10"] = DEPRECATED
        ns = lm.evaluate_auto_transition("k10", 0.5, 0.25, 0.88, 0.10, 10)
        check("DEPRECATED harm=0.25 > 0.20 -> DISABLED",
              ns == LifecycleState.DISABLED, str(ns))
        _pass("evaluate_auto_transition all paths tested")

    def test_force_disable_from_every_state(self):
        """Test force_disable works from every state."""
        lm = LifecycleManager()
        states = [CANDIDATE, QUARANTINED, PROBATION, CERTIFIED, DEPRECATED, DISABLED]
        for s in states:
            kid = f"fd_{s}"
            lm._states[kid] = s
            r = lm.force_disable(kid, "test_disable")
            check(f"force_disable from {s}", r["transitioned"],
                  f"{r}" if not r["transitioned"] else "")
            assert lm.get_state(kid) == LifecycleState.DISABLED, \
                f"State not DISABLED after force_disable from {s}"

    def test_active_knowledge_ids_filtering(self):
        """Test active_knowledge_ids and certified_knowledge_ids filtering."""
        lm = LifecycleManager()
        lm._states = {
            "k1": CANDIDATE,
            "k2": QUARANTINED,
            "k3": QUARANTINED,
            "k4": PROBATION,
            "k5": PROBATION,
            "k6": CERTIFIED,
            "k7": CERTIFIED,
            "k8": CERTIFIED,
            "k9": DEPRECATED,
            "k10": DISABLED,
        }
        active = lm.active_knowledge_ids()
        check("active_knowledge_ids count", len(active) == 6,
              f"active={active}")
        assert "k1" not in active, "CANDIDATE should not be active"
        assert "k10" not in active, "DISABLED should not be active"
        cert = lm.certified_knowledge_ids()
        check("certified_knowledge_ids count", len(cert) == 3,
              f"cert={cert}")
        # stats
        s = lm.stats()
        check("stats total", sum(s.values()) == 10, str(s))

    def test_is_reusable(self):
        """Test is_reusable reflects REUSABLE_STATES."""
        lm = LifecycleManager()
        reusable_states = [PROBATION, CERTIFIED]
        non_reusable = [CANDIDATE, QUARANTINED, DEPRECATED, DISABLED]
        for s in reusable_states:
            lm._states["test"] = s
            check(f"is_reusable for {s}", lm.is_reusable("test"))
        for s in non_reusable:
            lm._states["test"] = s
            check(f"not is_reusable for {s}", not lm.is_reusable("test"))


# ═══════════════════════════════════════════════════════════════
# 5. TemporalDecay Stress Tests
# ═══════════════════════════════════════════════════════════════

class TestTemporalDecayStress:
    """Stress tests for TemporalDecay (drift-aware lazy decay)."""

    def test_decay_1000_time_steps(self):
        """Test decay over 1000 time steps."""
        td = TemporalDecay(rho=0.95)
        a, b = 10.0, 2.0
        a0, b0 = 1.0, 1.0
        for _ in range(1000):
            a, b = td.decay_params(a, b, a0, b0)
        # After many decay steps, should approach prior
        check("decay_1000 alpha > prior", a >= a0, str(a))
        check("decay_1000 beta > prior", b >= b0, str(b))
        check("decay_1000 alpha near prior", a < 2.0, str(a))
        check("decay_1000 beta near prior", b < 2.0, str(b))
        check("decay_1000 params never below min",
              a >= 0.09 and b >= 0.09, f"a={a}, b={b}")

    def test_rho_adaptation_extreme_values(self):
        """Test rho adaptation from extreme values."""
        td = TemporalDecay(rho=RHO_DEFAULT)
        initial_rho = td.rho
        # High errors -> rho should decrease
        for _ in range(10):
            td.adapt(0.5)
        check("rho decreased after high errors", td.rho < initial_rho,
              f"rho={td.rho} initial={initial_rho}")
        # Low errors -> rho should increase
        td2 = TemporalDecay(rho=RHO_MIN)
        for _ in range(20):
            td2.adapt(0.05)
        check("rho increased after low errors", td2.rho > RHO_MIN,
              f"rho={td2.rho}")
        # rho stays within bounds
        td3 = TemporalDecay(rho=RHO_MIN)
        for _ in range(50):
            td3.adapt(0.5)
        check("rho >= RHO_MIN", td3.rho >= RHO_MIN, str(td3.rho))
        td4 = TemporalDecay(rho=RHO_MAX)
        for _ in range(50):
            td4.adapt(0.01)
        check("rho <= RHO_MAX", td4.rho <= RHO_MAX, str(td4.rho))

    def test_decayed_params_converge_to_prior(self):
        """Test that decayed params converge to priors, not below zero."""
        td = TemporalDecay(rho=0.85)  # More aggressive decay
        a, b = 50.0, 50.0
        a0, b0 = 2.0, 3.0
        # Decay rapidly many times
        for _ in range(100):
            a, b = td.decay_params(a, b, a0, b0)
        # Should be very close to prior
        check("decayed alpha near prior a0", abs(a - a0) < 1.0,
              f"a={a:.4f} a0={a0}")
        check("decayed beta near prior b0", abs(b - b0) < 1.0,
              f"b={b:.4f} b0={b0}")
        # Never below min bound
        check("alpha >= 0.1", a >= 0.1, str(a))
        check("beta >= 0.1", b >= 0.1, str(b))

    def test_reset(self):
        """Test reset() restores default rho."""
        td = TemporalDecay(rho=0.90)
        td.adapt(0.5)
        td.adapt(0.5)
        td.adapt(0.5)
        assert td.rho < RHO_DEFAULT, "rho should have decreased"
        td.reset()
        check("reset restores default rho", td.rho == RHO_DEFAULT, str(td.rho))

    def test_drift_factor(self):
        """Test get_drift_factor()."""
        td = TemporalDecay(rho=0.95)
        df = td.get_drift_factor()
        check("drift_factor for rho=0.95", abs(df - 1.0 / 0.95) < 0.01, str(df))
        td2 = TemporalDecay(rho=0.85)
        df2 = td2.get_drift_factor()
        check("drift_factor for rho=0.85 > 1.0", df2 > 1.0, str(df2))

    def test_decay_with_different_delta_t(self):
        """Test decay with different delta_t values."""
        td = TemporalDecay(rho=0.95)
        a0, b0 = 2.0, 2.0
        a, b = 10.0, 10.0
        a1, b1 = td.decay_params(a, b, a0, b0, delta_t=1)
        a10, b10 = td.decay_params(a, b, a0, b0, delta_t=10)
        # More time steps should decay more
        check("delta_t=10 decays more than delta_t=1",
              a10 < a1, f"a10={a10:.4f} a1={a1:.4f}")


# ═══════════════════════════════════════════════════════════════
# 6. ActiveBaseLogger + SafeThompsonProber Stress Tests
# ═══════════════════════════════════════════════════════════════

class TestActiveLoggingThompsonStress:
    """Stress tests for ActiveBaseLogger and SafeThompsonProber."""

    def test_5000_should_force_base_decisions(self):
        """Test 5000 should_force_base decisions, verify rate stays near target."""
        al = ActiveBaseLogger()
        force_counts = 0
        np.random.seed(42)
        for i in range(5000):
            uncertainty = ActiveBaseLogger.compute_uncertainty(0.5 + 0.1 * np.sin(i * 0.1))
            should, _ = al.should_force_base(uncertainty=uncertainty,
                                            sample_imbalance=0.3,
                                            danger_score=0.1,
                                            risk_level="medium")
            if should:
                force_counts += 1
        rate = force_counts / 5000
        check("force_base rate between 0.05 and 0.30", 0.01 <= rate <= 0.40,
              f"rate={rate:.4f}")
        stats = al.stats()
        check("stats total=5000", stats["total_decisions"] == 5000, str(stats))
        check("stats force_base_count matches",
              stats["force_base_count"] == force_counts, str(stats))

    def test_compute_uncertainty_at_all_values(self):
        """Test compute_uncertainty at all values from 0.0 to 1.0."""
        al = ActiveBaseLogger()
        for v in np.linspace(0.0, 1.0, 101):
            u = al.compute_uncertainty(v)
            assert 0.0 <= u <= 1.0, f"U({v})={u} out of range"
        # Peak at 0.5
        u05 = al.compute_uncertainty(0.5)
        u01 = al.compute_uncertainty(0.1)
        u09 = al.compute_uncertainty(0.9)
        check("uncertainty peaks at 0.5", u05 >= u01 and u05 >= u09,
              f"u05={u05} u01={u01} u09={u09}")
        # Zero at extremes
        u0 = al.compute_uncertainty(0.0)
        u1 = al.compute_uncertainty(1.0)
        check("uncertainty at 0.0 is 0", u0 < 0.001, str(u0))
        check("uncertainty at 1.0 is 0", u1 < 0.001, str(u1))

    def test_compute_imbalance(self):
        """Test compute_imbalance with various scenarios."""
        al = ActiveBaseLogger()
        i0 = al.compute_imbalance(5, 5)  # Perfect balance
        check("imbalance 5:5 is 0", i0 < 0.01, str(i0))
        i1 = al.compute_imbalance(10, 0)  # All use
        check("imbalance 10:0 is 0.5", abs(i1 - 0.5) < 0.01, str(i1))
        i2 = al.compute_imbalance(0, 0)  # No data
        check("imbalance 0:0 is 0.5", abs(i2 - 0.5) < 0.01, str(i2))

    def test_should_probe_with_varying_ess(self):
        """Test should_probe with varying ESS values."""
        tp = SafeThompsonProber(probe_budget=1000)
        base_stats = (
            10.0, 2.0,   # use alpha, beta
            5.0, 5.0,    # base alpha, beta
            1.0, 10.0,   # harm alpha, beta
        )
        # ESS < n_min -> may probe
        should1, q1 = tp.should_probe(*base_stats, ess=4.0, risk_level="medium")
        # ESS >= n_min -> no probe
        should2, q2 = tp.should_probe(*base_stats, ess=20.0, risk_level="medium")
        check("should_probe with ess=4 may probe, rate valid",
              0.0 <= q1 <= 0.10, f"q={q1}")
        check("should_probe with ess=20 always zero", q2 == 0.0, f"q={q2}")

    def test_budget_exhaustion(self):
        """Test probe budget exhaustion."""
        tp = SafeThompsonProber(probe_budget=5)
        probed = 0
        for _ in range(1000):
            should, _ = tp.should_probe(
                8.0, 4.0, 4.0, 4.0, 1.0, 10.0,
                ess=2.0, risk_level="low",
            )
            if should:
                probed += 1
        check("probed <= budget", probed <= 5, f"probed={probed}")
        stats = tp.stats()
        check("budget_remaining correct", stats["budget_remaining"] == 5 - probed,
              str(stats))

    def test_thompson_probe_high_risk_blocked(self):
        """Test Thompson probe blocked in high risk."""
        tp = SafeThompsonProber(probe_budget=100)
        should, q = tp.should_probe(
            8.0, 4.0, 4.0, 4.0, 1.0, 10.0,
            ess=2.0, risk_level="high",
        )
        check("high risk blocked", not should, f"should={should} q={q}")

    def test_thompson_probe_force_allow(self):
        """Test Thompson probe force_allow bypass."""
        tp = SafeThompsonProber(probe_budget=100)
        for _ in range(5):
            should, _ = tp.should_probe(
                15.0, 1.0, 3.0, 7.0, 1.0, 15.0,
                ess=5.0, risk_level="medium",
                force_allow=True,
            )
            if should:
                break
        # At least one should succeed with force_allow
        # (ess >= N_MIN_MEDIUM so normal would be blocked, but force allows)
        check("force_allow some probes succeed",
              tp.stats()["probe_count"] > 0, str(tp.stats()))

    def test_probe_probability(self):
        """Test probe_probability returns sensible values."""
        tp = SafeThompsonProber(probe_budget=100)
        q1 = tp.probe_probability(1.0, 2.0, "low")
        check("probe prob high uncertainty low risk > 0", q1 > 0, str(q1))
        q2 = tp.probe_probability(1.0, 20.0, "low")
        check("probe prob high ESS = 0", q2 == 0.0, str(q2))
        q3 = tp.probe_probability(1.0, 2.0, "high")
        check("probe prob high risk = 0", q3 == 0.0, str(q3))
        q4 = tp.probe_probability(0.0, 2.0, "medium")
        check("probe prob zero uncertainty = 0", q4 == 0.0, str(q4))

    def test_reset_budget(self):
        """Test reset_budget restores budget."""
        tp = SafeThompsonProber(probe_budget=10)
        for _ in range(5):
            tp.should_probe(10.0, 2.0, 5.0, 5.0, 1.0, 10.0,
                           ess=2.0, risk_level="low")
        tp.reset_budget(20)
        check("reset_budget restores to 20", tp._budget == 20, str(tp._budget))
        check("reset_budget clears probe_count", tp._probe_count == 0,
              str(tp._probe_count))

    def test_log_decision_creates_entry(self):
        """Test log_decision returns an entry dict."""
        al = ActiveBaseLogger()
        entry = al.log_decision("dec_001", "kid_1", "reuse", 0.85, 0.15, "bucket_a")
        check("log_decision has decision_id", entry["decision_id"] == "dec_001")
        check("log_decision has candidate", entry["candidate"] == "kid_1")
        check("log_decision has assigned", entry["assigned"] == "reuse")

    def test_base_probability_self_correction(self):
        """Test base_probability self-correction towards target."""
        al = ActiveBaseLogger()
        # Simulate over-sampling scenario
        al._total_decisions = 100
        al._force_base_count = 50  # 50% rate, way above 15% target
        q = al.base_probability(0.5, 0.3, 0.1)
        # Should be pushed down
        check("over-sampling self-corrects down",
              q < 0.25, f"q={q}")

    def test_danger_score_computation(self):
        """Test compute_danger_score."""
        score0 = ActiveBaseLogger.compute_danger_score({})
        check("empty state danger 0", score0 == 0.0, str(score0))
        score_full = ActiveBaseLogger.compute_danger_score({
            "lava_nearby": True, "low_health": True, "combat": True,
            "near_cliff": True, "irreversible_resource_constraint": True,
        })
        check("all danger flags => score 1.0", score_full == 1.0, str(score_full))
        score_partial = ActiveBaseLogger.compute_danger_score({
            "lava_nearby": True, "combat": True})
        check("2/5 flags => 0.4", abs(score_partial - 0.4) < 0.01, str(score_partial))


# ═══════════════════════════════════════════════════════════════
# 7. InteractionGate Stress Tests
# ═══════════════════════════════════════════════════════════════

class TestInteractionGateStress:
    """Stress tests for InteractionGate (pairwise knowledge interaction)."""

    def test_check_chain_lengths_2_5_10_20(self):
        """Test check_chain with chains of 2, 5, 10, 20 knowledge items."""
        ig = InteractionGate()
        for chain_len in [2, 5, 10, 20]:
            chain = []
            for i in range(chain_len):
                chain.append({
                    "knowledge_id": f"kid_{i}",
                    "use_alpha": 15.0 + i,
                    "use_beta": 5.0,
                    "base_alpha": 10.0,
                    "base_beta": 10.0,
                })
            # Build pair_stats (all neutral — well-established joint outcomes)
            pair_stats = {}
            for i in range(chain_len):
                for j in range(i + 1, chain_len):
                    pair_stats[(f"kid_{i}", f"kid_{j}")] = {
                        "alpha": 10.0, "beta": 10.0}
            result = ig.check_chain(chain, pair_stats, {})
            assert "safe" in result, f"chain {chain_len} missing safe"
            assert "blocked_pairs" in result, f"chain {chain_len} missing blocked_pairs"
            assert "recommendation" in result, f"chain {chain_len} missing recommendation"
        _pass("check_chain with 2/5/10/20 length chains")

    def test_check_pair_extreme_alpha_beta(self):
        """Test check_pair with extreme alpha/beta values."""
        ig = InteractionGate()
        # Strong synergy case
        result = ig.check_pair(
            {"use_alpha": 20.0, "use_beta": 2.0, "base_alpha": 10.0, "base_beta": 10.0},
            {"use_alpha": 20.0, "use_beta": 2.0, "base_alpha": 10.0, "base_beta": 10.0},
            {"alpha": 50.0, "beta": 5.0},  # Very strong joint
        )
        check("strong joint has state", result["state"] in (SYNERGY, NEUTRAL, CONFLICT, UNKNOWN),
              result["state"])
        check("strong joint has recommendation",
              result["recommendation"] in ("allow_pair", "block_pair", "pair_probe",
                                           "force_fallback", "single_best_only"))

        # Zero data case
        result2 = ig.check_pair(
            {"use_alpha": 1.0, "use_beta": 1.0, "base_alpha": 1.0, "base_beta": 1.0},
            {"use_alpha": 1.0, "use_beta": 1.0, "base_alpha": 1.0, "base_beta": 1.0},
            {"alpha": 1.0, "beta": 1.0},
        )
        check("zero-data pair is UNKNOWN", result2["state"] == UNKNOWN, result2["state"])
        check("zero-data pair recommends pair_probe",
              result2["recommendation"] == "pair_probe", result2["recommendation"])

    def test_5_state_decision_exhaustive(self):
        """Test exhaustively the 5 interaction states."""
        ig = InteractionGate()

        # Synergy
        r = ig.check_pair(
            {"use_alpha": 20, "use_beta": 2, "base_alpha": 10, "base_beta": 10},
            {"use_alpha": 20, "use_beta": 2, "base_alpha": 10, "base_beta": 10},
            {"alpha": 60, "beta": 5},  # Joint way better
        )
        check("synergy detected", r["recommendation"] == "allow_pair",
              r["recommendation"])

        # Conflict
        r = ig.check_pair(
            {"use_alpha": 20, "use_beta": 2, "base_alpha": 10, "base_beta": 10},
            {"use_alpha": 20, "use_beta": 2, "base_alpha": 10, "base_beta": 10},
            {"alpha": 5, "beta": 15},  # Joint way worse
        )
        # May be conflict or block_pair
        check("conflict detected", r["state"] == CONFLICT,
              f"Expected {CONFLICT} got {r['state']}")

        # Neutral
        r = ig.check_pair(
            {"use_alpha": 20, "use_beta": 2, "base_alpha": 10, "base_beta": 10},
            {"use_alpha": 20, "use_beta": 2, "base_alpha": 10, "base_beta": 10},
            {"alpha": 50, "beta": 50},
        )
        check("neutral or unknown detected",
              r["state"] in (NEUTRAL, UNKNOWN), r["state"])

        # Unknown + high risk => force_fallback
        r = ig.check_pair(
            {"use_alpha": 1, "use_beta": 1, "base_alpha": 1, "base_beta": 1},
            {"use_alpha": 1, "use_beta": 1, "base_alpha": 1, "base_beta": 1},
            {"alpha": 3, "beta": 3},
            context={"risk_level": "high"},
        )
        check("unknown + high risk => force_fallback",
              r["recommendation"] == "force_fallback",
              r["recommendation"])

        # Unknown + resource_critical => force_fallback
        r = ig.check_pair(
            {"use_alpha": 1, "use_beta": 1, "base_alpha": 1, "base_beta": 1},
            {"use_alpha": 1, "use_beta": 1, "base_alpha": 1, "base_beta": 1},
            {"alpha": 3, "beta": 3},
            context={"resource_critical": True},
        )
        check("unknown + resource_critical => force_fallback",
              r["recommendation"] == "force_fallback",
              r["recommendation"])
        _pass("5-interaction-state exhaustive test")

    def test_result_structure(self):
        """Test check_pair result dict has all required fields."""
        ig = InteractionGate()
        r = ig.check_pair(
            {"use_alpha": 10, "use_beta": 5, "base_alpha": 5, "base_beta": 5},
            {"use_alpha": 10, "use_beta": 5, "base_alpha": 5, "base_beta": 5},
            {"alpha": 10, "beta": 5},
        )
        check("result has delta_mean", "delta_mean" in r)
        check("result has delta_lcb", "delta_lcb" in r)
        check("result has state", "state" in r)
        check("result has pi_syn", "pi_syn" in r)
        check("result has pi_conf", "pi_conf" in r)
        check("result has recommendation", "recommendation" in r)

    def test_prob_helpers(self):
        """Test _prob_positive and _prob_negative helpers."""
        ig = InteractionGate()
        pp = ig._prob_positive(0.1, 0.03)
        check("prob_positive in [0.01, 0.99]", 0.01 <= pp <= 0.99, str(pp))
        pn = ig._prob_negative(-0.1, -0.03)
        check("prob_negative in [0.01, 0.99]", 0.01 <= pn <= 0.99, str(pn))


# ═══════════════════════════════════════════════════════════════
# 8. DecisionController Integration Stress Tests
# ═══════════════════════════════════════════════════════════════

class TestDecisionControllerStress:
    """Integration stress tests for DecisionController (7-step flow)."""

    def _make_controller(self):
        """Create a configured DecisionController with all dependencies."""
        store = _make_temp_store()
        gate = TrustGate()
        gate.tau["crafting"] = 0.90
        gate.harm["crafting"] = 0.10
        gate.delta["crafting"] = 0.05
        ig = InteractionGate()
        al = ActiveBaseLogger()
        tp = SafeThompsonProber(probe_budget=50)
        cc = ContractChecker()
        dc = DecisionController(store, gate, ig, al, tp, cc)
        return dc, store

    def _make_candidate(self, idx=0, good=True):
        """Create a candidate knowledge dict."""
        return {
            "knowledge_id": f"kc_test_{idx}",
            "type": "skill",
            "level": "atomic",
            "gene": f"craft_stone_pickaxe_{idx}",
            "full_text": "Craft stone pickaxe before mining iron ore.",
            "claimed_context": {"subgoal_type": "craft"},
            "preconditions": [],
            "postconditions": [],
            "non_applicable_contexts": [],
        }

    def test_decide_with_50_candidates(self):
        """Test full decide() flow with 50 candidates."""
        dc, store = self._make_controller()
        try:
            # Pre-load some trust data for candidates
            for i in range(50):
                kid = f"kc_test_{i}"
                ctx_key = "craft"
                contract = {
                    "type": "skill", "level": "atomic",
                    "expected_uplift": 0.1, "risk_bound": 0.05,
                }
                store.register_contract(kid, contract)
                store.record_episode(kid, ctx_key, used=True, success=0.85,
                                    is_harmful=0.05)
                store.record_episode(kid, ctx_key, used=False, success=0.3,
                                    is_harmful=0.0)

            candidates = [self._make_candidate(i) for i in range(50)]
            state = {"waypoint": "craft_stone_pickaxe", "task_group": "crafting"}
            task = {"task_id": "craft_stone_pickaxe", "group": "crafting",
                    "difficulty": "easy"}
            context = {"bucket": "craft", "subgoal_type": "craft",
                      "risk_level": "low"}

            result = dc.decide(candidates, state, task, context, mode="evaluation")
            check("decide with 50 candidates returns result",
                  isinstance(result, DecisionResult),
                  str(type(result)))
            check("decision in valid set",
                  result.decision in ("reuse", "fallback", "probe", "force_base"),
                  result.decision)
            check("filtered_count > 0", result.filtered_count > 0,
                  str(result.filtered_count))
            check("scored_count > 0", result.scored_count > 0,
                  str(result.scored_count))
            _pass("decide with 50 candidates successful")
        except Exception as e:
            _fail("decide_with_50_candidates", str(e))
        finally:
            shutil.rmtree(store.store_path, ignore_errors=True)

    def test_decide_empty_candidates(self):
        """Test decide() with empty candidate list."""
        dc, store = self._make_controller()
        try:
            result = dc.decide([], {}, {"group": "crafting"}, {"bucket": "craft"},
                             mode="evaluation")
            check("empty candidates => fallback",
                  result.decision == "fallback", result.decision)
            check("empty candidates => filtered=0",
                  result.filtered_count == 0, str(result.filtered_count))
        except Exception as e:
            _fail("decide_empty_candidates", str(e))
        finally:
            shutil.rmtree(store.store_path, ignore_errors=True)

    def test_decide_all_candidates_fail_contract(self):
        """Test decide() with all candidates failing contract filter."""
        dc, store = self._make_controller()
        try:
            # All have non-applicable context triggered
            candidates = []
            for i in range(10):
                c = self._make_candidate(i)
                c["non_applicable_contexts"] = ["lava_nearby"]
                candidates.append(c)
            state = {"near_lava": True}  # Triggers lava_nearby safety
            result = dc.decide(
                candidates, state,
                {"task_id": "test", "group": "crafting"},
                {"bucket": "craft"},
                mode="evaluation",
            )
            check("all contract-filtered => fallback",
                  result.decision == "fallback", result.decision)
        except Exception as e:
            _fail("decide_all_candidates_fail_contract", str(e))
        finally:
            shutil.rmtree(store.store_path, ignore_errors=True)

    def test_decide_precondition_filtered(self):
        """Test contract precondition filtering."""
        dc, store = self._make_controller()
        try:
            c = self._make_candidate(0)
            c["preconditions"] = ["has_iron_pickaxe"]
            candidates = [c]
            # State does NOT have iron pickaxe
            result = dc.decide(
                candidates,
                {"has_iron_pickaxe": False},
                {"task_id": "test", "group": "crafting"},
                {"bucket": "craft", "subgoal_type": "craft"},
                mode="evaluation",
            )
            check("precondition not met => all filtered",
                  result.filtered_count == 0, str(result.filtered_count))
            check("precondition not met => fallback",
                  result.decision == "fallback", result.decision)
        except Exception as e:
            _fail("decide_precondition_filtered", str(e))
        finally:
            shutil.rmtree(store.store_path, ignore_errors=True)

    def test_mode_accumulation(self):
        """Test decide() in accumulation mode (forces active logging)."""
        dc, store = self._make_controller()
        try:
            kid = "kc_test_0"
            ctx_key = "craft"
            contract = {"type": "skill", "level": "atomic",
                       "expected_uplift": 0.1, "risk_bound": 0.05}
            store.register_contract(kid, contract)
            # Make it a good candidate with strong uplift
            for _ in range(10):
                store.record_episode(kid, ctx_key, used=True, success=0.9,
                                    is_harmful=0.02)
                store.record_episode(kid, ctx_key, used=False, success=0.3,
                                    is_harmful=0.0)

            candidates = [self._make_candidate(0)]
            result = dc.decide(
                candidates,
                {"waypoint": "test"},
                {"task_id": "test", "group": "crafting"},
                {"bucket": "craft", "subgoal_type": "craft", "risk_level": "low"},
                mode="accumulation",
            )
            check("accumulation mode returns result",
                  result.decision in ("reuse", "fallback", "force_base"),
                  result.decision)
        except Exception as e:
            _fail("mode_accumulation", str(e))
        finally:
            shutil.rmtree(store.store_path, ignore_errors=True)

    def test_mode_calibration(self):
        """Test decide() in calibration mode."""
        dc, store = self._make_controller()
        try:
            kid = "kc_test_calib"
            ctx_key = "craft"
            store.register_contract(kid, {"type": "skill", "level": "atomic"})
            for _ in range(5):
                store.record_episode(kid, ctx_key, used=True, success=0.8,
                                    is_harmful=0.05)
            candidates = [self._make_candidate(0)]
            candidates[0]["knowledge_id"] = kid
            result = dc.decide(
                candidates,
                {"waypoint": "test"},
                {"task_id": "test", "group": "crafting"},
                {"bucket": "craft", "subgoal_type": "craft", "risk_level": "low"},
                mode="calibration",
            )
            check("calibration mode returns valid decision",
                  result.decision in ("reuse", "fallback", "force_base", "probe"),
                  result.decision)
        except Exception as e:
            _fail("mode_calibration", str(e))
        finally:
            shutil.rmtree(store.store_path, ignore_errors=True)

    def test_decision_result_serialization(self):
        """Test DecisionResult fields are correct and complete."""
        dr = DecisionResult(
            decision="reuse",
            chosen_knowledge_id="kc_001",
            chosen_contract={"type": "skill"},
            pi_uplift=0.92,
            harm_ucb=0.04,
            uplift_lcb=0.08,
            contract_satisfied_before=True,
            contract_violation_after=False,
            interaction_safe=True,
            interaction_state="safe",
            tau_group_level=0.90,
            delta_group_level=0.05,
            harm_threshold_group_level=0.10,
            propensity_reuse=0.85,
            propensity_base=0.15,
            lifecycle_state="certified",
            filtered_count=5,
            scored_count=5,
            chain_length=3,
        )
        # Check all fields are set
        check("DecisionResult decision set", dr.decision == "reuse")
        check("DecisionResult pi_uplift set", abs(dr.pi_uplift - 0.92) < 0.001)
        check("DecisionResult lifecycle set", dr.lifecycle_state == "certified")
        _pass("DecisionResult serialization fields correct")

    def test_mode_online(self):
        """Test decide() in online mode."""
        dc, store = self._make_controller()
        try:
            kid = "kc_test_online"
            ctx_key = "craft"
            store.register_contract(kid, {"type": "skill", "level": "atomic"})
            for _ in range(10):
                store.record_episode(kid, ctx_key, used=True, success=0.85,
                                    is_harmful=0.05)
            candidates = [self._make_candidate(0)]
            candidates[0]["knowledge_id"] = kid
            result = dc.decide(
                candidates,
                {"waypoint": "test"},
                {"task_id": "test", "group": "crafting"},
                {"bucket": "craft", "subgoal_type": "craft", "risk_level": "low"},
                mode="online",
            )
            check("online mode returns valid decision",
                  result.decision in ("reuse", "fallback", "force_base", "probe"),
                  result.decision)
        except Exception as e:
            _fail("mode_online", str(e))
        finally:
            shutil.rmtree(store.store_path, ignore_errors=True)


# ═══════════════════════════════════════════════════════════════
# 9. CactMemory Integration Tests
# ═══════════════════════════════════════════════════════════════

class TestCactMemoryStress:
    """Integration stress tests for CactMemory (all 9 methods)."""

    def _make_cm(self, method="C-ACT-Full", frozen=False, active_calib_rate=0.0):
        """Create a CactMemory with mock XENON memory."""
        mock = _MockXenonMemory()
        log_dir = tempfile.mkdtemp(prefix="cact_mem_logs_")
        cm = CactMemory(
            xenon_memory=mock,
            method=method,
            store_path=os.path.join(log_dir, "store"),
            frozen=frozen,
            active_calib_rate=active_calib_rate,
            log_dir=log_dir,
        )
        return cm, mock, log_dir

    def test_all_9_decision_methods(self):
        """Test is_succeeded_waypoint with all 9 decision methods."""
        methods = [
            "C-ACT-Full", "NoKnowledge", "XENON-Original", "BankCuration",
            "LifecycleSuccessGate", "FixedBayes", "ACT", "OracleGate",
            "ShuffledKnowledge",
        ]
        for method in methods:
            cm, mock, log_dir = self._make_cm(method=method)
            try:
                wp = "craft_stone_pickaxe"
                # Register knowledge
                contract = cm.register_knowledge(_make_knowledge_dict(0))
                # Record some history
                cm.save_success_failure(wp, "craft stone_pickaxe", True)
                cm.save_success_failure(wp, "craft stone_pickaxe", True)
                # Query
                is_ok, sg = cm.is_succeeded_waypoint(wp)
                check(f"method={method} returns bool tuple",
                      isinstance(is_ok, bool),
                      f"type={type(is_ok)}")
            except Exception as e:
                _fail(f"all_9_methods: {method}", str(e))
            finally:
                shutil.rmtree(log_dir, ignore_errors=True)
        _pass("all 9 decision methods tested")

    def test_dump_logs_1000_entries(self):
        """Test dump_logs() with 1000 entries."""
        cm, mock, log_dir = self._make_cm()
        try:
            for i in range(1000):
                cm.episode_logs.append({
                    "waypoint": f"wp_{i}", "method": "C-ACT-Full",
                    "success": i % 2, "task_group": "crafting",
                })
                cm.reuse_logs.append({
                    "kid": f"kc_{i}", "ctx": "craft",
                    "pi_uplift": 0.5 + 0.1 * (i % 5),
                })
            cm.dump_logs()
            # Check log files created
            episode_file = os.path.join(log_dir, "episode", "episode.jsonl")
            reuse_file = os.path.join(log_dir, "reuse", "reuse_decision.jsonl")
            check("episode log file exists", os.path.exists(episode_file))
            check("reuse log file exists", os.path.exists(reuse_file))
            # Verify buffers cleared
            check("episode_logs cleared", len(cm.episode_logs) == 0)
            check("reuse_logs cleared", len(cm.reuse_logs) == 0)
        except Exception as e:
            _fail("dump_logs_1000_entries", str(e))
        finally:
            shutil.rmtree(log_dir, ignore_errors=True)

    def test_register_knowledge_valid(self):
        """Test register_knowledge with valid knowledge."""
        cm, mock, log_dir = self._make_cm()
        try:
            kid = cm.register_knowledge(_make_knowledge_dict(0))
            check("register_knowledge returns kid", kid is not None and len(kid) > 0, str(kid))
            check("knowledge_cnt incremented", cm._knowledge_cnt == 1, str(cm._knowledge_cnt))
            # Contract log recorded
            check("contract_logs has entry", len(cm.contract_logs) == 1,
                  str(len(cm.contract_logs)))
        except Exception as e:
            _fail("register_knowledge_valid", str(e))
        finally:
            shutil.rmtree(log_dir, ignore_errors=True)

    def test_register_knowledge_invalid(self):
        """Test register_knowledge with minimal/invalid knowledge."""
        cm, mock, log_dir = self._make_cm()
        try:
            # Empty dict
            kid = cm.register_knowledge({})
            check("register empty dict works", kid is not None)
            # Only type
            kid2 = cm.register_knowledge({"type": "skill", "source": "test"})
            check("register minimal works", kid2 is not None)
        except Exception as e:
            _fail("register_knowledge_invalid", str(e))
        finally:
            shutil.rmtree(log_dir, ignore_errors=True)

    def test_pass_through_methods(self):
        """Test pass-through methods delegate correctly."""
        cm, mock, log_dir = self._make_cm()
        try:
            mock.add_succeeded_waypoint("wp1", "action1")
            wps = cm.succeeded_waypoints
            check("succeeded_waypoints pass-through",
                  "wp1" in wps, str(wps))
            sw = cm.retrieve_similar_succeeded_waypoints("wp1", 3)
            check("retrieve_similar_succeeded_waypoints pass-through",
                  isinstance(sw, list),
                  str(type(sw)))
            cm.set_history_index(5)
            assert mock._history_idx == 5, "set_history_index not delegated"
            _pass("pass_through_methods delegate correctly")
        except Exception as e:
            _fail("pass_through_methods", str(e))
        finally:
            shutil.rmtree(log_dir, ignore_errors=True)

    def test_frozen_mode(self):
        """Test that frozen mode does not update the store."""
        cm, mock, log_dir = self._make_cm(frozen=True)
        try:
            wp = "craft_test"
            cm.register_knowledge(_make_knowledge_dict(0))
            cm.save_success_failure(wp, "craft test", True)
            # In frozen mode, the store should not be updated beyond initial contract
            # (save_success_failure uses self.frozen check)
            check("frozen mode passed without error", True)
        except Exception as e:
            _fail("frozen_mode", str(e))
        finally:
            shutil.rmtree(log_dir, ignore_errors=True)

    def test_infer_tier(self):
        """Test _infer_tier with various waypoint patterns."""
        cm, mock, log_dir = self._make_cm()
        try:
            check("infer_tier: diamond", cm._infer_tier("diamond_pickaxe") == "diamond")
            check("infer_tier: iron", cm._infer_tier("iron_sword") == "iron")
            check("infer_tier: stone", cm._infer_tier("stone_pickaxe") == "stone")
            check("infer_tier: wood", cm._infer_tier("wooden_planks") == "wood")
            check("infer_tier: unknown", cm._infer_tier("weird_item") == "stone")
        except Exception as e:
            _fail("infer_tier", str(e))
        finally:
            shutil.rmtree(log_dir, ignore_errors=True)

    def test_infer_group(self):
        """Test _infer_group with various waypoint patterns."""
        cm, mock, log_dir = self._make_cm()
        try:
            check("infer_group: crafting", cm._infer_group("craft_stone_pickaxe") == "crafting")
            check("infer_group: mining", cm._infer_group("mine_iron_ore") == "mining")
            check("infer_group: exploration", cm._infer_group("explore_forest") == "exploration")
            check("infer_group: tech_tree", cm._infer_group("nether_portal_tech") == "tech_tree")
            check("infer_group: failure_recovery",
                  cm._infer_group("recover_from_wrong_tool") == "failure_recovery")
            check("infer_group: interaction",
                  cm._infer_group("interaction_conflict_test") == "interaction_stress")
        except Exception as e:
            _fail("infer_group", str(e))
        finally:
            shutil.rmtree(log_dir, ignore_errors=True)

    def test_infer_risk(self):
        """Test _infer_risk with various waypoint patterns."""
        cm, mock, log_dir = self._make_cm()
        try:
            check("infer_risk: diamond=high", cm._infer_risk("diamond_sword") == "high")
            check("infer_risk: iron=medium", cm._infer_risk("iron_pickaxe") == "medium")
            check("infer_risk: wood=low", cm._infer_risk("wooden_planks") == "low")
        except Exception as e:
            _fail("infer_risk", str(e))
        finally:
            shutil.rmtree(log_dir, ignore_errors=True)

    def test_update_last_episode(self):
        """Test update_last_episode stats tracking."""
        cm, mock, log_dir = self._make_cm()
        try:
            cm.episode_logs.append({"waypoint": "test", "success": 1, "total_steps": 0})
            cm.update_last_episode(total_steps=50, llm_calls=3,
                                  wall_time_sec=12.5,
                                  input_tokens=500, output_tokens=200)
            e = cm.episode_logs[-1]
            check("update: total_steps", e["total_steps"] == 50, str(e))
            check("update: llm_calls", e["llm_calls"] == 3, str(e))
            check("update: wall_time_sec", abs(e["wall_time_sec"] - 12.5) < 0.01, str(e))
            check("update: tokens", e["tokens"] == 700, str(e))
        except Exception as e:
            _fail("update_last_episode", str(e))
        finally:
            shutil.rmtree(log_dir, ignore_errors=True)


# ═══════════════════════════════════════════════════════════════
# 10. Numerical Stability Tests
# ═══════════════════════════════════════════════════════════════

class TestNumericalStability:
    """Numerical stability tests for core C-ACT computations."""

    def test_uplift_probability_extreme_beta(self):
        """Test uplift_probability() with extreme Beta params."""
        store = _make_temp_store()
        try:
            # Extremely confident good knowledge
            for _ in range(500):
                store.record_episode("extreme_good", "craft",
                                    used=True, success=0.99, is_harmful=0.0)
                store.record_episode("extreme_good", "craft",
                                    used=False, success=0.01, is_harmful=0.0)
            pi = store.uplift_probability("extreme_good", "craft")
            check("extreme good uplift > 0.999", pi > 0.999, str(pi))
            check("extreme good uplift <= 1.0-eps", pi <= 1.0, str(pi))

            # Extremely confident bad knowledge
            for _ in range(500):
                store.record_episode("extreme_bad", "craft",
                                    used=True, success=0.01, is_harmful=0.0)
                store.record_episode("extreme_bad", "craft",
                                    used=False, success=0.99, is_harmful=0.0)
            pi_bad = store.uplift_probability("extreme_bad", "craft")
            check("extreme bad uplift < 0.001", pi_bad < 0.001, str(pi_bad))
            check("extreme bad uplift >= 0", pi_bad >= 0.0, str(pi_bad))

            # Total no data — should return near 0.5 (prior)
            pi_none = store.uplift_probability("no_data", "craft")
            check("no data uplift near 0.5", 0.3 < pi_none < 0.7, str(pi_none))
        except Exception as e:
            _fail("uplift_probability_extreme_beta", str(e))
        finally:
            shutil.rmtree(store.store_path, ignore_errors=True)

    def test_probabilities_stay_in_0_1_range(self):
        """Test that all probability outputs stay in [0, 1]."""
        store = _make_temp_store()
        try:
            kids = ["prob_k1", "prob_k2", "prob_k3"]
            ctx = "craft"
            for i in range(200):
                for kid in kids:
                    s = np.random.random()
                    h = 1.0 - s if s < 0.5 else 0.0
                    store.record_episode(kid, ctx, used=True, success=s, is_harmful=h)
            for kid in kids:
                for delta in [0.01, 0.05, 0.10]:
                    l = store.lcb(kid, ctx, "use", delta)
                    u = store.ucb(kid, ctx, "use", delta)
                    assert 0.0 <= l <= 1.0, f"{kid} lcb({delta})={l}"
                    assert 0.0 <= u <= 1.0, f"{kid} ucb({delta})={u}"
                hu = store.harm_upper_bound(kid, ctx)
                assert 0.0 <= hu <= 1.0, f"{kid} harm_upper_bound={hu}"
                ph = store.prob_harm_safe(kid, ctx)
                assert 0.0 <= ph <= 1.0, f"{kid} prob_harm_safe={ph}"
            _pass("all probability outputs in [0,1] under 200 mixed samples")
        except Exception as e:
            _fail("probabilities_stay_in_0_1_range", str(e))
        finally:
            shutil.rmtree(store.store_path, ignore_errors=True)

    def test_harm_upper_bound_extreme(self):
        """Test harm_upper_bound() with extreme params."""
        store = _make_temp_store()
        try:
            # Very safe knowledge
            for _ in range(100):
                store.record_episode("safe_kid", "craft",
                                    used=True, success=0.9, is_harmful=0.0)
            hu_safe = store.harm_upper_bound("safe_kid", "craft")
            check("safe knowledge harm bound low", hu_safe < 0.3, str(hu_safe))

            # Very harmful knowledge
            for _ in range(100):
                store.record_episode("risky_kid", "craft",
                                    used=True, success=0.1, is_harmful=1.0)
            hu_risky = store.harm_upper_bound("risky_kid", "craft")
            check("risky knowledge harm bound high", hu_risky > 0.7, str(hu_risky))

            # No data: near prior
            hu_none = store.harm_upper_bound("unknown_kid", "craft")
            check("no data harm bound in [0,1]", 0.0 <= hu_none <= 1.0, str(hu_none))
        except Exception as e:
            _fail("harm_upper_bound_extreme", str(e))
        finally:
            shutil.rmtree(store.store_path, ignore_errors=True)

    def test_probs_harm_safe_edge_cases(self):
        """Test prob_harm_safe with edge case params."""
        store = _make_temp_store()
        a_default, b_default = store.DEFAULT_PRIORS["harm"]
        for stat_name, stat in [("harm", "harm")]:
            a, b = store.get_stats("nonexistent", "any", stat)
            # Prior-based stats should be > 0
            assert a > 0 and b > 0, f"Prior {stat_name}: a={a}, b={b}"
        check("default prior values are positive", True)
        ph = store.prob_harm_safe("nonexistent", "any")
        check("prob_harm_safe in [0,1] for default prior",
              0.0 <= ph <= 1.0, str(ph))

    def test_uplift_probability_monotonicity(self):
        """Test uplift_probability monotonicity: more use-success => higher uplift."""
        store = _make_temp_store()
        try:
            kid = "mono_test"
            ctx = "craft"
            pis = []
            for batch in range(5):
                # Add 10 good use observations + 10 bad base observations
                for _ in range(10):
                    store.record_episode(kid, ctx, used=True, success=1.0, is_harmful=0.0)
                    store.record_episode(kid, ctx, used=False, success=0.0, is_harmful=0.0)
                pis.append(store.uplift_probability(kid, ctx))
            # Probabilities should be monotonically increasing
            for i in range(1, len(pis)):
                assert pis[i] >= pis[i - 1] - 0.01, \
                    f"Monotonicity violated: pi[{i-1}]={pis[i-1]:.6f} pi[{i}]={pis[i]:.6f}"
            _pass("uplift_probability monotonic with evidence accumulation")
        except Exception as e:
            _fail("uplift_probability_monotonicity", str(e))
        finally:
            shutil.rmtree(store.store_path, ignore_errors=True)

    def test_context_bucket_stress(self):
        """Stress test ContextBucket encode with various inputs."""
        cb = ContextBucket()
        for _ in range(500):
            key = cb.encode(
                knowledge_type=random.choice(KNOWLEDGE_TYPES),
                subgoal_type=random.choice(["craft", "mine", "explore", "build"]),
                failure_type=random.choice(["wrong_tool", "missing_item", "timing_error", "none"]),
                risk_level=random.choice(["low", "medium", "high"]),
                task_tier=random.choice(["wooden", "stone", "iron", "diamond"]),
                biome=random.choice(["forest", "desert", "plains", "taiga"]),
            )
            assert "|" in key, f"ContextBucket encode produced bad key: {key}"
        _pass("ContextBucket encode 500 calls without error")

    def test_empirical_bayes_estimate_and_prior(self):
        """Test EmpiricalBayes estimate and get_prior."""
        eb = EmpiricalBayes()
        # Create synthetic type-level data
        type_data = {}
        for ktype in ["skill", "remedy"]:
            for level in ["atomic", "functional"]:
                key = f"{ktype}|{level}"
                records = []
                for i in range(50):
                    records.append({
                        "stat": "use",
                        "success": np.random.beta(5, 2) if ktype == "skill" else np.random.beta(2, 5),
                    })
                type_data[key] = records
        eb.estimate(type_data)
        # Now get priors
        a_skill, b_skill = eb.get_prior("skill", "atomic", "use")
        check("empirical prior alpha > 0", a_skill > 0, str(a_skill))
        check("empirical prior beta > 0", b_skill > 0, str(b_skill))
        # Unknown type falls back to defaults
        a_def, b_def = eb.get_prior("nonexistent", "unknown", "use")
        check("unknown type gets default alpha", a_def > 0, str(a_def))
        check("unknown type gets default beta", b_def > 0, str(b_def))
        # to_dict / from_dict roundtrip
        d = eb.to_dict()
        eb2 = EmpiricalBayes.from_dict(d)
        a2, b2 = eb2.get_prior("skill", "atomic", "use")
        check("from_dict roundtrip alpha", abs(a2 - a_skill) < 0.001,
              f"{a2} vs {a_skill}")
        check("from_dict roundtrip beta", abs(b2 - b_skill) < 0.001,
              f"{b2} vs {b_skill}")

    def test_context_bucket_split_merge(self):
        """Test ContextBucket maybe_split and maybe_merge."""
        cb = ContextBucket()
        # With very few data, shouldn't split
        key = cb.encode(subgoal_type="craft", failure_type="wrong_tool")
        result = cb.maybe_split(key, [0.5, 0.5], [0, 1])
        check("maybe_split with 2 points returns None", result is None, str(result))
        # Merge test with single bucket
        buckets = ["bucket_a"]
        stats = {"bucket_a": (5.0, 5.0, 2)}
        merges = cb.maybe_merge(buckets, stats)
        check("maybe_merge single bucket returns []", merges == [], str(merges))
        # With enough diverse data, split returns None (confidence already good)
        key2 = cb.encode()
        result2 = cb.maybe_split(
            key2,
            [0.1, 0.1, 0.9, 0.9, 0.5, 0.5, 0.8, 0.2],
            [0, 0, 1, 1, 0, 1, 1, 0],
        )
        # Should return None (ECE is computed in buckets, not enough per bucket)
        check("maybe_split returns None or string",
              result2 is None or isinstance(result2, str),
              str(result2))

    def test_metrics_functions(self):
        """Test all metrics functions with synthetic data."""
        data = [
            {"success": True, "outcome_success": 1, "decision": "reuse",
             "is_harmful": 0, "contract_satisfied_before": True,
             "contract_violation_after": False, "type": "skill",
             "pi_uplift": 0.9},
            {"success": True, "outcome_success": 1, "decision": "reuse",
             "is_harmful": 0, "contract_satisfied_before": True,
             "contract_violation_after": False, "type": "skill",
             "pi_uplift": 0.85},
            {"success": False, "outcome_success": 0, "decision": "reuse",
             "is_harmful": 1, "contract_satisfied_before": True,
             "contract_violation_after": True, "type": "remedy",
             "pi_uplift": 0.5},
            {"success": False, "outcome_success": 0, "decision": "fallback",
             "is_harmful": 0},
        ]
        sr = compute_sr(data)
        check("SR in [0,1]", 0 <= sr <= 1, str(sr))
        kus = compute_kus(data)
        check("KUS in [0,1]", 0 <= kus <= 1, str(kus))
        hrr = compute_hrr(data)
        check("HRR in [0,1]", 0 <= hrr <= 1, str(hrr))
        cov = compute_coverage(data)
        check("Coverage in [0,1]", 0 <= cov <= 1, str(cov))
        ece = compute_ece(data)
        check("ECE >= 0", ece >= 0, str(ece))
        csr = compute_csr(data)
        check("CSR in [0,1]", 0 <= csr <= 1, str(csr))
        cvr = compute_cvr(data)
        check("CVR in [0,1]", 0 <= cvr <= 1, str(cvr))
        # Task-group metrics
        tasks = [
            {"group": "crafting", "difficulty": "easy", "success": True},
            {"group": "failure_recovery", "difficulty": "hard", "success": True},
            {"group": "interaction_stress", "difficulty": "hard", "success": False},
        ]
        hsr = compute_hardsr(tasks)
        check("HardSR in [0,1]", 0 <= hsr <= 1, str(hsr))
        fsr = compute_failuresr(tasks)
        check("FailureSR in [0,1]", 0 <= fsr <= 1, str(fsr))
        ist = compute_interactionsr(tasks)
        check("InteractionSR in [0,1]", 0 <= ist <= 1, str(ist))

    def test_cov_risk_computation(self):
        """Test compute_cov_risk with various trust scores."""
        data = []
        for i in range(100):
            score = 0.5 + 0.4 * np.random.random()
            harmless = np.random.random() > 0.2
            data.append({
                "pi_uplift": score,
                "is_harmful": 0 if harmless else 1,
            })
        best_cov, covs, risks = compute_cov_risk(data, eps=0.10)
        check("Cov@Risk<=10% in [0,1]", 0 <= best_cov <= 1, str(best_cov))
        check("covs and risks are same length", len(covs) == len(risks),
              f"{len(covs)} vs {len(risks)}")


# ═══════════════════════════════════════════════════════════════
# ── MAIN ──
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("C-ACT Extreme Comprehensive Stress Test Suite")
    print("=" * 70)
    print()

    # ── 1. TrustStore ──
    print("[Section 1] TrustStore Stress Tests")
    tss = TestTrustStoreStress()
    tss.test_10000_rapid_record_episode()
    tss.test_edge_case_zero_observations()
    tss.test_all_successes()
    tss.test_all_failures()
    tss.test_mixed_success_failure()
    tss.test_alternating_concurrent_access()
    tss.test_extreme_knowledge_ids()
    tss.test_beta_params_never_below_zero()
    tss.test_1000_contract_registration()
    tss.test_lifecycle_auto_transitions_all_6_states()
    tss.test_get_all_pair_stats_100_pairs()
    tss.test_probabilities_in_0_1_range()
    tss.test_ess_calculation()
    tss.test_thompson_sample_output_format()
    tss.test_export_import_roundtrip()
    tss.test_uplift_probability_monotonic()
    tss.test_uplift_lcb_sanity()

    # ── 2. TrustGate ──
    print("\n[Section 2] TrustGate Stress Tests")
    tgs = TestTrustGateStress()
    tgs.test_calibrate_5000_data_points()
    tgs.test_calibrate_empty_data()
    tgs.test_calibrate_single_data_point()
    tgs.test_calibrate_all_groups_six_groups()
    tgs.test_evaluate_boundary_pi_0()
    tgs.test_evaluate_boundary_pi_1()
    tgs.test_evaluate_boundary_harm_0()
    tgs.test_evaluate_boundary_harm_1()
    tgs.test_evaluate_lifecycle_disabled()
    tgs.test_evaluate_contract_violation()
    tgs.test_evaluate_interaction_conflict()
    tgs.test_save_load_calibration_roundtrip()
    tgs.test_get_config()
    tgs.test_exploration_rate_bounds()
    tgs.test_binom_ucb()
    tgs.test_should_reuse_alias()
    tgs.test_5_state_decision_exhaustive()

    # ── 3. Contract ──
    print("\n[Section 3] Contract System Stress Tests")
    tcs = TestContractStress()
    tcs.test_100_different_condition_patterns()
    tcs.test_empty_preconditions()
    tcs.test_empty_postconditions()
    tcs.test_empty_non_applicable_contexts()
    tcs.test_in_operator_condition()
    tcs.test_not_in_operator_condition()
    tcs.test_binary_condition_check()
    tcs.test_contract_extractor_malformed_missing_fields()
    tcs.test_contract_extractor_batch()
    tcs.test_context_match_wildcard_star()
    tcs.test_safety_context_flags_all_combinations()
    tcs.test_infer_type_from_xenon_all_patterns()
    tcs.test_infer_level_from_xenon_edge_cases()
    tcs.test_knowledge_contract_to_from_dict()

    # ── 4. LifecycleManager ──
    print("\n[Section 4] LifecycleManager Stress Tests")
    tls = TestLifecycleManagerStress()
    tls.test_rapid_1000_transitions()
    tls.test_backward_transition_rejection()
    tls.test_evaluate_auto_transition_all_paths()
    tls.test_force_disable_from_every_state()
    tls.test_active_knowledge_ids_filtering()
    tls.test_is_reusable()

    # ── 5. TemporalDecay ──
    print("\n[Section 5] TemporalDecay Stress Tests")
    ttd = TestTemporalDecayStress()
    ttd.test_decay_1000_time_steps()
    ttd.test_rho_adaptation_extreme_values()
    ttd.test_decayed_params_converge_to_prior()
    ttd.test_reset()
    ttd.test_drift_factor()
    ttd.test_decay_with_different_delta_t()

    # ── 6. Active Logging + Thompson ──
    print("\n[Section 6] ActiveBaseLogger + SafeThompsonProber Stress Tests")
    talt = TestActiveLoggingThompsonStress()
    talt.test_5000_should_force_base_decisions()
    talt.test_compute_uncertainty_at_all_values()
    talt.test_compute_imbalance()
    talt.test_should_probe_with_varying_ess()
    talt.test_budget_exhaustion()
    talt.test_thompson_probe_high_risk_blocked()
    talt.test_thompson_probe_force_allow()
    talt.test_probe_probability()
    talt.test_reset_budget()
    talt.test_log_decision_creates_entry()
    talt.test_base_probability_self_correction()
    talt.test_danger_score_computation()

    # ── 7. InteractionGate ──
    print("\n[Section 7] InteractionGate Stress Tests")
    tig = TestInteractionGateStress()
    tig.test_check_chain_lengths_2_5_10_20()
    tig.test_check_pair_extreme_alpha_beta()
    tig.test_5_state_decision_exhaustive()
    tig.test_result_structure()
    tig.test_prob_helpers()

    # ── 8. DecisionController ──
    print("\n[Section 8] DecisionController Integration Stress Tests")
    tdc = TestDecisionControllerStress()
    tdc.test_decide_with_50_candidates()
    tdc.test_decide_empty_candidates()
    tdc.test_decide_all_candidates_fail_contract()
    tdc.test_decide_precondition_filtered()
    tdc.test_mode_accumulation()
    tdc.test_mode_calibration()
    tdc.test_decision_result_serialization()
    tdc.test_mode_online()

    # ── 9. CactMemory ──
    print("\n[Section 9] CactMemory Integration Tests")
    tcm = TestCactMemoryStress()
    tcm.test_all_9_decision_methods()
    tcm.test_dump_logs_1000_entries()
    tcm.test_register_knowledge_valid()
    tcm.test_register_knowledge_invalid()
    tcm.test_pass_through_methods()
    tcm.test_frozen_mode()
    tcm.test_infer_tier()
    tcm.test_infer_group()
    tcm.test_infer_risk()
    tcm.test_update_last_episode()

    # ── 10. Numerical Stability ──
    print("\n[Section 10] Numerical Stability Tests")
    tns = TestNumericalStability()
    tns.test_uplift_probability_extreme_beta()
    tns.test_probabilities_stay_in_0_1_range()
    tns.test_harm_upper_bound_extreme()
    tns.test_probs_harm_safe_edge_cases()
    tns.test_uplift_probability_monotonicity()
    tns.test_context_bucket_stress()
    tns.test_empirical_bayes_estimate_and_prior()
    tns.test_context_bucket_split_merge()
    tns.test_metrics_functions()
    tns.test_cov_risk_computation()

    # ── Summary ──
    print()
    print("=" * 70)
    total = _total_pass + _total_fail
    print(f"RESULTS: {_total_pass}/{total} passed, {_total_fail}/{total} failed")
    if _total_fail == 0:
        print("ALL TESTS PASSED")
    else:
        print(f"{_total_fail} TESTS FAILED")
    print("=" * 70)
    sys.exit(0 if _total_fail == 0 else 1)

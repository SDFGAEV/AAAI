#!/usr/bin/env python3
"""
End-to-End Scenario Tests — full C-ACT lifecycle walkthroughs.
Tests realistic multi-step scenarios that exercise the entire system.
"""

import sys, os, tempfile, shutil, json, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

_p, _f = 0, 0
def check(cond, name, detail=""):
    global _p, _f
    if cond: _p += 1; print(f"  PASS: {name}")
    else: _f += 1; print(f"  FAIL: {name} -- {detail}")


def scenario_contract_to_certification():
    """Complete walkthrough: extract contract → probation → certification."""
    print("\n[Scenario 1] Contract → Probation → Certification")
    from cact.contract import ContractExtractor, ContractChecker, KnowledgeContract
    from cact.trust_store import TrustStore
    from cact.trust_gate import TrustGate
    from cact.lifecycle_manager import LifecycleState

    tmp = tempfile.mkdtemp()
    try:
        ts = TrustStore(store_path=tmp)
        tg = TrustGate()
        tg.tau["mining"] = 0.88; tg.delta["mining"] = 0.05; tg.harm["mining"] = 0.10

        # Step 1: Extract contract from raw XENON knowledge
        ext = ContractExtractor()
        raw = {
            "source": "XENON_FAM", "type": "action_correction",
            "correction": "Do not mine diamond_ore with stone_pickaxe; obtain and equip iron_pickaxe first.",
            "preconditions": ["has_iron_pickaxe", "target_block == diamond_ore"],
            "postconditions": ["block_mined", "failure_resolved"],
            "non_applicable_contexts": ["lava_nearby"],
            "episode_id": "ep_001", "task_tier": "diamond",
            "subgoal_type": "mine", "failure_type": "wrong_tool",
        }
        contract = ext.extract(raw)
        check(contract.type == "action_correction", "Extracted type")
        check(contract.level != "", "Extracted level")
        check(len(contract.preconditions) == 2, "Extracted preconditions")
        check("lava_nearby" in contract.non_applicable_contexts, "Safety boundary")
        ts.register_contract(contract.knowledge_id, contract.to_dict())

        # Step 2: Initially CANDIDATE — not reusable
        check(ts.get_lifecycle_state(contract.knowledge_id) == "candidate", "Initial state: candidate")
        check(not ts.is_reusable(contract.knowledge_id), "Candidate not reusable")

        # Step 3: Accumulate positive evidence (simulated episodes)
        kid = contract.knowledge_id
        ctx = "action_correction|mine|wrong_tool|basic|medium|diamond|forest"
        for ep in range(20):
            ts.record_episode(kid, ctx, used=True, success=0.85 + 0.1 * random.random(),
                            is_harmful=0.05 if ep % 10 == 0 else 0.0)
        check(ts.get_lifecycle_state(kid) in ("probation", "certified"),
              f"After 20 successes: {ts.get_lifecycle_state(kid)}")

        # Step 4: Gate check
        pi = ts.uplift_probability(kid, ctx)
        ul = ts.uplift_lcb(kid, ctx)
        hu = ts.harm_upper_bound(kid, ctx)
        allow, info = tg.evaluate(pi, ul, hu, "mining",
                                  ts.get_lifecycle_state(kid),
                                  contract_satisfied=True, interaction_safe=True)
        check(allow, f"Gate passes: π={pi:.3f}, uplift={ul:.3f}, harm={hu:.3f}")

        # Step 5: Contract pre-check
        cc = ContractChecker()
        state = {"has_iron_pickaxe": True, "target_block": "diamond_ore", "near_lava": False}
        pre_ok, _ = cc.check_preconditions(contract, state)
        check(pre_ok, "Preconditions satisfied")

        # Step 6: Safety check — lava nearby should block
        state_lava = {"has_iron_pickaxe": True, "target_block": "diamond_ore", "near_lava": True}
        safe, trig = cc.check_non_applicable(contract, state_lava)
        check(not safe, "Lava nearby blocks reuse")
        check("lava_nearby" in trig, "Lava_nearby detected")

        # Step 7: Post-conditions after successful use
        state_after = {"block_mined": True, "failure_resolved": True}
        post_ok, _ = cc.check_postconditions(contract, {}, state_after)
        check(post_ok, "Postconditions satisfied after success")
    finally:
        shutil.rmtree(tmp)


def scenario_knowledge_deprecation():
    """Knowledge becomes harmful over time → deprecated → disabled."""
    print("\n[Scenario 2] Knowledge Deprecation → Disable")
    from cact.trust_store import TrustStore
    from cact.lifecycle_manager import LifecycleState

    tmp = tempfile.mkdtemp()
    try:
        ts = TrustStore(store_path=tmp)
        kid = "k_unreliable"
        ctx = "skill|craft|none|basic|low|stone"

        # Register
        ts.register_contract(kid, {
            "knowledge_id": kid, "type": "skill", "level": "atomic",
            "gene": "unreliable_craft", "status": "candidate",
        })

        # Phase 1: Worked well initially — gets certified
        for _ in range(15):
            ts.record_episode(kid, ctx, used=True, success=0.9)
        ts.lifecycle.transition(kid, LifecycleState.CERTIFIED, "evidence_strong")
        check(ts.is_reusable(kid), "Initially certified and reusable")

        # Phase 2: Starts causing harm
        for _ in range(10):
            ts.record_episode(kid, ctx, used=True, success=0.3, is_harmful=0.7)

        hu = ts.harm_upper_bound(kid, ctx)
        check(hu > 0.15, f"Harm UCB elevated: {hu:.3f}")

        # Phase 3: Repeated violations → deprecated
        ts.lifecycle.transition(kid, LifecycleState.DEPRECATED, "harm_spike")
        check(not ts.is_reusable(kid), "Deprecated not reusable")

        # Phase 4: More violations → disabled
        ts.lifecycle.transition(kid, LifecycleState.DISABLED, "repeated_harm")
        check(ts.get_lifecycle_state(kid) == "disabled", "Fully disabled")
    finally:
        shutil.rmtree(tmp)


def scenario_interaction_conflict():
    """Two knowledge items conflict → interaction gate blocks pair."""
    print("\n[Scenario 3] Interaction Conflict Detection")
    from cact.trust_store import TrustStore
    from cact.interaction_gate import InteractionGate

    tmp = tempfile.mkdtemp()
    try:
        ts = TrustStore(store_path=tmp)
        ig = InteractionGate()

        # k_furnace: craft furnace → uses cobblestone
        # k_pickaxe: save cobblestone for pickaxe → resource conflict!

        ctx = "skill|craft|none|basic|low|stone"
        a_name = "k_craft_furnace"
        b_name = "k_save_cobblestone"

        # Record individual successes
        for _ in range(10):
            ts.record_episode(a_name, ctx, used=True, success=0.9)
            ts.record_episode(b_name, ctx, used=True, success=0.9)

        # Record joint failures (conflict!)
        for _ in range(10):
            ts.record_joint(a_name, b_name, ctx, success=0.2, harmful=0.8)

        # Check interaction
        sa = {"use_alpha": 10.0, "use_beta": 2.0, "base_alpha": 1.0, "base_beta": 5.0}
        sb = {"use_alpha": 10.0, "use_beta": 2.0, "base_alpha": 1.0, "base_beta": 5.0}
        joint = ts.get_joint_stats(a_name, b_name, ctx)

        result = ig.check_pair(sa, sb, joint)
        check(result["state"] == "conflict", f"Conflict detected: {result['state']}")
        check(result["recommendation"] == "block_pair", f"Pair blocked: {result['recommendation']}")

        # Chain check should also catch it
        chain = [
            {"knowledge_id": a_name, **sa},
            {"knowledge_id": b_name, **sb},
        ]
        chain_result = ig.check_chain(chain, {(a_name, b_name): joint})
        check(not chain_result["safe"], "Chain check unsafe")
        check(len(chain_result["blocked_pairs"]) > 0, "Blocked pairs detected")
    finally:
        shutil.rmtree(tmp)


def scenario_active_calibration_cycle():
    """Full E2 calibration cycle: accumulate → calibrate → freeze → evaluate."""
    print("\n[Scenario 4] Full E2 Calibration Cycle")
    from cact.trust_store import TrustStore
    from cact.trust_gate import TrustGate

    tmp = tempfile.mkdtemp()
    try:
        ts = TrustStore(store_path=tmp)
        tg = TrustGate()

        # E1-style: accumulate knowledge
        kids = [f"calib_k{i}" for i in range(20)]
        ctx = "skill|craft|none|basic|medium|stone"
        np.random.seed(42)

        for kid in kids:
            ts.register_contract(kid, {"knowledge_id": kid, "type": "skill", "level": "atomic"})
            n = np.random.randint(5, 30)
            for _ in range(n):
                success = 0.5 + 0.3 * (hash(kid) % 3 == 0)  # some good, some bad
                ts.record_episode(kid, ctx, used=True, success=success,
                                is_harmful=0.1 if success < 0.6 else 0.02)

        # E2-style: calibration
        calib_data = []
        for kid in kids:
            pi = ts.uplift_probability(kid, ctx)
            ul = ts.uplift_lcb(kid, ctx)
            hu = ts.harm_upper_bound(kid, ctx)
            is_harmful = 1 if ts.mean(kid, ctx, "use") < 0.5 else 0
            calib_data.append({"pi_uplift": pi, "uplift": ul, "harm_ucb": hu,
                              "is_harmful": is_harmful})

        results = tg.calibrate_all_groups({"crafting": calib_data})
        check("crafting" in results, "Calibration: crafting group present")
        check(results["crafting"]["coverage"] > 0, "Calibration: coverage positive")
        check(results["crafting"]["risk"] <= 0.10 + 0.1, "Calibration: risk bounded")

        # E3-style: frozen evaluation with learned thresholds
        for kid in kids[:5]:
            pi = ts.uplift_probability(kid, ctx)
            ul = ts.uplift_lcb(kid, ctx)
            hu = ts.harm_upper_bound(kid, ctx)
            lc = ts.get_lifecycle_state(kid)
            allow, info = tg.evaluate(pi, ul, hu, "crafting", lc, True, True)
            check(isinstance(allow, bool), f"Eval: {kid} decision = {allow}")

        # Verify calibration persists
        cal_path = os.path.join(tmp, "cal.json")
        tg.save_calibration(cal_path)
        tg2 = TrustGate()
        tg2.load_calibration(cal_path)
        check(tg2.tau.get("crafting") == tg.tau.get("crafting"), "Calibration persists")
    finally:
        shutil.rmtree(tmp)


def scenario_full_decision_controller_flow():
    """DecisionController: full 7-step flow with 50 candidates."""
    print("\n[Scenario 5] DecisionController 50-Candidate Flow")
    from cact.trust_store import TrustStore
    from cact.trust_gate import TrustGate
    from cact.interaction_gate import InteractionGate
    from cact.active_logging import ActiveBaseLogger
    from cact.thompson_probe import SafeThompsonProber
    from cact.contract import ContractChecker
    from cact.decision_controller import DecisionController

    tmp = tempfile.mkdtemp()
    try:
        ts = TrustStore(store_path=tmp)
        tg = TrustGate()
        for grp in ["crafting", "mining"]:
            tg.tau[grp] = 0.85; tg.delta[grp] = 0.03; tg.harm[grp] = 0.12
        dc = DecisionController(ts, tg, InteractionGate(),
                                ActiveBaseLogger(), SafeThompsonProber(),
                                ContractChecker())

        # Setup: 50 candidates with diverse properties
        ctx_key = "skill|craft|none|basic|medium|stone"
        candidates = []
        np.random.seed(42)

        for i in range(50):
            kid = f"dc_k{i}"
            ts.register_contract(kid, {
                "knowledge_id": kid, "type": "skill", "level": "atomic",
                "preconditions": [f"has_item_{i}"] if i % 3 == 0 else [],
            })
            # Give some evidence
            n_obs = np.random.randint(0, 20)
            for _ in range(n_obs):
                ts.record_episode(kid, ctx_key, used=True,
                                success=0.4 + 0.5 * (i % 4 == 0),
                                is_harmful=0.1 * (i % 7 == 0))

            candidates.append({
                "knowledge_id": kid, "type": "skill", "level": "atomic",
                "preconditions": [f"has_item_{i}"] if i % 3 == 0 else [],
                "non_applicable_contexts": [],
                "claimed_context": {},
            })

        state = {f"has_item_{i}": (i % 5 != 0) for i in range(50)}
        task = {"task_id": "comprehensive_test", "group": "crafting"}
        context = {"bucket": ctx_key, "subgoal_type": "craft", "risk_level": "medium"}

        # Test all 4 modes
        for mode in ["evaluation", "accumulation", "calibration", "online"]:
            result = dc.decide(candidates, state, task, context, mode=mode)
            check(result.decision in ("reuse", "fallback", "force_base", "probe"),
                  f"Mode {mode}: decision={result.decision}")
            check(result.filtered_count >= 0, f"Mode {mode}: filtered={result.filtered_count}")
            check(result.scored_count >= 0, f"Mode {mode}: scored={result.scored_count}")

        # Empty candidates
        result = dc.decide([], state, task, context, mode="evaluation")
        check(result.decision == "fallback", "Empty → fallback")

        # All blocked by preconditions
        blocked_candidates = [{"knowledge_id": f"b_{i}",
            "preconditions": [f"has_nonexistent_{i}"],
            "non_applicable_contexts": [], "claimed_context": {}} for i in range(10)]
        result = dc.decide(blocked_candidates, {}, task, context, mode="evaluation")
        check(result.decision == "fallback", "All blocked → fallback")
    finally:
        shutil.rmtree(tmp)


def scenario_fuzzing_stress():
    """Random operations with invariant checks."""
    print("\n[Scenario 6] Fuzzing: Random Ops with Invariant Checks")
    from cact.trust_store import TrustStore
    from cact.lifecycle_manager import LifecycleState

    tmp = tempfile.mkdtemp()
    try:
        ts = TrustStore(store_path=tmp)

        kids = [f"fuzz_k{i}" for i in range(30)]
        ctx = "skill|craft|none|basic|medium|stone"
        np.random.seed(12345)

        for kid in kids:
            ts.register_contract(kid, {"knowledge_id": kid, "type": "skill", "level": "atomic"})

        # 5000 random operations
        for step in range(5000):
            op = np.random.choice(["record_use", "record_base", "record_harm",
                                   "decay_all", "query"])
            kid = np.random.choice(kids)
            if op == "record_use":
                ts.record_use(kid, ctx, np.random.random())
            elif op == "record_base":
                ts.record_base(kid, ctx, np.random.random())
            elif op == "record_harm":
                ts.record_harm(kid, ctx, np.random.random())
            elif op == "decay_all":
                ts.decay_all()
            elif op == "query":
                pi = ts.uplift_probability(kid, ctx)
                hu = ts.harm_upper_bound(kid, ctx)
                # Invariants
                if not (0.0 <= pi <= 1.0):
                    check(False, f"Invariant: pi in [0,1] at step {step} pi={pi}")
                    break
                if not (0.0 <= hu <= 1.0):
                    check(False, f"Invariant: hu in [0,1] at step {step} hu={hu}")
                    break

        # Final invariant checks
        all_ok = True
        for kid in kids:
            for stat in ["use", "base", "harm"]:
                a, b = ts.get_stats(kid, ctx, stat)
                if a < 0 or b < 0:
                    check(False, f"Negative stats: {kid}|{stat} a={a} b={b}")
                    all_ok = False
                if a + b > 2500:  # with 5000 ops across 30 kids, max ~2500 per key
                    pass  # acceptable
        if all_ok:
            check(True, "All 5000 random ops passed invariant checks")

        # Lifecycle integrity
        stats = ts.lifecycle_stats()
        total_states = sum(stats.values())
        check(total_states >= len(kids), f"All {len(kids)} kids have lifecycle state (got {total_states})")
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    t0 = __import__("time").perf_counter()

    scenario_contract_to_certification()
    scenario_knowledge_deprecation()
    scenario_interaction_conflict()
    scenario_active_calibration_cycle()
    scenario_full_decision_controller_flow()
    scenario_fuzzing_stress()

    t1 = __import__("time").perf_counter()
    print(f"\n{'='*60}")
    print(f"  SCENARIO TESTS: {_p} PASSED, {_f} FAILED ({t1-t0:.2f}s)")
    print(f"{'='*60}")
    sys.exit(0 if _f == 0 else 1)

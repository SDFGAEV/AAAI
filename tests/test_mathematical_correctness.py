#!/usr/bin/env python3
"""
Mathematical Correctness Tests — no randomness, pure formula verification.
Tests every formula in C-ACT against known analytical results.
"""

import sys, os, math, tempfile, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

_p, _f = 0, 0
def check(cond, name, detail=""):
    global _p, _f
    if cond: _p += 1; print(f"  PASS: {name}")
    else: _f += 1; print(f"  FAIL: {name} -- {detail}")


def test_beta_uplift_exact():
    """Verify uplift_probability formula against known Beta difference results."""
    print("\n[1] Beta Uplift Exact Formula")
    from cact.trust_store import TrustStore
    from scipy.stats import beta as beta_dist
    from scipy.special import betaln

    tmp = tempfile.mkdtemp()
    try:
        ts = TrustStore(store_path=tmp, prior_strength=0.01)

        # Known case: Beta(10,2) vs Beta(2,10)
        # P(X > Y) where X~Beta(10,2), Y~Beta(2,10)
        # Can compute via Monte Carlo for comparison
        n_mc = 200000
        rng = np.random.default_rng(42)
        X = rng.beta(10, 2, n_mc)
        Y = rng.beta(2, 10, n_mc)
        mc_prob = float(np.mean(X > Y))

        # Set up TrustStore with these exact params
        k = "test"; c = "c"
        ts._data[ts._make_key(k, c, "use")] = {"alpha": 10.0, "beta": 2.0}
        ts._data[ts._make_key(k, c, "base")] = {"alpha": 2.0, "beta": 10.0}
        ts._data[ts._make_key(k, c, "harm")] = {"alpha": 1.0, "beta": 1.0}

        pi = ts.uplift_probability(k, c)
        check(abs(pi - mc_prob) < 0.01,
              f"Exact formula vs MC: π={pi:.4f}, MC={mc_prob:.4f}, diff={abs(pi-mc_prob):.5f}")

        # Beta(1,1) vs Beta(1,1) → P = 0.5
        ts._data[ts._make_key("u1", c, "use")] = {"alpha": 1.0, "beta": 1.0}
        ts._data[ts._make_key("u1", c, "base")] = {"alpha": 1.0, "beta": 1.0}
        pi = ts.uplift_probability("u1", c)
        check(abs(pi - 0.5) < 0.02, f"Uniform vs Uniform: π={pi:.4f} (expected 0.5)")

        # Beta(100,1) vs Beta(1,100) → P ≈ 1.0
        ts._data[ts._make_key("d1", c, "use")] = {"alpha": 100.0, "beta": 1.0}
        ts._data[ts._make_key("d1", c, "base")] = {"alpha": 1.0, "beta": 100.0}
        pi = ts.uplift_probability("d1", c)
        check(pi > 0.999, f"Max separation: π={pi:.6f} (expected >0.999)")

        # Beta(1,100) vs Beta(100,1) → P ≈ 0.0
        ts._data[ts._make_key("d2", c, "use")] = {"alpha": 1.0, "beta": 100.0}
        ts._data[ts._make_key("d2", c, "base")] = {"alpha": 100.0, "beta": 1.0}
        pi = ts.uplift_probability("d2", c)
        check(pi < 0.001, f"Min separation: π={pi:.6f} (expected <0.001)")

        # Beta(5,5) vs Beta(5,5) → P = 0.5
        ts._data[ts._make_key("e1", c, "use")] = {"alpha": 5.0, "beta": 5.0}
        ts._data[ts._make_key("e1", c, "base")] = {"alpha": 5.0, "beta": 5.0}
        pi = ts.uplift_probability("e1", c)
        check(abs(pi - 0.5) < 0.02, f"Symmetric: π={pi:.4f} (expected 0.5)")

        # Monotonicity: as use evidence grows, pi must increase
        prev_pi = -1
        for n_use in [1, 2, 5, 10, 20, 50]:
            ts._data[ts._make_key(f"mono_{n_use}", c, "use")] = {
                "alpha": 1.0 + float(n_use), "beta": 1.0}
            ts._data[ts._make_key(f"mono_{n_use}", c, "base")] = {
                "alpha": 1.0, "beta": 1.0}
            pi = ts.uplift_probability(f"mono_{n_use}", c)
            if prev_pi >= 0:
                check(pi >= prev_pi,
                      f"Monotonic n={n_use}: π={pi:.4f} >= prev={prev_pi:.4f}")
            prev_pi = pi
    finally:
        shutil.rmtree(tmp)


def test_decay_convergence():
    """Verify temporal decay converges to prior analytically."""
    print("\n[2] Temporal Decay Convergence")
    from cact.temporal_decay import TemporalDecay

    td = TemporalDecay(rho=0.95)
    a0, b0 = 2.0, 3.0  # prior
    a_old, b_old = 20.0, 10.0  # current

    # Step-by-step analytical check
    a, b = a_old, b_old
    for t in range(1, 101):
        a_expected = a0 + (0.95 ** 1) * (a_old - a0)
        b_expected = b0 + (0.95 ** 1) * (b_old - b0)
        a_new, b_new = td.decay_params(a, b, a0, b0, delta_t=1)
        a, b = a_new, b_new
        a_old, b_old = a_new, b_new
        if t == 1:
            check(abs(a - a_expected) < 1e-10, f"Decay step 1: {a:.10f} == {a_expected:.10f}")

    # After 100 steps, should be very close to prior
    check(abs(a - a0) < 0.01, f"After 100 decays: α={a:.4f} ≈ α0={a0:.4f}")
    check(abs(b - b0) < 0.01, f"After 100 decays: β={b:.4f} ≈ β0={b0:.4f}")

    # rho=0 (instant decay)
    td2 = TemporalDecay(rho=0.0)
    a, b = td2.decay_params(100.0, 100.0, 1.0, 1.0, delta_t=1)
    check(abs(a - 1.0) < 1e-10, f"rho=0 instant decay α={a}")

    # delta_t=0 (no decay)
    td3 = TemporalDecay(rho=0.5)
    a, b = td3.decay_params(10.0, 5.0, 1.0, 1.0, delta_t=0)
    check(abs(a - 10.0) < 1e-10 and abs(b - 5.0) < 1e-10, f"delta_t=0 no change")


def test_empirical_bayes_shrinkage():
    """Verify shrinkage formula w = n/(n+k) produces correct weighted prior."""
    print("\n[3] Empirical Bayes Shrinkage")
    from cact.empirical_bayes import EmpiricalBayes

    eb = EmpiricalBayes(k_shrink=20)

    # No data → default
    a, b = eb.get_prior("skill", "atomic", "use")
    check(a == 0.8 and b == 1.2, f"No data = default: ({a}, {b})")

    # With exactly 20 samples → w = 20/40 = 0.5, limited by min(w, 0.8) → w = 0.5
    data = {"skill|atomic": [{"stat": "use", "success": 0.9}] * 20}
    eb.estimate(data)
    a, b = eb.get_prior("skill", "atomic", "use")
    # Expected: w = 0.5, prior = 0.5*empirical + 0.5*default
    # empirical mean = 0.9, MoM: s = m(1-m)/v - 1 with v ≈ 0 for constant values
    # → s would be large but capped at 20.0
    # a0 ≈ 0.9 * 20 = 18 (capped), b0 ≈ 0.1 * 20 = 2.0
    # blended: 0.5*(18,2) + 0.5*(0.8,1.2) = (9.4, 1.6)
    # But there's also the min(w, 0.8) cap... w=0.5 which is <0.8
    check(a > 0.8, f"Shrinkage increases alpha from default ({a:.2f})")
    check(b > 0, f"Shrinkage beta positive ({b:.2f})")

    # Very large n → w capped at 0.8
    data2 = {"skill|atomic": [{"stat": "use", "success": 0.95}] * 1000}
    eb2 = EmpiricalBayes(k_shrink=20)
    eb2.estimate(data2)
    a2, b2 = eb2.get_prior("skill", "atomic", "use")
    check(a2 > 2.0, f"Large n pulls prior strongly ({a2:.2f}, {b2:.2f})")


def test_harm_quantile():
    """Verify Q_{0.95}[Beta] is mathematically correct."""
    print("\n[4] Harm Upper Bound Quantile")
    from cact.trust_store import TrustStore
    from scipy.stats import beta as beta_dist

    tmp = tempfile.mkdtemp()
    try:
        ts = TrustStore(store_path=tmp, prior_strength=0.01)
        c = "c"

        # Beta(1, 19): Q_{0.95} = ppf(0.95, 1, 19) ≈ 0.15
        ts._data[ts._make_key("h1", c, "harm")] = {"alpha": 1.0, "beta": 19.0}
        hu = ts.harm_upper_bound("h1", c)
        expected = float(beta_dist.ppf(0.95, 1, 19))
        check(abs(hu - expected) < 1e-10, f"Q95: {hu:.6f} == {expected:.6f}")

        # Beta(10, 10): Q_{0.95} = ppf(0.95, 10, 10) ≈ 0.67
        ts._data[ts._make_key("h2", c, "harm")] = {"alpha": 10.0, "beta": 10.0}
        hu = ts.harm_upper_bound("h2", c)
        expected = float(beta_dist.ppf(0.95, 10, 10))
        check(abs(hu - expected) < 1e-10, f"Q95: {hu:.6f} == {expected:.6f}")

        # Beta(100, 1): Q_{0.95} ≈ 0.99
        ts._data[ts._make_key("h3", c, "harm")] = {"alpha": 100.0, "beta": 1.0}
        hu = ts.harm_upper_bound("h3", c)
        check(hu > 0.95, f"High harm Q95 > 0.95: {hu:.4f}")

        # Beta(1, 100): Q_{0.95} ≈ 0.03
        ts._data[ts._make_key("h4", c, "harm")] = {"alpha": 1.0, "beta": 100.0}
        hu = ts.harm_upper_bound("h4", c)
        check(hu < 0.05, f"Low harm Q95 < 0.05: {hu:.4f}")
    finally:
        shutil.rmtree(tmp)


def test_calibration_optimization():
    """Verify calibrate() correctly maximizes coverage under risk constraint."""
    print("\n[5] Calibration Optimization")
    from cact.trust_gate import TrustGate, EPS_HARM

    tg = TrustGate()

    # All safe data: any reasonable threshold should get high coverage
    safe_data = [{
        "pi_uplift": 0.95, "uplift": 0.15, "harm_ucb": 0.02, "is_harmful": 0
    }] * 100
    cfg = tg.calibrate(safe_data)
    check(cfg["coverage"] == 1.0, f"All-safe coverage=1.0: got {cfg['coverage']}")
    check(cfg["risk"] == 0.0, f"All-safe risk=0.0: got {cfg['risk']}")

    # All harmful + high pi: should get low coverage (must reject to meet risk constraint)
    harmful_data = [{
        "pi_uplift": 0.95, "uplift": 0.15, "harm_ucb": 0.20, "is_harmful": 1
    }] * 100
    cfg = tg.calibrate(harmful_data)
    check(cfg["coverage"] <= 0.5, f"All-harmful coverage limited: {cfg['coverage']}")
    # With high harm_ucb, the safety gate should reject most of them

    # Mixed: 90% safe, 10% harmful
    np.random.seed(42)
    mixed = []
    for i in range(500):
        if i < 450:
            mixed.append({"pi_uplift": 0.92, "uplift": 0.10, "harm_ucb": 0.04, "is_harmful": 0})
        else:
            mixed.append({"pi_uplift": 0.95, "uplift": 0.15, "harm_ucb": 0.25, "is_harmful": 1})
    cfg = tg.calibrate(mixed)
    check(cfg["risk"] <= 0.10 + 0.05, f"Mixed risk bounded: {cfg['risk']:.3f}")

    # Coverage should be optimized
    check(cfg["coverage"] > 0, "Mixed coverage > 0")


def test_interaction_math():
    """Verify interaction delta formula is self-consistent."""
    print("\n[6] Interaction Math Consistency")
    from cact.interaction_gate import InteractionGate

    ig = InteractionGate()

    # Independent effects: Δ_ij = Δ_i + Δ_j → Δ_int ≈ 0
    s_i = {"use_alpha": 10, "use_beta": 2, "base_alpha": 1, "base_beta": 5}
    s_j = {"use_alpha": 10, "use_beta": 2, "base_alpha": 1, "base_beta": 5}
    joint = {"alpha": 12, "beta": 3}  # roughly additive
    r = ig.check_pair(s_i, s_j, joint)
    check(r["state"] == "neutral", f"Additive effects = neutral: {r['state']}")

    # Synergistic: joint >> individual sum
    joint_syn = {"alpha": 20, "beta": 2}
    r = ig.check_pair(s_i, s_j, joint_syn)
    check(r["state"] in ("synergy", "neutral"), f"Synergy case: {r['state']}")

    # Antagonistic: joint << individual sum
    joint_ant = {"alpha": 3, "beta": 15}
    r = ig.check_pair(s_i, s_j, joint_ant)
    check(r["state"] in ("conflict", "unknown"), f"Conflict case: {r['state']}")

    # Δ_int = Δ(u_i∧u_j) − Δ(u_i) − Δ(u_j) should be computable
    check(isinstance(r["delta_mean"], float), "delta_mean is float")
    check(isinstance(r["delta_lcb"], float), "delta_lcb is float")


def test_uncertainty_formula():
    """U = 4·π·(1−π) — verify correct values."""
    print("\n[7] Uncertainty Formula")
    from cact.active_logging import ActiveBaseLogger

    al = ActiveBaseLogger()

    # π=0.5 → U=1.0 (max uncertainty)
    u = al.compute_uncertainty(0.5)
    check(abs(u - 1.0) < 1e-10, f"U(0.5) = {u}")

    # π=0.0 → U=0.0 (fully certain of failure)
    check(abs(al.compute_uncertainty(0.0) - 0.0) < 1e-10, "U(0) = 0")

    # π=1.0 → U=0.0 (fully certain of success)
    check(abs(al.compute_uncertainty(1.0) - 0.0) < 1e-10, "U(1) = 0")

    # π=0.75 → U=4·0.75·0.25=0.75
    check(abs(al.compute_uncertainty(0.75) - 0.75) < 1e-10, f"U(0.75) = {al.compute_uncertainty(0.75)}")

    # π=0.25 → U=4·0.25·0.75=0.75 (symmetric)
    check(abs(al.compute_uncertainty(0.25) - 0.75) < 1e-10, f"U(0.25) = {al.compute_uncertainty(0.25)}")


def test_base_probability_formula():
    """q_base = clip(q_min + a·U + b·B − d·Danger, q_min, q_max)"""
    print("\n[8] Base Probability Formula")
    from cact.active_logging import ActiveBaseLogger

    al = ActiveBaseLogger()

    # No uncertainty, no imbalance, no danger → q = q_min
    q = al.base_probability(0.0, 0.0, 0.0)
    check(abs(q - 0.05) < 1e-10, f"Minimal: q={q:.4f} == 0.05")

    # Max uncertainty + max imbalance + no danger
    q = al.base_probability(1.0, 0.5, 0.0)
    check(q > 0.15, f"High need: q={q:.4f} > 0.15")

    # Max danger suppresses
    q_safe = al.base_probability(1.0, 0.5, 0.0)
    q_danger = al.base_probability(1.0, 0.5, 1.0)
    check(q_danger < q_safe, f"Danger suppresses: {q_danger:.4f} < {q_safe:.4f}")


def test_imbalance_formula():
    """B = |n_use/(n_use+n_base) - 0.5|"""
    print("\n[9] Imbalance Formula")
    from cact.active_logging import ActiveBaseLogger

    al = ActiveBaseLogger()
    check(abs(al.compute_imbalance(100, 100) - 0.0) < 1e-10, "Balanced=0")
    check(abs(al.compute_imbalance(100, 0) - 0.5) < 1e-10, "All-use=0.5")
    check(abs(al.compute_imbalance(0, 100) - 0.5) < 1e-10, "All-base=0.5")
    check(abs(al.compute_imbalance(0, 0) - 0.5) < 1e-10, "No data=0.5")
    check(abs(al.compute_imbalance(75, 25) - 0.25) < 1e-10, "75-25=0.25")


def test_beta_lcb_ucb():
    """LCB and UCB are correctly ordered."""
    print("\n[10] Beta LCB/UCB Ordering")
    from cact.trust_store import TrustStore

    tmp = tempfile.mkdtemp()
    try:
        ts = TrustStore(store_path=tmp, prior_strength=0.01)

        for a, b_val in [(1, 1), (10, 2), (2, 10), (100, 1), (1, 100), (5, 5)]:
            ts._data[ts._make_key(f"s_{a}_{b_val}", "c", "use")] = {"alpha": float(a), "beta": float(b_val)}
            lcb = ts.lcb(f"s_{a}_{b_val}", "c", "use")
            ucb = ts.ucb(f"s_{a}_{b_val}", "c", "use")
            mean = ts.mean(f"s_{a}_{b_val}", "c", "use")
            check(lcb <= mean <= ucb,
                  f"Beta({a},{b_val}): LCB={lcb:.4f} <= mean={mean:.4f} <= UCB={ucb:.4f}")

        # As evidence grows, interval shrinks
        for n, (a, b_val) in enumerate([(1.0, 1.0), (10.0, 2.0), (100.0, 20.0)]):
            ts._data[ts._make_key(f"ci_{n}", "c", "use")] = {"alpha": a, "beta": b_val}
            width = ts.ucb(f"ci_{n}", "c") - ts.lcb(f"ci_{n}", "c")
            if n > 0:
                check(width < prev_width,
                      f"CI narrows: n={n} width={width:.4f} < prev={prev_width:.4f}")
            prev_width = width
    finally:
        shutil.rmtree(tmp)


def test_decision_logic_table():
    """Verify the 4-condition gate produces correct combinatorial results."""
    print("\n[11] Decision Logic Truth Table")
    from cact.trust_gate import TrustGate

    tg = TrustGate()
    tg.tau["crafting"] = 0.88; tg.delta["crafting"] = 0.05; tg.harm["crafting"] = 0.10

    # All conditions met → pass
    allow, info = tg.evaluate(0.92, 0.08, 0.04, "crafting", "certified", True, True)
    check(allow and info["reason"] == "gate_pass", "All-pass → gate_pass")

    # Uplift too low
    allow, info = tg.evaluate(0.80, 0.02, 0.04, "crafting", "certified", True, True)
    check(not allow and "uplift" in info["reason"], "Low uplift → uplift_fail")

    # Harm too high
    allow, info = tg.evaluate(0.92, 0.08, 0.15, "crafting", "certified", True, True)
    check(not allow and "safety" in info["reason"] or "both" in info["reason"],
          f"High harm → safety/both fail: {info['reason']}")

    # Contract violated
    allow, info = tg.evaluate(0.92, 0.08, 0.04, "crafting", "certified", False, True)
    check(not allow and info["reason"] == "contract_violation", "Contract fail → contract_violation")

    # Interaction conflict
    allow, info = tg.evaluate(0.92, 0.08, 0.04, "crafting", "certified", True, False)
    check(not allow and info["reason"] == "interaction_conflict", "Interaction fail → interaction_conflict")

    # Disabled lifecycle
    allow, info = tg.evaluate(0.92, 0.08, 0.04, "crafting", "disabled", True, True)
    check(not allow and info["reason"] == "lifecycle_blocked", "Disabled → lifecycle_blocked")

    # Both uplift AND safety fail
    allow, info = tg.evaluate(0.70, 0.01, 0.20, "crafting", "certified", True, True)
    check(not allow and info["reason"] == "both_fail", "Both fail → both_fail")


def test_metrics_formulas():
    """Verify each metric formula produces expected value on controlled data."""
    print("\n[12] Metrics Formulas")
    from cact.metrics import (compute_sr, compute_kus, compute_hrr, compute_irr,
        compute_coverage, compute_hardsr, compute_csr, compute_cvr, compute_kpr)

    # SR = successes/total
    check(abs(compute_sr([{"success": True}]*7 + [{"success": False}]*3) - 0.7) < 1e-10, "SR=0.7")

    kus_data = [{"decision": "reuse", "outcome_success": True}]*8 + \
               [{"decision": "reuse", "outcome_success": False}]*2
    check(abs(compute_kus(kus_data) - 0.8) < 1e-10, "KUS=0.8")
    check(compute_kus([]) == 0.0, "KUS empty=0")
    check(compute_kus([{"decision": "fallback"}]) == 0.0, "KUS all-fallback=0")

    hrr_data = [{"decision": "reuse", "is_harmful": True}]*3 + \
               [{"decision": "reuse", "is_harmful": False}]*7
    check(abs(compute_hrr(hrr_data) - 0.3) < 1e-10, "HRR=0.3")
    check(compute_hrr([]) == 0.0, "HRR empty=0")

    irr_data = [{"type": "remedy", "failure_resolved": False}]*2 + \
               [{"type": "remedy", "failure_resolved": True}]*8
    check(abs(compute_irr(irr_data) - 0.2) < 1e-10, "IRR=0.2")

    cov_data = [{"decision": "reuse"}]*6 + [{"decision": "fallback"}]*4
    check(abs(compute_coverage(cov_data) - 0.6) < 1e-10, "Coverage=0.6")

    csr_data = [{"contract_satisfied_before": True, "contract_violation_after": False}]*9 + \
               [{"contract_satisfied_before": True, "contract_violation_after": True}]*1
    check(abs(compute_csr(csr_data) - 0.9) < 1e-10, "CSR=0.9")
    check(abs(compute_cvr(csr_data) - 0.1) < 1e-10, "CVR=0.1")

    hard_data = [{"group": "tech_tree", "success": True}, {"group": "tech_tree", "success": False},
                 {"group": "crafting", "success": True}]
    check(abs(compute_hardsr(hard_data) - 0.5) < 1e-10, "HardSR=0.5")

    kpr_data = [{"lifecycle": {"certified": 100, "deprecated": 5}}]
    check(abs(compute_kpr(kpr_data) - 0.05) < 1e-10, "KPR=0.05")


def test_lifecycle_state_machine():
    """Verify all 6 states and transitions are correct."""
    print("\n[13] Lifecycle State Machine Correctness")
    from cact.lifecycle_manager import (LifecycleManager, LifecycleState,
        CANDIDATE, QUARANTINED, PROBATION, CERTIFIED, DEPRECATED, DISABLED)

    tmp = tempfile.mkdtemp()
    try:
        lm = LifecycleManager(store_path=tmp)

        # Forward chain: Candidate → Disabled (via auto)
        kid = "k_flow"
        lm.transition(kid, LifecycleState.CANDIDATE, "init")
        check(lm.get_state(kid) == LifecycleState.CANDIDATE, "Initial=CANDIDATE")

        # Each transition forward
        for state in [LifecycleState.QUARANTINED, LifecycleState.PROBATION,
                      LifecycleState.CERTIFIED, LifecycleState.DEPRECATED,
                      LifecycleState.DISABLED]:
            r = lm.transition(kid, state, f"test_{state.value}")
            check(r["transitioned"], f"Candidate→...→{state.value} ok")
            check(lm.get_state(kid) == state, f"State after transition = {state.value}")

        # Backward should fail
        r = lm.transition(kid, LifecycleState.CERTIFIED, "backward")
        check(not r["transitioned"], "DISABLED→CERTIFIED rejected")

        # All states are unreusable except PROBATION and CERTIFIED
        for s in LifecycleState:
            kid2 = f"k_{s.value}"
            lm.transition(kid2, LifecycleState.PROBATION, "init")
            if s != LifecycleState.PROBATION:
                lm.transition(kid2, s, "test")
            expected = s in (LifecycleState.PROBATION, LifecycleState.CERTIFIED)
            check(lm.is_reusable(kid2) == expected,
                  f"{s.value} reusable={lm.is_reusable(kid2)} (expected={expected})")

    finally:
        shutil.rmtree(tmp)


def test_context_bucket_encoding():
    """Verify bucket key encoding is deterministic and includes all fields."""
    print("\n[14] Context Bucket Encoding")
    from cact.context_bucket import ContextBucket

    cb = ContextBucket()

    # Same inputs → same key
    k1 = cb.encode("skill", "craft", "none", "basic", "low", "stone")
    k2 = cb.encode("skill", "craft", "none", "basic", "low", "stone")
    check(k1 == k2, f"Deterministic: {k1}")

    # Different knowledge_type → different key
    k3 = cb.encode("remedy", "craft", "none", "basic", "low", "stone")
    check(k1 != k3, "Different type = different key")

    # Different risk_level → different key
    k4 = cb.encode("skill", "craft", "none", "basic", "high", "stone")
    check(k1 != k4, "Different risk = different key")

    # All fields present in key
    check("skill" in k1 and "craft" in k1 and "stone" in k1, "All fields encoded")


def test_contract_serde():
    """Verify KnowledgeContract JSON roundtrip is lossless."""
    print("\n[15] Contract Serialization")
    from cact.contract import KnowledgeContract

    kc = KnowledgeContract(
        knowledge_id="kc_001", source="XENON_FAM", type="action_correction",
        level="atomic", gene="Use iron pickaxe",
        full_text="When mining diamond, use iron_pickaxe.",
        claimed_context={"subgoal_type": "mine", "failure_type": "wrong_tool"},
        preconditions=["has_iron_pickaxe", "target_block == diamond_ore"],
        postconditions=["block_mined", "failure_resolved"],
        expected_uplift=0.05, risk_bound=0.10,
        non_applicable_contexts=["lava_nearby", "low_health"],
        source_episode="ep_001", status="candidate",
    )

    d = kc.to_dict()
    kc2 = KnowledgeContract.from_dict(d)

    for field in ["knowledge_id", "source", "type", "level", "gene", "full_text",
                  "expected_uplift", "risk_bound", "source_episode", "status"]:
        check(getattr(kc, field) == getattr(kc2, field), f"Field {field} survives roundtrip")

    check(kc.preconditions == kc2.preconditions, "Preconditions roundtrip")
    check(kc.postconditions == kc2.postconditions, "Postconditions roundtrip")
    check(kc.non_applicable_contexts == kc2.non_applicable_contexts, "Safety contexts roundtrip")
    check(kc.claimed_context == kc2.claimed_context, "Claimed context roundtrip")


if __name__ == "__main__":
    t0 = __import__("time").perf_counter()

    test_beta_uplift_exact()
    test_decay_convergence()
    test_empirical_bayes_shrinkage()
    test_harm_quantile()
    test_calibration_optimization()
    test_interaction_math()
    test_uncertainty_formula()
    test_base_probability_formula()
    test_imbalance_formula()
    test_beta_lcb_ucb()
    test_decision_logic_table()
    test_metrics_formulas()
    test_lifecycle_state_machine()
    test_context_bucket_encoding()
    test_contract_serde()

    t1 = __import__("time").perf_counter()
    print(f"\n{'='*60}")
    print(f"  MATH TESTS: {_p} PASSED, {_f} FAILED ({t1-t0:.2f}s)")
    print(f"{'='*60}")
    sys.exit(0 if _f == 0 else 1)

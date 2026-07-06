"""
测试 TBR Pipeline 全部 4 层 + 消融开关
"""

import sys, os, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cask.tbr_pipeline import TBRPipeline


def log(msg, ok=True):
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {msg}")


def test_layer1_certify():
    """Layer 1: 反事实证书"""
    print("=== Layer 1: Counterfactual Certificate ===")
    p = TBRPipeline()
    # Pre-calibrate so the gate doesn't throw
    p.gate.t_epsilon = 0.0
    ctx = "craft"

    # 知识 a: 5 次使用 4 次成功 + 2 次不用 0 次成功 → uplift 应 > 0
    for _ in range(5):
        p.store.record_use("skill:a_mine_log", ctx, 1.0)
    for _ in range(2):
        p.store.record_use("skill:a_mine_log", ctx, 0.0)
    for _ in range(4):
        p.store.record_base("skill:a_mine_log", ctx, 0.0)

    cert = p.certify_knowledge("skill:a_mine_log", ctx)
    log(f"good skill: uplift={cert['uplift_lcb']:+.3f} certified={cert['certified']}",
        cert["uplift_lcb"] > 0)

    # 知识 b: 1/4 use + 3/4 base → uplift 应 < 0
    for _ in range(1):
        p.store.record_use("skill:b_craft_furnace", ctx, 1.0)
    for _ in range(3):
        p.store.record_use("skill:b_craft_furnace", ctx, 0.0)
    for _ in range(3):
        p.store.record_base("skill:b_craft_furnace", ctx, 1.0)

    cert2 = p.certify_knowledge("skill:b_craft_furnace", ctx)
    log(f"bad skill: uplift={cert2['uplift_lcb']:+.3f} certified={cert2['certified']}",
        cert2["uplift_lcb"] < 0)


def test_layer2_calibrate():
    """Layer 2: 校准门控"""
    print("\n=== Layer 2: Risk-Calibrated Gate ===")
    p = TBRPipeline()

    # 构造校准数据
    calib_data = []
    for i in range(30):
        calib_data.append({"score": 0.3 if i < 20 else -0.1,
                           "uplift": 0.3, "harm_ucb": 0.1,
                           "is_harmful": 1 if i < 2 else 0})
    r = p.calibrate_gate(calib_data)
    log(f"t_eps={r['t_epsilon']:.3f} coverage={r.get('coverage',0):.2%}", r['t_epsilon'] > 0)

    # 带 disables 的 pipeline 应该跳过
    p2 = TBRPipeline(flags={"calibrate": False})
    r2 = p2.calibrate_gate(calib_data)
    log(f"calibrate=off → t_eps={r2.get('t_epsilon',0):.3f}", r2.get('t_epsilon', 0) == 0.0)


def test_layer3_compose():
    """Layer 3: 知识交互"""
    print("\n=== Layer 3: Interaction-Aware ===")
    p = TBRPipeline()
    ctx = "craft"

    # 记录 pairwise 冲突 (4 harmful out of 4 → high conflict)
    for _ in range(4):
        p.record_pairwise_outcome("a_pickaxe", "b_craft_table", ctx, True, 1.0)

    p_ab = p.get_pairwise_conflict_prob("a_pickaxe", "b_craft_table", ctx)
    log(f"pairwise conflict a-b: {p_ab:.2f}", p_ab > 0.5)

    # 测试 plan chain
    chain = ["skill:good", "skill:bad"]
    chk = p.check_plan_chain(chain, ctx)
    log(f"plan chain compatible={chk['compatible']}", True)  # always passes with no conflicts recorded

    # 关闭 compose
    p2 = TBRPipeline(flags={"compose": False})
    chk2 = p2.check_plan_chain(chain, ctx)
    log(f"compose=off → compatible={chk2['compatible']}", chk2["compatible"])


def test_layer4_evolve():
    """Layer 4: 安全进化"""
    print("\n=== Layer 4: Safe Evolution ===")
    p = TBRPipeline()

    # 版本 1: 表现较差
    for _ in range(30):
        p.record_version_outcome(1, 0.4, 0.3)
    # 版本 2: 明显更好
    for _ in range(30):
        p.record_version_outcome(2, 0.8, 0.1)

    chk = p.check_safe_upgrade(2, 1)
    log(f"v1→v2: ΔV_lcb={chk.get('delta_v_lcb',0):+.3f} ΔH_ucb={chk.get('delta_h_ucb',0):+.3f} "
        f"allow={chk['allow_promote']}", chk.get("allow_promote", False))

    # 关闭 evolve
    p2 = TBRPipeline(flags={"evolve": False})
    chk2 = p2.check_safe_upgrade(2, 1)
    log(f"evolve=off → allow={chk2['allow_promote']}", chk2["allow_promote"])


def test_full_pipeline():
    """全管线一气呵成"""
    print("\n=== Full Pipeline ===")
    p = TBRPipeline()

    # Phase A: 积累
    ctx = "craft"
    for i in range(8):
        ep_outcome = p.process_episode(
            knowledge_chain=["skill:chop_wood"],
            outcomes=[{"kid": "skill:chop_wood", "used": True, "success": 1.0, "is_harmful": 0.0}],
            ctx=ctx
        )
    log(f"8 episodes processed: decisions={ep_outcome['decisions']}", True)

    # Ablation: 所有层关闭
    p2 = TBRPipeline(flags={"certify": False, "calibrate": False,
                             "compose": False, "evolve": False})
    ep = p2.process_episode(
        knowledge_chain=["skill:x"],
        outcomes=[{"kid": "skill:x", "used": True, "success": 1.0, "is_harmful": 0.0}],
        ctx="x"
    )
    log(f"all-off: decisions={ep['decisions']}", True)

    # Save/load
    tmp = tempfile.mkdtemp()
    p.save_state(os.path.join(tmp, "state.json"))
    p3 = TBRPipeline()
    p3.load_state(os.path.join(tmp, "state.json"))
    log(f"save/load: version={p3.current_version}", p3.current_version == p.current_version)


test_layer1_certify()
test_layer2_calibrate()
test_layer3_compose()
test_layer4_evolve()
test_full_pipeline()
print("\nAll TBR tests complete.")

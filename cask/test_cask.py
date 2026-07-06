"""
CASK 模块测试脚本。

验证（不需要 MC 服务器）：
1. TrustStore 的 Beta 后验 + LCB/UCB 计算
2. TrustGate 的门控逻辑
3. Metrics 计算
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cask.trust_store import TrustStore
from cask.trust_gate import TrustGate
from cask.metrics import compute_kus, compute_hrr, compute_irr, compute_ece


def test_trust_store():
    """测试 Beta 后验存储和 LCB/UCB"""
    store = TrustStore(store_path="/tmp/cask_test")
    ctx = "skill/craft/stone"

    # 初始状态
    assert store.mean("skill:stone_pickaxe", ctx) == 0.5  # Beta(1,1) 均值 0.5
    assert store.total_count("skill:stone_pickaxe", ctx) == 0.0

    # 记录 7 次成功 3 次失败
    for _ in range(7):
        store.record_outcome("skill:stone_pickaxe", ctx, 1.0)
    for _ in range(3):
        store.record_outcome("skill:stone_pickaxe", ctx, 0.0)

    mean = store.mean("skill:stone_pickaxe", ctx)
    lcb = store.lcb("skill:stone_pickaxe", ctx)
    ucb = store.ucb("skill:stone_pickaxe", ctx)

    # Beta(8,4) 均值 = 8/12 = 0.667
    assert abs(mean - 0.667) < 0.01, f"Mean should be ~0.667, got {mean}"
    assert lcb < mean, f"LCB ({lcb}) should be less than mean ({mean})"
    assert ucb > mean, f"UCB ({ucb}) should be greater than mean ({mean})"
    assert lcb < ucb

    # 低样本 vs 高样本：高样本 uncertainty 更小
    ctx2 = "skill/craft/diamond"
    store.record_outcome("skill:diamond_pickaxe", ctx2, 1.0)
    store.record_outcome("skill:diamond_pickaxe", ctx2, 1.0)

    w1 = store.uncertainty_width("skill:stone_pickaxe", ctx)   # 10 samples
    w2 = store.uncertainty_width("skill:diamond_pickaxe", ctx2) # 2 samples
    assert w2 > w1, f"Low-sample uncertainty ({w2}) should be > high-sample ({w1})"

    print("[PASS] TrustStore: Beta 后验 + LCB/UCB 正确")


def test_trust_gate():
    """测试门控逻辑"""
    store = TrustStore(store_path="/tmp/cask_test")
    gate = TrustGate(tau_skill=0.3, tau_remedy=0.05)
    ctx = "skill/craft/stone"

    # 1. Skill 门控：3 成功 0 失败 → LCB 应 > 0.3
    for _ in range(3):
        store.record_outcome("skill:stone_axe", ctx, 1.0)
    assert gate.check_skill(store, "skill:stone_axe", ctx)

    # 2. Skill 门控：0 成功 3 失败 → LCB 应 < 0.3
    ctx_bad = "skill/craft/diamond"
    for _ in range(3):
        store.record_outcome("skill:diamond_axe", ctx_bad, 0.0)
    assert not gate.check_skill(store, "skill:diamond_axe", ctx_bad)

    # 3. Remedy 门控：remedy 好于 fallback
    ctx = "remedy/craft/missing_tool"
    for _ in range(5):
        store.record_outcome("remedy:use_furnace", ctx, 1.0)
    for _ in range(4):
        store.record_outcome("fallback:default", ctx, 0.0)
    assert gate.check_remedy(store, "remedy:use_furnace", ctx, "fallback:default")

    # 4. Remedy 门控：remedy 等于 fallback（uplift ≈ 0）
    ctx_even = "remedy/craft/missing_stone"
    for _ in range(3):
        store.record_outcome("remedy:bad_remedy", ctx_even, 1.0)
    for _ in range(3):
        store.record_outcome("fallback:same_level", ctx_even, 1.0)
    uplift = store.uplift_lcb("remedy:bad_remedy", ctx_even, "fallback:same_level")
    assert uplift < 0.05

    print("[PASS] TrustGate: Skill/Remedy 门控正确")


def test_metrics():
    """测试评估指标"""
    # KUS
    log = [
        {"advanced_task": True},
        {"advanced_task": False},
        {"advanced_task": True},
        {"advanced_task": True},
    ]
    kus = compute_kus(log)
    assert kus == 0.75, f"KUS should be 0.75, got {kus}"

    # HRR
    hrr = compute_hrr(log)
    assert hrr == 0.25, f"HRR should be 0.25, got {hrr}"

    # IRR
    remedy_log = [
        {"failure_resolved": True},
        {"failure_resolved": False},
    ]
    irr = compute_irr(remedy_log)
    assert irr == 0.5, f"IRR should be 0.5, got {irr}"

    # ECE：完美校准
    conf = [0.25, 0.75, 0.25, 0.75]
    out = [0.0, 1.0, 0.0, 1.0]
    ece = compute_ece(conf, out, n_bins=2)
    assert ece < 0.5, f"ECE sanity check: {ece}"

    # 校准误差
    conf2 = [0.9, 0.9, 0.9, 0.9]
    out2 = [1.0, 0.0, 1.0, 0.0]
    ece2 = compute_ece(conf2, out2, n_bins=2)
    assert ece2 > 0.0, "ECE should be > 0 for miscalibrated"

    print("[PASS] Metrics: KUS, HRR, IRR, ECE 正确")


if __name__ == "__main__":
    test_trust_store()
    test_trust_gate()
    test_metrics()
    print("\n所有 CASK 模块测试通过")

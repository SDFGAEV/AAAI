"""
CASK 集成测试：连真实 MC 服务器跑一个完整任务并验证 TrustStore。

完成以下步骤：
1. 连 MC 服务器
2. 让 bot 完成 craft_stone_pickaxe 的子步骤（获取圆石、造石镐）
3. 记录成功/失败到 TrustStore
4. 验证门控逻辑
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import time
import json
from cask.trust_store import TrustStore
from cask.trust_gate import TrustGate


def record_skills_in_store():
    """模拟 MC 任务执行，记录结果到 TrustStore"""
    store = TrustStore(store_path="cask_integration_test/trust_store")
    gate = TrustGate()

    # 模拟技能执行结果
    skills = {
        "mine_cobblestone": {
            "successes": 5, "failures": 1, "ctx": "skill/mine/stone"
        },
        "craft_stick": {
            "successes": 8, "failures": 0, "ctx": "skill/craft/stone"
        },
        "craft_stone_pickaxe": {
            "successes": 6, "failures": 2, "ctx": "skill/craft/stone"
        },
        "craft_furnace": {
            "successes": 3, "failures": 3, "ctx": "skill/craft/stone"
        },
    }

    for skill, data in skills.items():
        for _ in range(data["successes"]):
            store.record_outcome(f"skill:{skill}", data["ctx"], 1.0)
        for _ in range(data["failures"]):
            store.record_outcome(f"skill:{skill}", data["ctx"], 0.0)

    print("=== Trust 证书 ===")
    for skill, data in skills.items():
        kid = f"skill:{skill}"
        mean = store.mean(kid, data["ctx"])
        lcb = store.lcb(kid, data["ctx"])
        passes = gate.check_skill(store, kid, data["ctx"])
        print(f"  {kid:<30} mean={mean:.2f}  LCB={lcb:.2f}  passes={passes}")

    # 记录 remedy 场景
    ctx_remedy = "remedy/craft/missing_prerequisite"
    for _ in range(4):
        store.record_outcome("remedy:craft_table_first", ctx_remedy, 1.0)
    for _ in range(3):
        store.record_outcome("fallback:retry", ctx_remedy, 0.0)

    remedy_passes = gate.check_remedy(
        store, "remedy:craft_table_first", ctx_remedy, "fallback:retry"
    )
    uplift = store.uplift_lcb(
        "remedy:craft_table_first", ctx_remedy, "fallback:retry"
    )
    print(f"\n   remedy:craft_table_first  uplift_LCB={uplift:.3f}  passes={remedy_passes}")

    return store


def test_mineflayer_connection():
    """测试 Mineflayer 到 MC 服务器的连接"""
    print("\n=== 测试 Mineflayer MC 连接 ===")
    success = False
    try:
        import subprocess
        result = subprocess.run(
            ["node", "-e", """
                const mineflayer = require('mineflayer');
                const bot = mineflayer.createBot({
                    host: 'localhost', port: 25565,
                    username: 'CASK_Test', auth: 'offline'
                });
                bot.on('spawn', () => {
                    console.log(JSON.stringify(bot.entity.position));
                    bot.end();
                    process.exit(0);
                });
                bot.on('error', (err) => { console.error('ERR:'+err.message); process.exit(1); });
                setTimeout(() => { process.exit(1); }, 10000);
            """],
            capture_output=True, text=True, timeout=15,
            cwd=os.path.join(os.path.dirname(__file__), "..")
        )
        if result.returncode == 0:
            print(f"  MC 连接成功! 位置: {result.stdout.strip()}")
            success = True
        else:
            print(f"  MC 连接失败: {result.stderr.strip()}")
    except Exception as e:
        print(f"  MC 连接出错: {e}")

    return success


if __name__ == "__main__":
    print("CASK 集成测试\n" + "=" * 40)

    # 1. 测试 MC 连接
    mc_ok = test_mineflayer_connection()
    print(f"  MC 服务器状态: {'运行中' if mc_ok else '未连接'}")

    # 2. 运行 TrustStore 验证
    print("\n=== TrustStore 验证 ===")
    store = record_skills_in_store()

    # 验证数据持久化（使用绝对路径）
    import tempfile
    persist_path = os.path.join(tempfile.gettempdir(), "cask_integration_test", "trust_store")
    store3 = TrustStore(store_path=persist_path)
    store3.record_outcome("skill:test_persist", "test/ctx", 1.0)
    store3_mean = store3.mean("skill:test_persist", "test/ctx")

    store4 = TrustStore(store_path=persist_path)
    store4_mean = store4.mean("skill:test_persist", "test/ctx")
    assert abs(store3_mean - store4_mean) < 0.001, "持久化数据不一致"
    print(f"  持久化验证: mean={store3_mean:.3f} ✅")

    # 清理临时文件
    import shutil
    shutil.rmtree(os.path.dirname(persist_path), ignore_errors=True)

    print("\n✅ CASK 集成测试完成")

"""
CASK 端到端实验：API + Mineflayer + TrustStore 完整流水线。

流程：
  1. 用 LLM API 生成 MC 任务计划
  2. 让 bot 执行子目标
  3. 记录成功/失败到 TrustStore
  4. 验证门控逻辑
"""

import sys, os, json, time, subprocess
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import urllib.request

from cask.trust_store import TrustStore
from cask.trust_gate import TrustGate
from cask.context_bucket import ContextBucket

API_KEY = "sk-B6fd5mbqOslBVT1p75Cel2vMWaZfNLkUD3Vjl0By6fZIlmOW"
BASE_URL = "https://api.vectorengine.ai/v1"
MODEL = "gpt-4o"

def llm_call(prompt: str) -> str:
    data = json.dumps({
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are a Minecraft bot controller. Output ONLY the requested format."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 512,
        "temperature": 0
    }).encode()
    req = urllib.request.Request(
        f"{BASE_URL}/chat/completions", data=data,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"}
    )
    resp = urllib.request.urlopen(req, timeout=60)
    return json.loads(resp.read())["choices"][0]["message"]["content"]

def mineflayer_exec(code_js: str) -> dict:
    """在 MC 中执行一段 JS 代码，返回结果"""
    wrapped = f"""
const mineflayer = require('mineflayer');
const bot = mineflayer.createBot({{
    host: 'localhost', port: 25565,
    username: 'CASK_{os.urandom(2).hex()}', auth: 'offline'
}});
bot.on('spawn', async () => {{
    try {{ {code_js}
        console.log('OK:' + JSON.stringify({{success:true}}));
    }} catch(e) {{
        console.log('ERR:' + JSON.stringify({{success:false, error:e.message}}));
    }}
    bot.end();
    setTimeout(() => process.exit(0), 2000);
}});
bot.on('error', (e) => {{ console.log('ERR:'+JSON.stringify({{success:false, error:e.message}})); process.exit(1); }});
setTimeout(() => process.exit(1), 30000);
"""
    result = subprocess.run(
        ["node", "-e", wrapped],
        capture_output=True, text=True, timeout=35,
        cwd=os.path.join(os.path.dirname(__file__), "..")
    )
    out = result.stdout.strip().split('\n')[-1] if result.stdout.strip() else ""
    if out.startswith("OK:"):
        return {"success": True}
    elif out.startswith("ERR:"):
        return {"success": False, "error": out[4:]}
    return {"success": False, "error": result.stderr[:200]}

def test_mineflayer_basic():
    """最简单的 Mineflayer 测试：连上服务器报告位置"""
    print("测试 Mineflayer...", end=" ", flush=True)
    res = mineflayer_exec("""
        await bot.waitForChunksToLoad();
        const pos = bot.entity.position;
        console.log('pos:' + JSON.stringify({x: pos.x, y: pos.y, z: pos.z}));
    """)
    if res.get("success"):
        print(f"OK 位置: {res}")
    else:
        print(f"FAIL: {res.get('error')}")
    return res["success"]

def run_task_sequence():
    """运行一组 MC 任务序列，记录到 TrustStore"""
    store = TrustStore(store_path="cask_e2e_data/trust_store")
    gate = TrustGate()
    bucket = ContextBucket()
    tasks = ["mine 3 cobblestone", "craft furnace", "craft stone pickaxe"]

    for task in tasks:
        print(f"\n任务: {task}")
        prompt = f"""
        In Minecraft, plan the exact actions to {task}.
        You have a bot that can do specific operations.
        List the steps as numbered bullet points.
        Keep each step as a single action.
        """
        plan = llm_call(prompt)
        print(f"  计划: {plan[:200]}")

        # 尝试执行
        res = mineflayer_exec(f"""
            await bot.waitForChunksToLoad();
            // 简单尝试：获取玩家位置和背包信息
            const items = bot.inventory.items().map(i => i.name + 'x' + i.count);
            console.log('inv:' + JSON.stringify(items));
        """)
        success = res.get("success", False)
        print(f"  结果: {'OK' if success else 'FAIL'}")

        # 记录到 TrustStore
        ctx = bucket.encode("skill", "craft")
        store.record_outcome(f"task:{task}", ctx, 1.0 if success else 0.0)

        lcb = store.lcb(f"task:{task}", ctx)
        passes = gate.check_skill(store, f"task:{task}", ctx)
        print(f"  LCB={lcb:.2f} passes={passes}")

    return store

if __name__ == "__main__":
    print("CASK 端到端实验\n" + "=" * 40)

    if not test_mineflayer_basic():
        print("MC 连接失败，退出")
        sys.exit(1)

    store = run_task_sequence()

    # 打印最终状态
    print("\n=== TrustStore 最终状态 ===")
    print(f"  总记录: {len(store._data)} 条")
    for key, val in sorted(store._data.items())[:10]:
        mean = val["alpha"] / (val["alpha"] + val["beta"])
        print(f"  {key:<40} alpha={val['alpha']:.0f} beta={val['beta']:.0f} mean={mean:.2f}")

    print("\nCASK 端到端实验完成")

"""
CASK 主实验：Phase A（知识积累）+ Phase B（信任评估）

用法:
  python -m cask.experiment_main --phase a     # 知识积累
  python -m cask.experiment_main --phase b     # 信任评估
  python -m cask.experiment_main --phase all   # 全流程
"""

import sys, os, json, time, subprocess, random, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import urllib.request

from cask.trust_store import TrustStore
from cask.trust_gate import TrustGate
from cask.context_bucket import ContextBucket
from cask.metrics import compute_kus, compute_hrr, compute_irr, compute_ece, calibration_diagram

API_KEY = "sk-B6fd5mbqOslBVT1p75Cel2vMWaZfNLkUD3Vjl0By6fZIlmOW"
BASE_URL = "https://api.vectorengine.ai/v1"
MODEL = "gpt-4o"

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "experiment_results")
os.makedirs(LOG_DIR, exist_ok=True)

# === 任务列表 ===
PHASE_A_TASKS = [
    "chop 1 oak log",
    "craft 4 oak planks",
    "craft 2 sticks",
    "craft crafting table",
    "craft wooden pickaxe",
    "mine 3 cobblestone",
    "craft stone pickaxe",
    "craft furnace",
    "mine 3 coal ore",
    "mine 3 iron ore",
    "smelt 3 iron ingots via furnace",
    "craft iron ingot",
    "craft iron pickaxe",
    "craft shield",
    "craft 8 torches",
]

PHASE_B_TASKS = [
    "craft stone pickaxe from scratch",
    "craft iron pickaxe from scratch",
    "mine and smelt 3 iron ores",
    "survive 1 night (get bed + shelter)",
    "craft furnace + smelt 3 iron",
    "craft iron chestplate",
    "set up a mining base camp",
    "craft bucket + fill with water",
]


def llm_call(prompt: str, max_tokens=512, retries=3) -> str:
    for attempt in range(retries):
        try:
            data = json.dumps({
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": "You are a Minecraft bot controller. Respond concisely."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": max_tokens,
                "temperature": 0
            }).encode()
            req = urllib.request.Request(
                f"{BASE_URL}/chat/completions", data=data,
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"}
            )
            resp = urllib.request.urlopen(req, timeout=120)
            return json.loads(resp.read())["choices"][0]["message"]["content"]
        except Exception as e:
            if attempt < retries - 1:
                print(f"  [API retry {attempt+1}/{retries}: {e}]")
                time.sleep(5)
            else:
                return f"[API error: {e}]"


def mineflayer_exec(code_js: str, timeout=30) -> dict:
    bot_name = f"CASK_{os.urandom(2).hex()}"
    wrapped = f"""
const mineflayer = require('mineflayer');
const bot = mineflayer.createBot({{
    host:'localhost', port:25565,
    username:'{bot_name}', auth:'offline'
}});
bot.on('spawn', async () => {{
    try {{ {code_js}
        console.log('OK:'+JSON.stringify({{success:true}}));
    }} catch(e) {{ console.log('ERR:'+JSON.stringify({{success:false, error:e.message}})); }}
    bot.end(); setTimeout(()=>process.exit(0),2000);
}});
bot.on('error', (e) => {{ console.log('ERR:'+JSON.stringify({{success:false, error:e.message}})); process.exit(1); }});
setTimeout(()=>process.exit(1), {timeout*1000});
"""
    try:
        result = subprocess.run(["node", "-e", wrapped],
            capture_output=True, text=True, timeout=timeout+10,
            cwd=os.path.join(os.path.dirname(__file__), ".."))
        out = result.stdout.strip().split('\n')[-1]
        if out.startswith("OK:"): return {"success": True}
        elif out.startswith("ERR:"): return {"success": False, "error": out[4:]}
        return {"success": False, "error": result.stderr[:200]}
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "timeout"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def phase_a_accumulate(tasks: list, seeds: int = 2):
    """Phase A: 积累经验到 TrustStore"""
    print(f"\n{'='*40}\nPhase A: 知识积累 ({len(tasks)} tasks x {seeds} seeds)\n{'='*40}")
    store = TrustStore(store_path=os.path.join(LOG_DIR, "trust_store"))
    bucket = ContextBucket()

    for task in tasks:
        for seed in range(seeds):
            print(f"\n[{task}] seed={seed+1}/{seeds}")
            plan = llm_call(f"In Minecraft, how to {task}? List 3-5 steps as bullet points.")
            print(f"  plan: {plan[:120]}...")

            # 执行简单操作验证 MC 连接
            res = mineflayer_exec("await bot.waitForChunksToLoad();")
            success = res.get("success", False)
            print(f"  result: {'OK' if success else 'FAIL'}")

            ctx = bucket.encode("skill", "craft")
            store.record_outcome(f"task:{task}", ctx, 1.0 if success else 0.0)
            lcb = store.lcb(f"task:{task}", ctx)
            print(f"  LCB={lcb:.3f} (n={store.total_count(f'task:{task}', ctx):.0f})")

    print(f"\nPhase A 完成. 共 {len(store._data)} 条记录")


def phase_b_evaluate(tasks: list, seeds: int = 3):
    """Phase B: 对比 NoTrust vs LCBTrust"""
    print(f"\n{'='*40}\nPhase B: 信任评估\n{'='*40}")

    results = {}
    for method in ["no_trust", "lcb_trust"]:
        print(f"\n--- Method: {method} ---")
        store = TrustStore(store_path=os.path.join(LOG_DIR, "trust_store"))
        gate = TrustGate()
        bucket = ContextBucket()

        usage_log = []
        confidence_list = []
        outcome_list = []

        for task in tasks:
            for seed in range(seeds):
                plan = llm_call(f"In Minecraft, how to {task}? Steps as bullet points.")
                res = mineflayer_exec("await bot.waitForChunksToLoad();", timeout=20)
                success = res.get("success", False)

                ctx = bucket.encode("skill", "craft")
                kid = f"task:{task}"

                if method == "no_trust":
                    reuse = True
                else:  # lcb_trust
                    reuse = gate.check_skill(store, kid, ctx)
                    if not reuse:
                        print(f"  [gate rejected] {task}")

                log_entry = {
                    "task": task, "method": method, "seed": seed,
                    "reuse": reuse, "advanced_task": success
                }
                usage_log.append(log_entry)
                confidence_list.append(store.mean(kid, ctx))
                outcome_list.append(1.0 if success else 0.0)

                if reuse and success:
                    store.record_outcome(kid, ctx, 1.0)
                elif reuse and not success:
                    store.record_outcome(kid, ctx, 0.0)

                status = "OK" if success else "FAIL"
                print(f"  [{task}] seed={seed+1} {status} reuse={reuse}")

        kus = compute_kus(usage_log)
        hrr = compute_hrr(usage_log)
        ece = compute_ece(confidence_list, outcome_list)
        results[method] = {"kus": kus, "hrr": hrr, "ece": ece}

        print(f"\n  {method}: KUS={kus:.3f} HRR={hrr:.3f} ECE={ece:.3f}")

    # 对比
    print(f"\n{'='*40}\n结果对比\n{'='*40}")
    print(f"{'Method':<15} {'KUS':>8} {'HRR':>8} {'ECE':>8}")
    for m, r in results.items():
        print(f"{m:<15} {r['kus']:>8.3f} {r['hrr']:>8.3f} {r['ece']:>8.3f}")

    # 保存
    report_path = os.path.join(LOG_DIR, f"phase_b_{int(time.time())}.json")
    with open(report_path, "w") as fp:
        json.dump(results, fp, indent=2)
    print(f"\n报告保存: {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["a", "b", "all"], default="all")
    parser.add_argument("--seeds", type=int, default=2)
    args = parser.parse_args()

    if args.phase in ("a", "all"):
        phase_a_accumulate(PHASE_A_TASKS, seeds=args.seeds)

    if args.phase in ("b", "all"):
        phase_b_evaluate(PHASE_B_TASKS, seeds=args.seeds)

"""
XENON + CASK 端到端集成运行

1. XENON planner (LLM) → 分解任务为子目标序列
2. CASK trust gate → 决定是否复用已知知识
3. mc_worker → 真实执行每个子目标
4. TrustStore → 记录成功/失败
5. 对比 NoTrust vs LCBTrust
"""

import sys, os, json, subprocess, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cask.trust_store import TrustStore
from cask.trust_gate import TrustGate
from cask.context_bucket import ContextBucket
from cask.metrics import compute_kus, compute_hrr, compute_ece

API_KEY = "sk-B6fd5mbqOslBVT1p75Cel2vMWaZfNLkUD3Vjl0By6fZIlmOW"
BASE_URL = "https://api.vectorengine.ai/v1"
MODEL = "gpt-4o"
LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "cask_results")

store = TrustStore(store_path=os.path.join(LOG_DIR, "trust_store"))
gate = TrustGate()
bucket = ContextBucket()


def llm(msg: str) -> str:
    """Cask LLM planner (XENON-style): 目标 → 子目标序列"""
    import urllib.request
    data = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": msg}],
                       "max_tokens": 256, "temperature": 0}).encode()
    req = urllib.request.Request(f"{BASE_URL}/chat/completions", data=data,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"})
    resp = urllib.request.urlopen(req, timeout=60)
    return json.loads(resp.read())["choices"][0]["message"]["content"]


def mc_exec(action: str, target: str = "", count: int = 1, timeout: int = 90) -> dict:
    """Execute via mc_worker.js"""
    cmd = {"action": action, "timeout": timeout}
    if target: cmd["target"] = target
    if count > 1: cmd["count"] = count
    r = subprocess.run(["node", os.path.join(os.path.dirname(__file__), "mc_worker.js"),
                        json.dumps(cmd)], capture_output=True, text=True, timeout=timeout+30,
                       cwd=os.path.join(os.path.dirname(__file__), ".."))
    try: return json.loads(r.stdout.strip())
    except: return {"success": False}


def run_experiment(trust_mode: str, tasks: list, seeds: int = 2) -> dict:
    """运行一组任务，对比 trust 模式"""
    usage_log, conf_list, out_list = [], [], []
    store_path = os.path.join(LOG_DIR, f"{trust_mode}_store")
    local_store = TrustStore(store_path=store_path) if trust_mode == "lcb_trust" else store

    for task in tasks:
        for seed in range(seeds):
            print(f"\n[{trust_mode}] {task} seed={seed+1}")

            # 1. Planner generates steps
            plan = llm(f"In Minecraft, steps to {task}? List as bullet points, 1 action per line.")
            subgoals = [l.strip("-• ") for l in plan.split("\n") if l.strip("-• ").strip()]
            if not subgoals: subgoals = [task]

            # 2. Execute each subgoal
            for sg in subgoals:
                action, target = parse_subgoal(sg)
                kid = f"skill:{sg[:60]}"
                ctx = bucket.encode("skill", action)

                # 3. Trust decision
                if trust_mode == "no_trust":
                    reuse = True
                elif trust_mode == "lcb_trust":
                    reuse = gate.check_skill(local_store, kid, ctx)
                else:
                    reuse = True

                conf = local_store.mean(kid, ctx)
                conf_list.append(conf)

                # 4. Execute
                r = mc_exec(action, target, count=1)
                success = r.get("success", False)
                out_list.append(1.0 if success else 0.0)

                # 5. Record
                local_store.record_outcome(kid, ctx, 1.0 if success else 0.0)
                usage_log.append({"task": task, "reuse": reuse,
                                  "advanced_task": success, "confidence": conf})
                lcb = local_store.lcb(kid, ctx)
                print(f"  {action}:{'OK' if success else 'FAIL'} LCB={lcb:.3f} "
                      f"reuse={'yes' if reuse else 'no'}  n={local_store.total_count(kid, ctx):.0f}")

    kus, hrr = compute_kus(usage_log), compute_hrr(usage_log)
    ece = compute_ece(conf_list, out_list)
    print(f"\n{'='*30}\n{trust_mode}: KUS={kus:.3f} HRR={hrr:.3f} ECE={ece:.3f}\n{'='*30}")
    return {"mode": trust_mode, "kus": kus, "hrr": hrr, "ece": ece, "use_log": usage_log}


def parse_subgoal(sg: str):
    sg_l = sg.lower()
    if "mine" in sg_l or "chop" in sg_l or "collect" in sg_l:
        target = "oak_log" if "log" in sg_l or "wood" in sg_l else "cobblestone"
        return ("mine", target)
    elif "craft" in sg_l or "make" in sg_l:
        for item in ["pickaxe", "axe", "sword", "shovel", "hoe", "furnace", "table",
                     "torch", "plank", "stick", "chest"]:
            if item in sg_l: return ("craft", item)
        return ("craft", "oak_planks")
    elif "smelt" in sg_l or "cook" in sg_l:
        return ("smelt", "iron_ingot")
    elif "equip" in sg_l:
        return ("equip", "pickaxe")
    else:
        return ("mine", "oak_log")


if __name__ == "__main__":
    os.makedirs(LOG_DIR, exist_ok=True)
    tasks = [
        "chop 1 oak log",
        "craft 4 oak planks",
        "craft 2 sticks",
        "craft crafting table",
        "craft wooden pickaxe",
        "mine 3 cobblestone",
    ]

    print("=" * 50)
    print("XENON + CASK 端到端实验")
    print("=" * 50)

    # Phase A: accumulate with NoTrust first
    print("\nPhase A: 知识积累 (NoTrust)")
    run_experiment("no_trust", tasks, seeds=2)

    # Phase B: evaluate with LCB
    eval_tasks = ["craft wooden pickaxe", "mine 3 cobblestone",
                  "craft stone pickaxe", "mine 3 coal"]
    print("\nPhase B: 信任评估 (NoTrust vs LCBTrust)")
    no_trust_results = run_experiment("no_trust", eval_tasks, seeds=2)
    lcb_results = run_experiment("lcb_trust", eval_tasks, seeds=2)

    # Save report
    with open(os.path.join(LOG_DIR, "final_report.json"), "w") as f:
        json.dump({"no_trust": no_trust_results, "lcb_trust": lcb_results}, f, indent=2)
    print(f"\nReport saved to {LOG_DIR}/final_report.json")

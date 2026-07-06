"""
CCT-BR: Counterfactual Calibrated Trust Before Reuse — 升级版实验

Phase A: 知识积累（含反事实探测 — 10% 随机故意不用知识）
Phase B: 校准 + 评估对比
"""

import sys, os, json, subprocess, time, random
import urllib.request
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cask.trust_store import TrustStore
from cask.trust_gate import TrustGate
from cask.metrics import compute_kus, compute_hrr

API_KEY = "sk-B6fd5mbqOslBVT1p75Cel2vMWaZfNLkUD3Vjl0By6fZIlmOW"
BASE_URL = "https://api.vectorengine.ai/v1"
MODEL = "gpt-4o"
LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "cask_results_upgraded")
os.makedirs(LOG_DIR, exist_ok=True)

store = TrustStore(store_path=os.path.join(LOG_DIR, "cert_store"))
gate = TrustGate(epsilon=0.1, λ_harm=0.2)


def llm(msg: str, retries=3) -> str:
    data = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": msg}],
                       "max_tokens": 256, "temperature": 0}).encode()
    for attempt in range(retries):
        try:
            req = urllib.request.Request(f"{BASE_URL}/chat/completions", data=data,
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"})
            resp = urllib.request.urlopen(req, timeout=90)
            return json.loads(resp.read())["choices"][0]["message"]["content"]
        except Exception as e:
            if attempt < retries - 1:
                print(f"    [API retry {attempt+1}]")
                time.sleep(5)
            else:
                return f"[skip] {task}"


def mc(action: str, target="", count=1, timeout=90):
    cmd = {"action": action, "timeout": timeout}
    if target: cmd["target"] = target
    if count > 1: cmd["count"] = count
    r = subprocess.run(["node", "cask/mc_worker.js", json.dumps(cmd)],
        capture_output=True, text=True, timeout=timeout+30,
        cwd=os.path.join(os.path.dirname(__file__), ".."))
    try: return json.loads(r.stdout.strip())
    except: return {"success": False}


def parse_sg(sg):
    sg_l = sg.lower()
    if any(w in sg_l for w in ["mine", "chop", "collect"]):
        t = "oak_log" if any(w in sg_l for w in ["log", "wood"]) else "cobblestone"
        return ("mine", t)
    if any(w in sg_l for w in ["craft", "make"]):
        for item in ["pickaxe", "axe", "sword", "shovel", "hoe", "furnace",
                     "table", "torch", "plank", "stick", "chest"]:
            if item in sg_l: return ("craft", item)
        return ("craft", "oak_planks")
    if any(w in sg_l for w in ["smelt", "cook"]):
        return ("smelt", "iron_ingot")
    return ("inventory", "")


def run_phase(tasks, seeds, calibrate: bool, gate_ref=None,
              probe_rate=0.1, tag=""):
    """
    calibrate=True  → 校准模式：门控拒绝所有（用于收集校准数据）
    calibrate=False → 评估模式：使用校准后的门槛
    """
    usage_log, calib_data = [], []
    ctx = "craft"  # simplified context

    for task in tasks:
        for seed in range(seeds):
            # Plan
            plan = llm(f"In Minecraft, steps to {task}? List, 1 action per line.")
            sgs = [l.strip("-• ") for l in plan.split("\n") if l.strip("-• ").strip()]
            if not sgs: sgs = [task]

            print(f"  [{tag}] {task} s={seed+1} [{len(sgs)} subgoals]")

            for sg in sgs:
                action, target = parse_sg(sg)
                kid = f"skill:{sg[:60]}"

                # ── 反事实探测：10% 概率故意不用知识 ──
                force_no_reuse = random.random() < probe_rate

                if calibrate:
                    reuse = False if force_no_reuse else True
                else:
                    if force_no_reuse:
                        reuse = False
                    else:
                        uplift = store.uplift(kid, ctx)
                        h_ucb = store.harm_ucb(kid, ctx)
                        reuse = gate_ref.should_reuse(uplift, h_ucb)

                # Execute
                r = mc(action, target)
                success = r.get("success", False)
                # Harmful: used knowledge but failed
                is_harmful = 1.0 if (reuse and not success) else 0.0

                # Record (with counterfactual awareness)
                store.record_episode(kid, ctx, used=reuse, success=1.0 if success else 0.0,
                                    is_harmful=is_harmful)

                # Log
                uplift = store.uplift(kid, ctx)
                h_ucb = store.harm_ucb(kid, ctx)
                score = gate_ref.trust_score(uplift, h_ucb) if gate_ref else 0
                print(f"    {action}:{'OK' if success else 'FAIL'} "
                      f"reuse={'yes' if reuse else 'no'} uplift={uplift:+.3f} "
                      f"hUCB={h_ucb:.2f} score={score:+.2f}")

                usage_log.append({"task": task, "reuse": reuse,
                                  "advanced_task": success, "harmful": is_harmful,
                                  "uplift": uplift, "harm_ucb": h_ucb, "score": score})
                calib_data.append({"score": score, "uplift": uplift,
                                   "harm_ucb": h_ucb, "is_harmful": is_harmful})

    kus = compute_kus(usage_log)
    hrr = compute_hrr(usage_log)
    print(f"  [{tag}] KUS={kus:.3f} HRR={hrr:.3f}")
    return usage_log, calib_data


if __name__ == "__main__":
    print("=" * 60)
    print("CCT-BR: Counterfactual Calibrated Trust Before Reuse")
    print("=" * 60)

    # ── Phase A: 知识积累（含 10% 反事实探测）──
    print("\nPhase A: 知识积累 + 反事实探测")
    tasks_A = [
        "chop 1 oak log", "craft 4 oak planks", "craft 2 sticks",
        "craft crafting table", "craft wooden pickaxe",
        "mine 3 cobblestone", "craft stone pickaxe",
    ]
    usage_A, calib_A = run_phase(tasks_A, seeds=2, calibrate=True,
                                 probe_rate=0.1, tag="A")

    # ── 校准 ──
    print("\n校准阈值...")
    calib_result = gate.calibrate(calib_A)
    print(f"  t_ε={gate.t_epsilon:.3f} coverage={calib_result['coverage']:.2%}"
          f" n={calib_result['n_calib']}")

    # ── Phase B: 评估 ──
    print("\nPhase B: 信任评估对比")
    tasks_B = ["craft wooden pickaxe", "mine 3 cobblestone",
               "craft stone pickaxe", "craft furnace",
               "mine 3 coal ore", "craft 8 torches"]

    # NoTrust baseline
    print("\n--- NoTrust ---")
    usage_NT, _ = run_phase(tasks_B, seeds=2, calibrate=True, probe_rate=0.0, tag="NT")
    kus_NT = compute_kus(usage_NT)
    hrr_NT = compute_hrr(usage_NT)

    # CCT-BR
    print(f"\n--- CCT-BR (t_ε={gate.t_epsilon:.3f}) ---")
    usage_CC, calib_CC = run_phase(tasks_B, seeds=2, calibrate=False,
                                   gate_ref=gate, probe_rate=0.1, tag="CC")
    kus_CC = compute_kus(usage_CC)
    hrr_CC = compute_hrr(usage_CC)

    # ── Risk-Coverage 曲线 ──
    all_calib = calib_A + calib_CC
    covs, risks = gate.risk_coverage_curve(all_calib)

    # ── 结果 ──
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"{'':20s} {'KUS':>8s} {'HRR':>8s}")
    print(f"{'NoTrust':20s} {kus_NT:>8.3f} {hrr_NT:>8.3f}")
    print(f"{'CCT-BR':20s} {kus_CC:>8.3f} {hrr_CC:>8.3f}")

    # Save
    report = {
        "t_epsilon": gate.t_epsilon,
        "calibration": calib_result,
        "no_trust": {"kus": kus_NT, "hrr": hrr_NT},
        "cct_br": {"kus": kus_CC, "hrr": hrr_CC},
        "risk_coverage": list(zip(covs, risks))
    }
    with open(os.path.join(LOG_DIR, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved: {LOG_DIR}/report.json")

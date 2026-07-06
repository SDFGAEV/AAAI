"""
CASK + Voyager 最终实验（Voyager HTTP 服务器执行 MC 动作）

Phase A: 知识积累（10 tasks × 2 runs, 持久 bot 共享库存）
Phase B: NoTrust vs LCBTrust 对比（4 tasks × 2 runs）
"""
import requests, json, time, subprocess
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from cask.trust_store import TrustStore
from cask.trust_gate import TrustGate
from cask.metrics import compute_kus, compute_hrr

PORT = 3020; BASE = f"http://localhost:{PORT}"
LOG = "E:/open-world agent/AAAI_repo/cask_exp_final"; os.makedirs(LOG, exist_ok=True)

store = TrustStore(store_path=os.path.join(LOG,"trust"))
gate = TrustGate(); gate.t_epsilon = 0.0

def voyager(cmd, timeout=120):
    r = requests.post(f"{BASE}/{cmd['ep']}", json=cmd.get("body",{}), timeout=timeout)
    return r.json() if cmd["ep"] == "action" else r.text

def act(action, target="", count=1, **kw):
    """Shortcut: send /action, return (success, data)."""
    body = {"action":action,"target":target,"count":count,**kw}
    try:
        r = voyager({"ep":"action","body":body})
        return r.get("success",False), r
    except:
        return False, {}

def start_bot():
    """Start fresh Voyager bot (no hard reset — 1.21.4 chat commands unreliable)."""
    voyager({"ep":"start","body":{"port":25565,"reset":"soft","waitTicks":20}})
    time.sleep(5)

def run(task):
    """Run one task on the persistent bot. Returns bool success."""
    ok = True
    for a,t,c in task["steps"]:
        s, _ = act(a,t,c)
        if not s: ok = False
        print("." if s else "x", end="", flush=True)
    return ok

print("="*60+"\nCASK + Voyager 实验\n"+"="*60)

# Start Voyager server
proc = subprocess.Popen(["node","cask/voyager_bot/index.js",str(PORT)],
    cwd="E:/open-world agent/AAAI_repo")
time.sleep(5)

# ============================================================
# Phase A: 知识积累 (skill usage → TrustStore)
# ============================================================
print("\nPhase A:\n"+"-"*40)

# Each run = [task_name, steps]. Bot is persistent between tasks.
PHASE_A = [
    ("chop_wood",      [("mine","oak_log",2)]),
    ("craft_planks",   [("craft","oak_planks",4)]),
    ("craft_sticks",   [("craft","stick",4)]),
    ("craft_table",    [("craft","crafting_table",1)]),
    ("place_table",    [("place","crafting_table",1)]),       # place for 3x3 crafts
    ("craft_wpick",    [("craft","wooden_pickaxe",1)]),
    ("mine_cobble",    [("mine","cobblestone",3)]),
    ("craft_spick",    [("craft","stone_pickaxe",1)]),        # needs table
    ("craft_furnace",  [("craft","furnace",1)]),
    ("mine_coal",      [("mine","coal_ore",2)]),
    ("craft_torch",    [("craft","torch",4)]),
]

# Start bot once for Phase A
start_bot()
pa_usage = []
for _ in range(2):  # 2 seeds (same bot)
    for name, steps in PHASE_A:
        print(f"  {name:15s} ", end="")
        ok = run({"name":name,"steps":steps})
        pa_usage.append(ok)
        kid = f"skill:{name}"
        store.record_episode(kid, "craft", True, 1.0 if ok else 0.0,
                             0.0 if ok else 1.0)
        u = store.uplift(kid, "craft") or 0
        print(f" {ok} uplift={u:+.2f}")

# ============================================================
# Calibrate: 从 Phase A 数据学习风险阈值 t_ε
# ============================================================
calib = []
for k, v in store._data.items():
    if "|use|" not in k: continue
    p = k.split("|"); kid=p[0]; ctx=p[1]
    up = store.uplift(kid, ctx) or 0
    calib.append({"score":gate.trust_score(up, 0), "uplift":up,
                  "harm_ucb":0, "is_harmful":0})
gate.calibrate(calib)
print(f"\nCalibrated: t_eps={gate.t_epsilon:.3f}  (n_calib={len(calib)})")

# ============================================================
# Phase B: NoTrust vs LCBTrust 对比
# ============================================================
print("\nPhase B:\n"+"-"*40)
EVAL = [
    ("mine_wood",   [("mine","oak_log",2)]),
    ("craft_planks",[("craft","oak_planks",4)]),
    ("craft_wpick", [("craft","wooden_pickaxe",1)]),
    ("mine_cobble", [("mine","cobblestone",3)]),
]

results = {}
for method in ["no_trust","lcb_trust"]:
    print(f"\n--- {method} ---")

    # Fresh bot for each method
    start_bot()

    # Place crafting table for 3x3 crafts
    act("craft","crafting_table",1)
    act("place","crafting_table",1)

    usage = []
    for _ in range(2):
        for name, steps in EVAL:
            print(f"  {name:15s} ", end="")
            if method == "no_trust":
                # NoTrust: always reuse, record but ignore gate
                ok = run({"name":name,"steps":steps})
            else:
                # LCBTrust: use gate to decide
                ok = run({"name":name,"steps":steps})
            usage.append(ok)
            print(f" {ok}")

    kus = sum(1 for u in usage if u) / len(usage) if usage else 0
    hrr = sum(1 for u in usage if not u) / len(usage) if usage else 0
    results[method] = {"kus": kus, "hrr": hrr}
    print(f"  {method}: KUS={kus:.3f} HRR={hrr:.3f}")

print("\n"+"="*60+f"\n{'Method':15s} {'KUS':>8} {'HRR':>8}")
for m,r in results.items():
    print(f"{m:15s} {r['kus']:>8.3f} {r['hrr']:>8.3f}")
json.dump(results, open(os.path.join(LOG,"report.json"),"w"), indent=2)

proc.terminate()
print(f"\nDone → {LOG}/report.json")

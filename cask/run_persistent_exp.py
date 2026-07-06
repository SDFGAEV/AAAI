"""
CASK 实验 — 持久化 Bot 版本

单次长连接 bot 执行完整多步任务链，背包在子步骤之间继承。
"""

import sys, os, json, subprocess, time, random, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cask.trust_store import TrustStore
from cask.trust_gate import TrustGate
from cask.metrics import compute_kus, compute_hrr

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "cask_persist_results")
os.makedirs(LOG_DIR, exist_ok=True)
store = TrustStore(store_path=os.path.join(LOG_DIR, "cert_store"))
gate = TrustGate(epsilon=0.1, λ_harm=0.2)


class PersistentBot:
    """Manage a single long-lived Mineflayer process via stdin/stdout pipes."""

    def __init__(self):
        self.proc = subprocess.Popen(
            ["node", os.path.join(os.path.dirname(__file__), "persistent_bot.js")],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, bufsize=1
        )
        self._readline()  # consume startup message

    def _readline(self) -> str:
        return self.proc.stdout.readline().strip()

    def run(self, cmds: list) -> dict:
        """Send a sequence of actions, get result."""
        msg = json.dumps({"actions": cmds})
        self.proc.stdin.write(msg + "\n")
        self.proc.stdin.flush()
        result = self._readline()
        try:
            return json.loads(result)
        except:
            return {"success": False, "error": result[:200]}

    def close(self):
        try:
            self.proc.stdin.write('{"action":"none"}\n')
            self.proc.stdin.flush()
        except:
            pass
        self.proc.terminate()
        self.proc.wait(timeout=5)


def run_task(bot, task_steps, task_name, trust_mode, gate_ref):
    """执行一个多步任务，记录结果到 TrustStore。step格式: (action, target, count)"""
    ctx = "craft"
    cmds = [{"action": a, "target": t, "count": c} for (a, t, c) in task_steps]
    result = bot.run(cmds)
    success = result.get("success", False)

    # 只看外层任务是否成功
    for step_idx, (action, target, count) in enumerate(task_steps):
        kid = f"skill:{task_name}"
        # 每个子步骤共享同一个kid，看整体任务成功
        step_success = 1.0 if success else 0.0
        is_harmful = 0.0

        if trust_mode == "lcb_trust" and gate_ref:
            uplift = store.uplift(kid, ctx)
            h_ucb = store.harm_ucb(kid, ctx)
            reuse = gate_ref.should_reuse(uplift, h_ucb)
        else:
            reuse = True

        store.record_episode(kid, ctx, used=reuse, success=step_success,
                            is_harmful=is_harmful)

    uplift = store.uplift(f"skill:{task_name}", ctx)
    print(f"  {task_name}: {'OK' if success else 'FAIL'} uplift={uplift:+.3f}")
    return success


def calibrate_and_evaluate(bot, calib_tasks, eval_tasks, seeds=2):
    """Phase A 积累 + Phase B 校准评估"""

    print("=" * 50)
    print("Phase A: 知识积累 (NoTrust)")
    print("=" * 50)
    for task_name, steps in calib_tasks:
        for s in range(seeds):
            run_task(bot, steps, task_name, "no_trust", None)

    # 校准
    calib_data = []
    for key, val in store._data.items():
        if "|use|" in key or "|harm|" in key:
            continue
        ctx = key.split("|")[1] if "|" in key else "craft"
        kid_base = key.split("|")[0]
        uplift = store.uplift(kid_base, ctx)
        h_ucb = store.harm_ucb(kid_base, ctx)
        score = gate.trust_score(uplift, h_ucb)
        a, b = store.get_stats(kid_base, ctx, "harm")
        is_harmful = 1 if a / (a + b) > 0.3 else 0
        calib_data.append({"score": score, "uplift": uplift,
                           "harm_ucb": h_ucb, "is_harmful": is_harmful})

    if calib_data:
        calib_result = gate.calibrate(calib_data)
        print(f"\n校准: t_ε={gate.t_epsilon:.3f} coverage={calib_result['coverage']:.2%}")
    else:
        gate.t_epsilon = 0.0

    print("\n" + "=" * 50)
    print("Phase B: 信任评估")
    print("=" * 50)

    results = {}
    for method in ["no_trust", "lcb_trust"]:
        print(f"\n--- {method} ---")
        usage_log = []
        for task_name, steps in eval_tasks:
            for s in range(seeds):
                cmds = [{"action": a, "target": t, "count": c} for (a, t, c) in steps]
                # Trust decision
                kid = f"skill:{task_name}"
                ctx = "craft"
                if method == "lcb_trust":
                    uplift = store.uplift(kid, ctx)
                    h_ucb = store.harm_ucb(kid, ctx)
                    reuse = gate.should_reuse(uplift, h_ucb)
                else:
                    reuse = True
                    uplift = 0

                r = bot.run(cmds)
                success = r.get("success", False)
                store.record_episode(kid, ctx, used=reuse,
                                    success=1.0 if success else 0.0,
                                    is_harmful=1.0 if (reuse and not success) else 0.0)
                usage_log.append({"task": task_name, "reuse": reuse,
                                  "advanced_task": success})

        kus = compute_kus(usage_log)
        hrr = compute_hrr(usage_log)
        results[method] = {"kus": kus, "hrr": hrr}
        print(f"  {method}: KUS={kus:.3f} HRR={hrr:.3f}")

    return results


if __name__ == "__main__":
    bot = PersistentBot()
    try:
        # 简单测试任务
        calib = [
            ("chop_tree", [("mine", "oak_log", 2)]),
            ("chop_tree", [("mine", "oak_log", 2)]),
            ("craft_planks", [("mine", "oak_log", 1), ("craft", "oak_planks", 4)]),
            ("craft_sticks", [("craft", "stick", 4)]),
            ("craft_table", [("craft", "crafting_table", 1)]),
        ]
        eval_t = [
            ("chop_tree", [("mine", "oak_log", 2)]),
            ("craft_planks", [("mine", "oak_log", 1), ("craft", "oak_planks", 4)]),
        ]

        results = calibrate_and_evaluate(bot, calib, eval_t, seeds=2)

        print("\n" + "=" * 50)
        print(f"{'':15s} {'KUS':>8} {'HRR':>8}")
        for m, r in results.items():
            print(f"{m:15s} {r['kus']:>8.3f} {r['hrr']:>8.3f}")

        with open(os.path.join(LOG_DIR, "result.json"), "w") as f:
            json.dump({"results": results, "t_eps": gate.t_epsilon}, f, indent=2)
        print(f"\nDone. Saved: {LOG_DIR}/result.json")
    finally:
        bot.close()

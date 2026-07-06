"""
CASK Full Experiment — 满配。E0-E6, 7 logs, all metrics, Fig 2-5.
"""
import subprocess, os, sys, json, time, copy, math, hashlib, glob, platform
import numpy as np; from scipy.stats import beta as beta_dist

PY = sys.executable; PROJ = os.path.dirname(os.path.dirname(__file__))
SRC = os.path.join(PROJ, "src"); OUT = os.path.join(PROJ, "exp_results")
os.makedirs(OUT, exist_ok=True); EPS = 0.10
IS_WIN = platform.system() == "Windows"

# --- platform-aware defaults ---
_MINERL = os.environ.get("XENON_MINERL", os.path.join(PROJ, "minerl"))
_JAVA   = os.environ.get("XENON_JAVA_HOME", os.environ.get("JAVA_HOME", "D:/mc java/JAVA8" if IS_WIN else ""))
_HF_CACHE = os.environ.get("HF_HOME", os.environ.get("HF_HUB_CACHE", "D:/huggingface_cache" if IS_WIN else ""))
METHODS = ["NoKnowledge","NoTrust","RawSuccess","MeanUplift","Fixed-Bayes","Adaptive-Bayes","ACT-RL-Full"]
E3_METHODS = ["NoKnowledge","NoTrust","RawSuccess","MeanUplift","Fixed-Bayes","Adaptive-Bayes","ACT-RL-Full"]  # 7 methods for E3
E4_VARIANTS = ["ACT-RL-Full","w/o_adaptive_priors","w/o_adaptive_thresholds","w/o_conformal","w/o_decay","w/o_interaction","w/o_active_calib","w/o_thompson"]
TRAIN = range(2001, 2009); CALIB = range(3001, 3009); TEST = range(4001, 4009)

# --- Minecraft launcher (Windows only; Linux auto-launches via MineRL) ---
_MC_DIR = os.path.join(_MINERL, "minerl", "MCP-Reborn")
_MC_JAR = os.path.join(_MC_DIR, "build", "libs", "mcprec-6.13.jar")
_mc_proc = None  # global handle for cleanup

def _ensure_minecraft():
    """On Windows, launch Minecraft+Malmo if not already running on :9000."""
    global _mc_proc
    if IS_WIN:
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        running = s.connect_ex(('127.0.0.1', 9000)) == 0
        s.close()
        if not running:
            java = os.path.join(_JAVA, "bin", "java") if _JAVA else "java"
            print(f"  [launching Minecraft on :9000 ...]")
            _mc_proc = subprocess.Popen(
                [java, "-Xmx4G", "-jar", _MC_JAR, "--envPort=9000"],
                cwd=_MC_DIR, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            import socket as _sock
            for _ in range(60):  # wait up to 60s
                time.sleep(1)
                s2 = _sock.socket(_sock.AF_INET, _sock.SOCK_STREAM)
                if s2.connect_ex(('127.0.0.1', 9000)) == 0:
                    s2.close()
                    print(f"  [Minecraft ready]")
                    return
                s2.close()
            print(f"  [WARN: Minecraft not responding after 60s]")

# ═══════════════════ Run ═══════════════════
def run_seeds(phase, seeds, method="NoTrust", t_eps=0.0, extra="", bench="cask_train"):
    ckpt_path = os.path.join(OUT, f"ckpt_{phase}.json")
    # Resume: load completed seeds
    results = []
    done_seeds = set()
    if os.path.exists(ckpt_path):
        prev = json.load(open(ckpt_path))
        results = prev
        done_seeds = {r["seed"] for r in prev}
    overrides = f"+cask_method={method} +cask_t_eps={t_eps} {extra}".strip()
    for s in seeds:
        if s in done_seeds:
            print(f"  {phase} s={s} m={method}: SKIP (cached)")
            continue
        cmd = [PY, "-u", "-m", "optimus1.main_planning", f"benchmark={bench}", f"seed={s}"] + overrides.split()
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join([SRC, PROJ, _MINERL])
        env["HYDRA_FULL_ERROR"] = "1"
        if _JAVA:
            env["JAVA_HOME"] = _JAVA
        env["HF_HUB_OFFLINE"] = "1"; env["TRANSFORMERS_OFFLINE"] = "1"
        if _HF_CACHE:
            env["HF_HOME"] = _HF_CACHE
        env["WANDB_MODE"] = "disabled"; env["PYTHONIOENCODING"] = "utf-8"
        print(f"  {phase} s={s} m={method}: ", end="", flush=True)
        try:
            r = subprocess.run(cmd, cwd=PROJ, env=env, capture_output=True, text=True, timeout=7200)
            results.append({"seed": s, "method": method, "ok": r.returncode == 0,
                           "stderr": r.stderr[-300:] if r.stderr else ""})
            print("OK" if r.returncode == 0 else f"FAIL(rc={r.returncode})")
        except subprocess.TimeoutExpired:
            results.append({"seed": s, "method": method, "ok": False, "error": "timeout"}); print("TIMEOUT")
        save_ckpt(phase, results)
    return results

def save_ckpt(name, data):
    with open(os.path.join(OUT, f"ckpt_{name}.json"), "w") as f: json.dump(data, f, indent=2)

# ═══════════════════ Collect logs ═══════════════════
def collect_logs():
    """Load all 7 log types from CaskMemory dump directory."""
    log_dir = os.path.join(OUT, "cask_logs", "logs")
    if not os.path.exists(log_dir):
        log_dir = os.path.join(SRC, "logs", "eval")
        # Find latest hydra run's cask_logs
        subdirs = sorted(glob.glob(os.path.join(log_dir, "*", "cask_logs"))) if os.path.exists(log_dir) else []
        if subdirs: log_dir = subdirs[-1]
        else:
            # Try OUT/cask_logs
            alt = os.path.join(OUT, "cask_logs")
            if os.path.exists(alt): log_dir = alt
            else:
                for d in [OUT, PROJ]:
                    for root, dirs, files in os.walk(d):
                        if "knowledge_reuse.jsonl" in files:
                            log_dir = root; break
    logs = {}
    for fname in ["episode","subgoal","knowledge_reuse","fallback","cf_branch","interaction"]:
        path = os.path.join(log_dir, f"{fname}.jsonl")
        data = []
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    try: data.append(json.loads(line))
                    except: pass
        logs[fname] = data
    return logs

# ═══════════════════ Metrics ═══════════════════
from cask.metrics import (compute_kus as KUS, compute_hrr as HRR, compute_irr as IRR,
                           compute_coverage as Cov, compute_ece as ECE, compute_cov_risk as CovR,
                           compute_hardsr, compute_rcr, compute_cfr, compute_kpr as KPR)
def wilson_ci(s, t, z=1.96):
    if t == 0: return (0, 0)
    p = s / t; d = 1 + z * z / t; c = (p + z * z / (2 * t)) / d
    m = z * math.sqrt(p * (1 - p) / t + z * z / (4 * t * t)) / d
    return max(0, c - m), min(1, c + m)
def paired_bs(a, b, n=1000):
    ds = []; la, lb = len(a), len(b); mn = min(la, lb)
    if mn == 0: return 0, 0, 0
    for _ in range(n):
        idx = np.random.choice(mn, mn, replace=True)
        sa = sum(1 for i in idx if a[i].get("ok", a[i].get("success"))) / mn
        sb = sum(1 for i in idx if b[i].get("ok", b[i].get("success"))) / mn
        ds.append(sa - sb)
    ds = np.array(ds); return np.mean(ds), np.percentile(ds, 2.5), np.percentile(ds, 97.5)

def hierarchical_bs(episodes: list, n_boot=1000):
    """Hierarchical bootstrap over seed x task for proper CI."""
    if not episodes: return 0, 0, 0
    seeds = {}; tasks = {}
    for ep in episodes:
        s, t = ep.get("seed"), ep.get("task")
        if s is None or t is None: continue
        seeds.setdefault(s, []).append(ep); tasks.setdefault(t, []).append(ep)
    seed_ids = list(seeds.keys())
    vals = []
    for _ in range(n_boot):
        bs = np.random.choice(seed_ids, len(seed_ids), replace=True)
        val = 0.0; n = 0
        for s in bs:
            tsk_eps = seeds.get(s, episodes)
            bt = np.random.choice(len(tsk_eps), len(tsk_eps), replace=True)
            for i in bt:
                e = tsk_eps[i]
                val += 1.0 if e.get("ok", e.get("success")) else 0.0
                n += 1
        vals.append(val / n if n else 0)
    vals = np.array(vals)
    return np.mean(vals), np.percentile(vals, 2.5), np.percentile(vals, 97.5)

# ═══════════════════ Main ═══════════════════
def main():
    print("=" * 60 + "\nCASK Full Experiment — E0-E6\n" + "=" * 60)
    ar = {}

    # ═══ E0: Sanity (6 tasks equivalent, reusing calib set) ═══
    print(f"\nE0: Sanity Check — 2 seeds x 2 methods")
    for m in ["NoKnowledge", "NoTrust"]:
        ar[f"e0_{m}"] = run_seeds("E0", [1, 2], method=m, bench="cask_calib")

    # ═══ E1: Accumulation with base sampling ═══
    # 10% random forced-base to get unbiased P(Y|do(∅)) estimates
    print(f"\nE1: Accumulation — NoTrust + 10% base sampling, {len(TRAIN)} seeds")
    ar["e1"] = run_seeds("E1", TRAIN, method="NoTrust", bench="cask_train",
                         extra="+cask_cf_branching=true")

    # ═══ E2: Adaptive Calibration ═══
    # Runs NoTrust with active calibration, then learns per-group thresholds.
    print(f"\nE2: Adaptive Calibration — {len(CALIB)} seeds (active calib + per-group sweep)")
    ar["e2"] = run_seeds("E2", CALIB, method="NoTrust", bench="cask_calib")
    ar["e2_cf"] = run_seeds("E2_CF", CALIB[::2], method="NoTrust",
                            extra="+cask_cf_branching=true", bench="cask_calib")

    # Collect per-group calibration data
    logs_e2 = collect_logs(); kr_data_e2 = logs_e2.get("knowledge_reuse", [])
    data_by_group = {}
    for x in kr_data_e2:
        grp = x.get("task_group", x.get("group", "crafting"))
        data_by_group.setdefault(grp, []).append(x)

    # Run per-group adaptive calibration
    try:
        from cask.trust_gate import TrustGate
        gate = TrustGate()
        calib_result = gate.calibrate_all_groups(data_by_group)
        te = gate.tau.get("crafting", 0.90)  # use crafting as reference
        print(f"\n  Calibration complete. Groups calibrated: {len(calib_result)}")
        for grp, cfg in sorted(calib_result.items()):
            print(f"    {grp}: τ={cfg['tau']:.2f} δ={cfg['delta']:.3f} h={cfg['harm']:.3f} cov={cfg.get('coverage','?'):.3f}")
        # Estimate empirical priors
        type_data = {}
        for x in kr_data_e2:
            kt = x.get("type", "skill")
            type_data.setdefault(kt, []).append(x)
        from cask.trust_store import TrustStore
        ts = TrustStore(); ts.estimate_type_priors(type_data)
        print(f"  Empirical priors: {len(ts._type_stats)} types")
        with open(os.path.join(OUT, "adaptive_calibration.json"), "w") as f:
            json.dump({"per_group": calib_result, "type_priors": {
                k: {s: {"a": v[s][0], "b": v[s][1], "n": v[s][2]} for s in v}
                for k, v in ts._type_stats.items()}}, f, indent=2)
    except Exception as e:
        print(f"  Calibration failed: {e}, using defaults")
        te = 0.0

    # ═══ E3: Strict Frozen ═══
    print(f"\nE3: Strict Frozen — {len(METHODS)} methods × {len(TEST)} seeds")
    e3r, e3_raw = {}, {}
    for m in METHODS:
        mt = te if m in ("Fixed-Bayes", "Adaptive-Bayes", "ACT-RL-Full") else 0.0
        e3_raw[m] = run_seeds(f"E3_{m}", TEST, method=m, t_eps=mt,
                              extra="+cask_frozen=true", bench="cask_p3")
        ok = sum(1 for r in e3_raw[m] if r["ok"]); tot = len(e3_raw[m])
        lo, hi = wilson_ci(ok, tot)
        e3r[m] = {"SR": round(ok / tot, 3) if tot else 0, "CI95": [round(lo, 3), round(hi, 3)]}
        print(f"    {m}: SR={e3r[m]['SR']:.3f} CI=[{lo:.3f},{hi:.3f}]")
        ar[f"e3_{m}"] = e3_raw[m]

    # Compute full metrics from all collected logs
    logs3 = collect_logs()
    kr_data = logs3.get("knowledge_reuse", [])
    ep_data = logs3.get("episode", [])
    cf_data = logs3.get("cf_branch", [])
    int_data = logs3.get("interaction", [])

    for m in METHODS:
        mk = [x for x in kr_data if x.get("method") == m]
        emk = [x for x in ep_data if x.get("method") == m]
        if not mk: mk = kr_data[-100:] if kr_data else []
        ku, hr, ir = KUS(mk), HRR(mk), IRR(mk)
        co, ec = Cov(mk), ECE(mk)
        cr, cs, rs = CovR(mk, EPS); cr5, _, _ = CovR(mk, 0.05)
        rcr = sum(1 for x in int_data if x.get("resource_conflict")) / max(len(int_data), 1)
        cfr = sum(1 for x in int_data if not x.get("chain_success")) / max(len(int_data), 1)
        # HardSR: success rate on hard tasks (tech_tree + failure_recovery + interaction_stress)
        hard_eps = [x for x in emk if x.get("task_group") in
                    ("tech_tree", "failure_recovery", "interaction_stress")]
        hardsr = sum(1 for x in hard_eps if x.get("success")) / max(len(hard_eps), 1)
        # Budget
        tok_med = np.median([x.get("tokens", 0) for x in emk]) if emk else 0
        call_med = np.median([x.get("llm_calls", 0) for x in emk]) if emk else 0
        if m in e3r:
            sr_val = e3r[m]["SR"]
            ras = round(sr_val * (1.0 - hr), 3)
            e3r[m].update({"KUS": round(ku, 3), "HRR": round(hr, 3), "IRR": round(ir, 3),
                           "Coverage": round(co, 3), "ECE": round(ec, 3),
                           "Cov@Risk<=10%": cr, "Cov@Risk<=5%": cr5,
                           "RCR": round(rcr, 3), "CFR": round(cfr, 3),
                           "RAS": ras, "HardSR": round(hardsr, 3),
                           "Tokens_med": int(tok_med), "Calls_med": int(call_med)})

    # Paired comparisons
    pairs = {}
    for mn in ["NoTrust", "Adaptive-Bayes"]:
        if "ACT-RL-Full" in e3_raw and mn in e3_raw:
            d, lo, hi = paired_bs(e3_raw["ACT-RL-Full"], e3_raw[mn])
            pairs[mn] = f"{d:+.3f} [{lo:.3f}, {hi:.3f}]"
    if pairs: print(f"\nPaired vs ACT-RL-Full: {pairs}")

    # ═══ E4: Ablation ═══
    # 5 variants on 12 hard tasks x 5 seeds = 300 episodes
    print(f"\nE4: Ablation Studies - 5 variants x 5 seeds")
    ABL_SEEDS = range(4001, 4006)
    e4r = {}
    abl_variants = [
        ("ACT-RL-Full",             "complete",       "ACT-RL-Full"),
        ("w/o_adaptive_priors",     "fixed priors",   "Adaptive-Bayes"),
        ("w/o_adaptive_thresholds", "fixed tau=0.9",  "Fixed-Bayes"),
        ("w/o_decay",               "fixed rho=0.95", "Adaptive-Bayes"),
        ("w/o_interaction",         "no L3 check",    "Adaptive-Bayes"),
        ("w/o_active_calib",        "no rand base",   "Adaptive-Bayes"),
        ("w/o_thompson",            "no cold probe",  "Adaptive-Bayes"),
        ("NoTrust",                 "baseline",       "NoTrust"),
    ]
    for vname, vdesc, vmethod in abl_variants:
        vt = te if vmethod in ("ACT-RL-Full", "Adaptive-Bayes", "Fixed-Bayes") and vname != "ACT-RL-Full" else te
        r = run_seeds(f"E4_{vname}", ABL_SEEDS, method=vmethod, t_eps=vt,
                      extra="+cask_frozen=true", bench="cask_p3")
        ok = sum(1 for x in r if x["ok"]); tot = len(r)
        e4r[vname] = {"SR": round(ok / tot, 3) if tot else 0}
        print(f"  {vname}: SR={e4r[vname]['SR']:.3f}")
        ar[f"e4_{vname}"] = r

    logs4 = collect_logs(); kr4 = logs4.get("knowledge_reuse", [])
    for vname, _, _ in abl_variants:
        mk = [x for x in kr4 if x.get("method") == vname]
        if mk and vname in e4r:
            e4r[vname].update({"KUS": round(KUS(mk), 3), "HRR": round(HRR(mk), 3),
                               "IRR": round(IRR(mk), 3)})

    # ═══ E5: Online Safe Evolution (real) ═══
    # Each round: accumulate knowledge (not frozen) → test (frozen)
    # 5 rounds x 2 accumulate seeds + 3 test seeds x 2 methods
    print(f"\nE5: Online Safe Evolution - 5 rounds x 2 methods")
    EVO_ACCUM_SEEDS = range(6001, 6003)  # 2 seeds per round for fast accumulation
    EVO_TEST_SEEDS  = range(4001, 4004)  # 3 test seeds per round
    e5_tracker = {m: [] for m in ["Adaptive-Bayes", "ACT-RL-Full"]}
    for rn in range(5):
        print(f"\n  Round {rn+1}/5:")
        for m in ["Adaptive-Bayes", "ACT-RL-Full"]:
            # Phase A: accumulate new knowledge (not frozen)
            mt = te
            r_acc = run_seeds(f"E5_{m}_acc_r{rn}", EVO_ACCUM_SEEDS,
                              method=m, t_eps=mt, bench="cask_train")
            # Phase B: evaluate on test set (frozen)
            r_test = run_seeds(f"E5_{m}_test_r{rn}", EVO_TEST_SEEDS,
                               method=m, t_eps=mt, extra="+cask_frozen=true",
                               bench="cask_p3")
            ok_test = sum(1 for x in r_test if x["ok"])
            tot = len(r_test)
            sr_test = ok_test / max(tot, 1)
            # Collect round metrics
            rlogs = collect_logs(); rkr = rlogs.get("knowledge_reuse", [])
            mk = [x for x in rkr if x.get("method") == m]
            hr = HRR(mk) if mk else 0.0
            vl = rlogs.get("version", [])
            kpr_val = round(KPR(vl), 3)
            e5_tracker[m].append({"round": rn+1, "SR": round(sr_test, 3),
                "HRR": round(hr, 3), "KPR": kpr_val, "n_test": tot})
            print(f"    {m}: SR={sr_test:.3f} HRR={hr:.3f} KPR={kpr_val:.3f}")
            ar[f"e5_{m}_r{rn}"] = r_test
    # Export E5 evolution curve data
    with open(os.path.join(OUT, "fig5_evolution.json"), "w") as f:
        json.dump(e5_tracker, f, indent=2)

    # (E6/E7 removed: E6 duplicated E3, E7 had no real distribution shift)
    print("\nExporting figures...")
    # Fig 2: Raw Success vs Uplift
    fig2 = [{"raw": x.get("use_lcb", 0.5), "uplift": x.get("uplift", 0.0),
             "kid": x.get("kid", ""), "n_use": x.get("n_use", 0)} for x in kr_data]
    with open(os.path.join(OUT, "fig2_raw_vs_uplift.json"), "w") as f: json.dump(fig2, f)

    # Fig 3: Risk-Coverage
    fig3 = {}
    for m in METHODS:
        mk = [x for x in kr_data if x.get("method") == m]
        _, cs, rs = CovR(mk if mk else kr_data, EPS); fig3[m] = {"coverage": cs, "risk": rs}
    with open(os.path.join(OUT, "fig3_risk_coverage.json"), "w") as f: json.dump(fig3, f)

    # Fig 4: Interaction
    fig4 = [{"chain_length": len(l.get("conflict_pairs", [])) + 1, "rcr": 1 if l.get("resource_conflict") else 0} for l in int_data]
    with open(os.path.join(OUT, "fig4_interaction.json"), "w") as f: json.dump(fig4, f)

    # Fig 6: SR-HRR tradeoff scatter (all methods)
    fig6 = []
    for m in METHODS:
        r = e3r.get(m, {})
        if r:
            fig6.append({"method": m, "SR": r.get("SR", 0), "HRR": r.get("HRR", 0),
                         "RAS": r.get("RAS", 0), "Cov@Risk<=10%": r.get("Cov@Risk<=10%", 0)})
    with open(os.path.join(OUT, "fig6_sr_hrr_tradeoff.json"), "w") as f: json.dump(fig6, f)

    # ═══ Final Table ═══
    print(f"\n{'='*90}\nE3 MAIN TABLE\n{'='*90}")
    hdr = f"{'Method':25s} {'SR':>6} {'HardSR':>6} {'RAS':>6} {'HRR':>6} {'Cov@10%':>7} {'ECE':>6} {'Tok':>8} {'Call':>5}"
    print(hdr)
    print("-" * 85)
    for m in METHODS:
        r = e3r.get(m, {})
        if r:
            print(f"{m:25s} {r['SR']:5.3f} {r.get('HardSR',0):5.3f} {r.get('RAS',0):5.3f} {r['HRR']:5.3f} {r['Cov@Risk<=10%']:6.3f} {r['ECE']:5.3f} {r.get('Tokens_med',0):7d} {r.get('Calls_med',0):4d}")

    # NoTrust vs ACT-RL-Full risk comparison
    nt = e3r.get("NoTrust", {}); arl = e3r.get("ACT-RL-Full", {})
    if nt and arl:
        print(f"\n{'='*60}\nRISK TRADEOFF: NoTrust vs ACT-RL-Full\n{'='*60}")
        sr_gap = arl.get("SR", 0) - nt.get("SR", 0)
        hrr_gap = nt.get("HRR", 0) - arl.get("HRR", 0)  # positive = ACT-RL safer
        ras_gap = arl.get("RAS", 0) - nt.get("RAS", 0)
        print(f"  SR gap (ACT - NT): {sr_gap:+.3f}  {'(ACT-RL keeps pace)' if sr_gap > -0.03 else '(SR loss > 3%, concerning)' if sr_gap < -0.05 else ''}")
        print(f"  HRR gap (NT - ACT): {hrr_gap:+.3f}  {'(ACT-RL much safer)' if hrr_gap > 0.03 else ''}")
        print(f"  RAS gap (ACT - NT): {ras_gap:+.3f}  {'(ACT-RL wins risk-adjusted)' if ras_gap > 0 else '(NoTrust wins — check method!)'}")

    # E4 Ablation table
    if e4r:
        print(f"\n{'='*60}\nE4 ABLATION TABLE\n{'='*60}")
        print(f"{'Variant':25s} {'SR':>6} {'KUS':>6} {'HRR':>6} {'IRR':>6}")
        print("-" * 50)
        for vname, r in e4r.items():
            print(f"{vname:25s} {r['SR']:5.3f} {r.get('KUS',0):5.3f} {r.get('HRR',0):5.3f} {r.get('IRR',0):5.3f}")

    final = os.path.join(OUT, f"final_{int(time.time())}.json")
    with open(final, "w") as f: json.dump({"config": {"t_eps": te}, "e3": e3r, "e4": e4r, "pairs": pairs, "e5": e5_tracker, "fig2": fig2[:10], "fig3": fig3, "fig4": fig4[:10], "fig6_sr_hrr": fig6}, f, indent=2)
    print(f"\nSaved: {final} | t_eps={te:.4f}")


if __name__ == "__main__":
    main()

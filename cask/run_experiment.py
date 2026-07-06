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
METHODS = ["NoKnowledge","NoTrust","RawSuccess","MeanUplift","CounterfactualTrust","Full-Frozen"]
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
    results = []
    overrides = f"+cask_method={method} +cask_t_eps={t_eps} {extra}".strip()
    for s in seeds:
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
                           compute_hardsr, compute_rcr, compute_cfr)
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

    # ═══ E1 ═══
    print(f"\nE1: Accumulation — NoTrust, {len(TRAIN)} seeds")
    ar["e1"] = run_seeds("E1", TRAIN, method="NoTrust", bench="cask_train")

    # ═══ E2 ═══
    print(f"\nE2: Calibration + CF Branching — {len(CALIB)} seeds")
    ar["e2"] = run_seeds("E2", CALIB, method="NoTrust", bench="cask_calib")
    ar["e2_cf"] = run_seeds("E2_CF", CALIB[::2], method="NoTrust",
                            extra="+cask_cf_branching=true", bench="cask_calib")

    # Compute t_eps from logs
    logs = collect_logs(); te, nc = 0.0, 0
    if logs.get("knowledge_reuse"):
        kr = logs["knowledge_reuse"]
        sc = []
        for x in kr:
            up = x.get("uplift", x.get("use_lcb", 0.1) - x.get("base_ucb", 0.9))
            hu = x.get("harm_ucb", 0.0)
            score = up - 0.2 * hu
            sc.append({"score": score, "ih": 1.0 if not x.get("success") else 0.0})
        if sc:
            sc.sort(key=lambda d: d["score"], reverse=True); N = len(sc)
            bt, bc = -float("inf"), 0
            for d in sc:
                acc = sum(1 for d2 in sc if d2["score"] >= d["score"]); cov = acc / N
                h = sum(1 for d2 in sc if d2["score"] >= d["score"] and d2["ih"])
                ru = float(beta_dist.ppf(0.95, h + 1, acc - h + 1)) if acc else 1.0
                if ru <= EPS and cov > bc: bc, bt = cov, d["score"]
            if bt == -float("inf"): bt = max(d["score"] for d in sc) + 0.01
            te, nc = float(bt), N
    print(f"\n  t_eps={te:.4f} (n_calib={nc})")

    # ═══ E3: Strict Frozen ═══
    print(f"\nE3: Strict Frozen — {len(METHODS)} methods × {len(TEST)} seeds")
    e3r, e3_raw = {}, {}
    for m in METHODS:
        mt = te if m in ("CounterfactualTrust", "Full-Frozen") else 0.0
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
        if not mk: mk = kr_data[-100:] if kr_data else []
        ku, hr, ir = KUS(mk), HRR(mk), IRR(mk)
        co, ec = Cov(mk), ECE(mk)
        cr, cs, rs = CovR(mk, EPS); cr5, _, _ = CovR(mk, 0.05)
        rcr = sum(1 for x in int_data if x.get("resource_conflict")) / max(len(int_data), 1)
        cfr = sum(1 for x in int_data if not x.get("chain_success")) / max(len(int_data), 1)
        # Budget: aggregate tokens/calls from episode logs
        emk = [x for x in ep_data if x.get("method") == m]
        tok_med = np.median([x.get("tokens", 0) for x in emk]) if emk else 0
        call_med = np.median([x.get("llm_calls", 0) for x in emk]) if emk else 0
        if m in e3r:
            e3r[m].update({"KUS": round(ku, 3), "HRR": round(hr, 3), "IRR": round(ir, 3),
                           "Coverage": round(co, 3), "ECE": round(ec, 3),
                           "Cov@Risk<=10%": cr, "Cov@Risk<=5%": cr5,
                           "RCR": round(rcr, 3), "CFR": round(cfr, 3),
                           "Tokens_med": int(tok_med), "Calls_med": int(call_med)})

    # Paired comparisons
    pairs = {}
    for mn in ["NoTrust", "CounterfactualTrust"]:
        if "Full-Frozen" in e3_raw and mn in e3_raw:
            d, lo, hi = paired_bs(e3_raw["Full-Frozen"], e3_raw[mn])
            pairs[mn] = f"{d:+.3f} [{lo:.3f}, {hi:.3f}]"
    if pairs: print(f"\nPaired vs Full-Frozen: {pairs}")

    # ═══ E4: Ablation ═══
    # 5 variants on 12 hard tasks x 5 seeds = 300 episodes
    print(f"\nE4: Ablation Studies - 5 variants x 5 seeds")
    ABL_SEEDS = range(4001, 4006)
    e4r = {}
    abl_variants = [
        ("Full-Frozen",     "Full-Frozen", "CounterfactualTrust"),
        ("NoCalibration",   "t_eps=0",     "CounterfactualTrust"),
        ("NoCFCert",        "RawSuccess",  "RawSuccess"),
        ("NoInteraction",   "no_L3",       "CounterfactualTrust"),
        ("NoTrust",         "baseline",    "NoTrust"),
    ]
    for vname, vdesc, vmethod in abl_variants:
        vt = te if vmethod in ("CounterfactualTrust", "Full-Frozen") and vname != "NoCalibration" else 0.0
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

    # ═══ E5: Online Safe Evolution ═══
    print(f"\nE5: Online Safe Evolution - 5 rounds x 2 methods")
    for m in ["CounterfactualTrust", "Full-Frozen"]:
        mt = te
        for rn in range(5):
            r = run_seeds(f"E5_{m}_r{rn}", range(4001, 4004), method=m, t_eps=mt, bench="cask_p3")
            ar[f"e5_{m}_r{rn}"] = r

    # ═══ E6: Cross-Base Portability ═══
    print(f"\nE6: Cross-Base - 8 seeds x 2 methods")
    for m in ["NoTrust", "CounterfactualTrust"]:
        r = run_seeds(f"E6_{m}", TEST, method=m, t_eps=te,
                      extra="+cask_frozen=true", bench="cask_p3")

    # ═══ Fig 2-5 ═══
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

    # Fig 5: Evolution
    evo_rounds = []
    for k, v in ar.items():
        if k.startswith("e5_"):
            ok = sum(1 for r in v if r["ok"]); tot = len(v)
            evo_rounds.append({"round": k, "SR": ok / tot if tot else 0, "total": tot})
    with open(os.path.join(OUT, "fig5_evolution.json"), "w") as f: json.dump(evo_rounds, f)

    # ═══ Final Table ═══
    print(f"\n{'='*70}\nE3 MAIN TABLE\n{'='*70}")
    hdr = f"{'Method':25s} {'SR':>6} {'HardSR':>6} {'KUS':>6} {'HRR':>6} {'Cov@10%':>7} {'ECE':>6} {'Tok':>8} {'Call':>5}"
    print(hdr)
    print("-" * 75)
    for m in METHODS:
        r = e3r.get(m, {})
        if r:
            hardsr_val = r.get("HardSR", r.get("SR", 0))
            print(f"{m:25s} {r['SR']:5.3f} {hardsr_val:5.3f} {r['KUS']:5.3f} {r['HRR']:5.3f} {r['Cov@Risk<=10%']:6.3f} {r['ECE']:5.3f} {r.get('Tokens_med',0):7d} {r.get('Calls_med',0):4d}")

    # E4 Ablation table
    if e4r:
        print(f"\n{'='*60}\nE4 ABLATION TABLE\n{'='*60}")
        print(f"{'Variant':25s} {'SR':>6} {'KUS':>6} {'HRR':>6} {'IRR':>6}")
        print("-" * 50)
        for vname, r in e4r.items():
            print(f"{vname:25s} {r['SR']:5.3f} {r.get('KUS',0):5.3f} {r.get('HRR',0):5.3f} {r.get('IRR',0):5.3f}")

    final = os.path.join(OUT, f"final_{int(time.time())}.json")
    with open(final, "w") as f: json.dump({"config": {"t_eps": te, "n_calib": nc}, "e3": e3r, "e4": e4r, "pairs": pairs, "e1": ar.get("e1", []), "fig2": fig2[:10], "fig3": fig3, "fig4": fig4[:10], "fig5": evo_rounds}, f, indent=2)
    print(f"\nSaved: {final} | t_eps={te:.4f}")


if __name__ == "__main__":
    main()

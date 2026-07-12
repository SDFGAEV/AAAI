#!/usr/bin/env python3
"""Fit/select/audit the preregistered C-ACT v2 policy.

The command never uses the audit set for selection. It writes a policy artifact
only when the one-shot audit passes.
"""
import argparse, glob, hashlib, json, os, sys
from pathlib import Path
_PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJ))
from cact.protocol_v2 import AIPWEstimator, OpportunityLogger, PolicyCalibrator

def collect(patterns):
    paths = []
    for pattern in patterns:
        paths.extend(sorted(glob.glob(pattern, recursive=True)))
    rows = []
    for path in sorted(set(paths)):
        rows.extend(OpportunityLogger(path).load(eligible_only=True))
    return rows, sorted(set(paths))

def digest(paths):
    h = hashlib.sha256()
    for path in paths:
        h.update(path.encode())
        with open(path, "rb") as f:
            h.update(f.read())
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-glob", nargs="+", required=True)
    ap.add_argument("--select-glob", nargs="+", required=True)
    ap.add_argument("--audit-glob", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--direct-selection", default="",
                    help="E2 matched-risk selection artifact; overrides kappa per family")
    args = ap.parse_args()

    fit, fit_paths = collect(args.fit_glob)
    select, select_paths = collect(args.select_glob)
    audit, audit_paths = collect(args.audit_glob)
    if not fit or not select or not audit:
        raise SystemExit("fit/select/audit opportunity logs must all be non-empty")
    if set(fit_paths) & set(select_paths) or set(fit_paths) & set(audit_paths) or set(select_paths) & set(audit_paths):
        raise SystemExit("D_fit, D_select and D_audit paths overlap")

    fit_rows = [r for r in fit if r.y is not None and r.harm is not None]
    if len(fit_rows) < 40:
        raise SystemExit(f"D_fit has only {len(fit_rows)} labeled rows (need >=40); ensure E1b produced enough complete episodes with outcome labels")
    fold_seeds = (17, 29, 41)

    def ensemble(rows):
        by_key = {}
        for seed in fold_seeds:
            est = AIPWEstimator(n_folds=5, seed=seed)
            for item in est.aggregate(est.cross_fit(rows, fit_rows=fit_rows)):
                by_key.setdefault(item.key, []).append(item)
        combined = []
        for key, items in by_key.items():
            ref = items[0]
            ref.delta_y = sum(x.delta_y for x in items) / len(items)
            ref.risk_abs = sum(x.risk_abs for x in items) / len(items)
            ref.risk_inc = sum(x.risk_inc for x in items) / len(items)
            ref.se_y = max(x.se_y for x in items)
            ref.se_abs = max(x.se_abs for x in items)
            ref.se_inc = max(x.se_inc for x in items)
            ref.n = min(x.n for x in items); ref.n_reuse = min(x.n_reuse for x in items)
            ref.n_base = min(x.n_base for x in items); ref.ess = min(x.ess for x in items)
            ref.supported = all(x.supported for x in items)
            combined.append(ref)
        return combined

    select_est = ensemble(select)
    audit_est = ensemble(audit)
    # Select and audit each controller family independently.  They currently
    # share the same causal estimates, but separate policy objects prevent a
    # future ledger-aware selector from silently reusing the other family.
    calibrator_full = PolicyCalibrator()
    calibrator_pointwise = PolicyCalibrator()
    full_policy = calibrator_full.audit(calibrator_full.select(select_est), audit_est)
    pointwise_policy = calibrator_pointwise.audit(calibrator_pointwise.select(select_est), audit_est)
    if not full_policy.audit_passed or not pointwise_policy.audit_passed:
        raise SystemExit("D_audit failed for at least one controller family; no deployable policy was written")

    selection_source = "aipw_select"
    if args.direct_selection:
        direct = json.loads(Path(args.direct_selection).read_text(encoding="utf-8"))
        selected = direct.get("selection", {})
        direct_policies = {}
        for family in ("full", "pointwise"):
            if family not in selected or "kappa" not in selected[family]:
                raise SystemExit(f"direct selection missing {family} kappa")
            kappa = float(selected[family]["kappa"])
            # Re-materialize the admitted group set at the directly selected
            # kappa; do not retain estimates selected at another kappa.
            direct_policy = PolicyCalibrator(kappas=(kappa,)).select(select_est)
            direct_policy = PolicyCalibrator().audit(direct_policy, audit_est)
            if not direct_policy.audit_passed:
                raise SystemExit(f"D_audit failed after direct {family} kappa selection")
            direct_policies[family] = direct_policy
        full_policy, pointwise_policy = direct_policies["full"], direct_policies["pointwise"]
        selection_source = "e2_direct_matched_risk"
    policy = full_policy
    # Global-Risk Only (§18): select κ using overall constraints (no per-stratum)
    from cact.protocol_v2 import DEFAULT_KAPPAS, DEFAULT_EPS_ABS, DEFAULT_EPS_INC
    global_policy = None
    best_cov = -1.0
    for k in DEFAULT_KAPPAS:
        cov_g = 0; hrr_g = 0; eahr_g = 0; n_g = 0
        for e in full_policy.estimates.values():
            n = int(e.get("n", 0))
            if n < 20: continue
            cov_g += n
            hrr_g += float(e.get("risk_abs", 0.5)) * n
            eahr_g += float(e.get("risk_inc", 0.5)) * n
            n_g += n
        if not n_g: continue
        hrr_g /= n_g; eahr_g /= n_g
        if hrr_g <= DEFAULT_EPS_ABS and eahr_g <= DEFAULT_EPS_INC and cov_g > best_cov:
            best_cov = cov_g
            global_policy = PolicyCalibrator(kappas=(k,)).select(select_est)
            global_policy.kappa = k  # retain overall-only kappa
    if global_policy is not None:
        direct_policies["global_only"] = global_policy

    artifact = policy.to_dict()
    families = {"full": full_policy.to_dict(), "pointwise": pointwise_policy.to_dict()}
    if "global_only" in direct_policies:
        families["global_only"] = direct_policies["global_only"].to_dict()
        global_out = Path(args.out).with_stem(Path(args.out).stem + "_global_only")
        global_out.parent.mkdir(parents=True, exist_ok=True)
        global_out.write_text(json.dumps(direct_policies["global_only"].to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    artifact["families"] = families
    artifact["family_selection"] = {"full": "independent_select_audit", "pointwise": "independent_select_audit"}
    artifact.update({
        "schema_version": "cact.v2.policy",
        "fit_rows": len(fit), "select_rows": len(select), "audit_rows": len(audit),
        "fit_hash": digest(fit_paths), "select_hash": digest(select_paths),
        "audit_hash": digest(audit_paths),
        "selection_source": selection_source,
        "estimator": {"name": "episode_clustered_cross_fitted_aipw",
                      "folds": 5, "fold_seeds": [17, 29, 41]},
    })
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"out": str(out), "kappa": policy.kappa,
                      "coverage": policy.coverage, "audit_passed": policy.audit_passed},
                     indent=2))

if __name__ == "__main__":
    main()


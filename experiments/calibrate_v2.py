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
    args = ap.parse_args()

    fit, fit_paths = collect(args.fit_glob)
    select, select_paths = collect(args.select_glob)
    audit, audit_paths = collect(args.audit_glob)
    if not fit or not select or not audit:
        raise SystemExit("fit/select/audit opportunity logs must all be non-empty")
    if set(fit_paths) & set(select_paths) or set(fit_paths) & set(audit_paths) or set(select_paths) & set(audit_paths):
        raise SystemExit("D_fit, D_select and D_audit paths overlap")

    fit_rows = [r for r in fit if r.y is not None and r.harm is not None]
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
    policy = PolicyCalibrator().select(select_est)
    policy = PolicyCalibrator().audit(policy, audit_est)
    if not policy.audit_passed:
        raise SystemExit("D_audit failed; no deployable policy was written")

    artifact = policy.to_dict()
    artifact.update({
        "schema_version": "cact.v2.policy",
        "fit_rows": len(fit), "select_rows": len(select), "audit_rows": len(audit),
        "fit_hash": digest(fit_paths), "select_hash": digest(select_paths),
        "audit_hash": digest(audit_paths),
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


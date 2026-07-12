#!/usr/bin/env python3
"""Pre-experiment power analysis (§17.3): determine minimum detectable effects."""
from __future__ import annotations
import argparse, json, hashlib, math
import numpy as np

def _seed(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "big")

def simulate(n_tasks: int = 12, n_seeds: int = 8, n_episodes: int = 1728,
             hrr_baseline: float = 0.30, effect_size: float = -0.08,
             reps: int = 2000, seed: int = 42) -> dict:
    """Simulate E3 paired comparison power."""
    rng = np.random.default_rng(seed)
    per_cell = n_episodes // (n_tasks * n_seeds)
    detected = 0
    for _ in range(reps):
        delta_hrr = rng.normal(effect_size, 0.04, size=n_tasks * n_seeds).reshape(n_tasks, n_seeds)
        cell_means = delta_hrr.mean(axis=1)
        cell_ses = delta_hrr.std(axis=1, ddof=1) / math.sqrt(n_seeds)
        result = _paired_bootstrap_test(cell_means, cell_ses, n_tasks, rng)
        if result["significant"]:
            detected += 1
    power = detected / reps
    return {"power": power, "effect_size": effect_size, "n_tasks": n_tasks,
            "n_seeds": n_seeds, "n_episodes": n_episodes, "reps": reps}

def _paired_bootstrap_test(cell_means, cell_ses, n_tasks, rng):
    """Task-clustered bootstrap test for paired difference."""
    # Bootstrap task-level means
    observed = cell_means.mean()
    boot_values = []
    for _ in range(10000):
        idx = rng.integers(0, n_tasks, size=n_tasks)
        boot_values.append(cell_means[idx].mean())
    se_boot = np.std(boot_values, ddof=1)
    lower = observed - 2.0 * se_boot
    return {"observed": float(observed), "se_boot": float(se_boot),
            "ci_95_low": float(lower), "significant": bool(lower < 0) if observed < 0 else bool(lower > 0)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None); ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    results = {}
    for effect in (-0.06, -0.08, -0.10, -0.12):
        r = simulate(effect_size=effect, seed=args.seed)
        print(f"  effect={effect:+.2f}: power={r['power']:.3f} (target: >=0.80)")
        results[f"hrr_{abs(effect):.0f}pp"] = r
        if r["power"] >= 0.80:
            mde = effect
    threshold = next((e for e, d in sorted(results.items()) if d["power"] >= 0.80), None)
    print(f"Minimal detectable effect (>=80% power): {threshold}")
    if args.out:
        json.dump({"simulation_reps": args.reps, "results": {k: {kk: vv for kk, vv in v.items() if kk != "reps"}
                   for k, v in results.items()}}, open(args.out, "w"), indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()

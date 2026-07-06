"""
TrustGate: Adaptive counterfactual calibration gate (ACT-RL v2)

Per-group adaptive thresholds selected from calibration data:
  τ_g  — uplift probability threshold per task group
  δ_g  — minimum uplift effect size per group
  h_g  — harm upper bound per group

Conformal risk controller provides finite-sample safety calibration.

4-state interaction classification: synergy / neutral / conflict / unknown
"""

import numpy as np
import bisect
from typing import Dict, List, Tuple, Optional
from scipy.stats import beta as beta_dist

# Task groups for adaptive thresholds
TASK_GROUPS = ["crafting", "mining", "exploration", "tech_tree",
               "failure_recovery", "interaction_stress"]

UPLIFT_CANDIDATES = [0.80, 0.85, 0.88, 0.90, 0.92, 0.95]
DELTA_CANDIDATES = [0.02, 0.03, 0.05, 0.08, 0.10]
HARM_CANDIDATES = [0.06, 0.08, 0.10, 0.12, 0.15]
SYNERGY_THRESHOLDS = [0.70, 0.80, 0.90]
CONFLICT_THRESHOLDS = [0.70, 0.80, 0.90]


class TrustGate:
    def __init__(self):
        # Per-group thresholds (populated by calibrate)
        self.tau: Dict[str, float] = {}       # τ_g
        self.delta: Dict[str, float] = {}     # δ_g
        self.harm: Dict[str, float] = {}      # h_g
        self.theta_syn: float = 0.70           # synergy threshold
        self.theta_conf: float = 0.70          # conflict threshold
        self._calibrated = False

    # ── Single knowledge check ──
    def uplift_certificate(self, use_alpha: float, use_beta: float,
                           base_alpha: float, base_beta: float,
                           delta: float = 0.05) -> float:
        lcb_use = float(beta_dist.ppf(delta, use_alpha, use_beta))
        ucb_base = float(beta_dist.ppf(1 - delta, base_alpha, base_beta))
        return lcb_use - ucb_base

    def harm_ucb(self, harm_alpha: float, harm_beta: float,
                 delta: float = 0.05) -> float:
        return float(beta_dist.ppf(1 - delta, harm_alpha, harm_beta))

    def trust_score(self, uplift: float, harm_ucb_val: float,
                    lambda_harm: float = 0.2, t_eps: float = 0.0) -> float:
        return uplift - lambda_harm * harm_ucb_val

    # ── Calibration (per-group adaptive) ──
    def calibrate(self, calib_data: List[Dict], task_group: str = None):
        """
        Select adaptive τ_g, δ_g, h_g from calibration data.
        If task_group is None, calibrate globally.
        Returns selected config dict.
        """
        if not calib_data:
            return {"tau": 0.90, "delta": 0.05, "harm": 0.10}

        best_config = None; best_coverage = -1.0
        for tau in UPLIFT_CANDIDATES:
            for delta in DELTA_CANDIDATES:
                for harm in HARM_CANDIDATES:
                    coverage, risk, risk_plus = self._evaluate(
                        calib_data, tau, delta, harm)
                    if risk_plus <= 0.10 and coverage > best_coverage:
                        best_coverage = coverage
                        best_config = {"tau": tau, "delta": delta, "harm": harm,
                                       "coverage": coverage, "risk": risk,
                                       "n_calib": len(calib_data)}

        if best_config is None:
            best_config = {"tau": 0.95, "delta": 0.10, "harm": 0.06,
                           "coverage": 0.0, "risk": 0.0,
                           "n_calib": len(calib_data)}

        key = task_group or "_global"
        self.tau[key] = best_config["tau"]
        self.delta[key] = best_config["delta"]
        self.harm[key] = best_config["harm"]
        self._calibrated = True
        return best_config

    def calibrate_all_groups(self, data_by_group: Dict[str, List[Dict]]):
        """Calibrate per-group thresholds from grouped calibration data."""
        results = {}
        for grp in TASK_GROUPS:
            if grp in data_by_group and data_by_group[grp]:
                results[grp] = self.calibrate(data_by_group[grp], grp)
            else:
                results[grp] = {"tau": 0.90, "delta": 0.05, "harm": 0.10}
        # Calibrate interaction thresholds
        self._calibrate_interaction(data_by_group.get("calib", []))
        return results

    def _evaluate(self, data, tau, delta, harm):
        """Evaluate a candidate threshold config."""
        scores = []
        for d in data:
            pu = d.get("prob_uplift", d.get("pi_uplift", 0.5))
            hu = d.get("harm_ucb", d.get("harm", 0.05))
            up = d.get("uplift", 0.0)
            ok = pu >= tau and up >= delta and hu <= harm
            scores.append((ok, d.get("is_harmful", 0)))
        accepted = sum(1 for ok, _ in scores if ok)
        N = len(scores)
        coverage = accepted / N if N else 0
        risk = sum(h for ok, h in scores if ok) / max(accepted, 1)
        risk_plus = self._binom_ucb(int(sum(h for ok, h in scores if ok)), int(accepted))
        return coverage, risk, risk_plus

    def _binom_ucb(self, k: int, n: int, delta: float = 0.1) -> float:
        if n == 0: return 1.0
        return float(beta_dist.ppf(1 - delta, k + 1, n - k + 1))

    def _calibrate_interaction(self, calib_data: List[Dict]):
        """Select optimal synergy/conflict thresholds from calibration."""
        inter_data = [d for d in calib_data if d.get("interaction_state")]
        if len(inter_data) < 10: return
        best_f1 = -1
        for syn_t in SYNERGY_THRESHOLDS:
            for conf_t in CONFLICT_THRESHOLDS:
                tp = fp = fn = 0
                for d in inter_data:
                    pred_syn = d.get("pi_syn", 0) >= syn_t
                    pred_conf = d.get("pi_conf", 0) >= conf_t
                    true_syn = d.get("interaction_synergy", False)
                    true_conf = d.get("interaction_conflict", False)
                    if pred_syn == true_syn and pred_conf == true_conf:
                        tp += 1
                    else:
                        fp += 1
                f1 = tp / max(tp + fp/2.0, 1)
                if f1 > best_f1:
                    best_f1 = f1
                    self.theta_syn = syn_t
                    self.theta_conf = conf_t

    # ── Gate decision ──
    def should_reuse(self, prob_uplift: float, uplift_val: float,
                     harm_ucb_val: float, task_group: str = None,
                     lifecycle: str = "candidate") -> Tuple[bool, Dict]:
        """
        ACT-RL gate decision.

        Returns (reuse, info_dict)
        """
        key = task_group or "_global"
        tau = self.tau.get(key, 0.90)
        delta = self.delta.get(key, 0.05)
        harm = self.harm.get(key, 0.10)

        # Disabled knowledge is never reused
        if lifecycle in ("disabled",):
            return False, {"reason": "disabled", "tau": tau, "delta": delta, "harm": harm}

        # Gate check
        uplift_ok = prob_uplift >= tau and uplift_val >= delta
        safety_ok = harm_ucb_val <= harm

        if uplift_ok and safety_ok:
            return True, {"reason": "gate_pass", "tau": tau, "delta": delta, "harm": harm,
                          "prob_uplift": prob_uplift, "uplift_val": uplift_val,
                          "harm_ucb": harm_ucb_val}
        elif not uplift_ok:
            return False, {"reason": "uplift_fail", "tau": tau, "delta": delta, "harm": harm,
                           "prob_uplift": prob_uplift, "uplift_val": uplift_val}
        else:
            return False, {"reason": "safety_fail", "tau": tau, "delta": delta, "harm": harm,
                           "harm_ucb": harm_ucb_val}

    # ── Adaptive exploration ──
    def exploration_rate(self, ess: float, prob_uplift: float,
                         risk_level: str = "medium",
                         sample_imbalance: float = 0.0) -> float:
        """
        Adaptive active calibration base rate.

        High uncertainty + low risk → more exploration
        High risk → less exploration
        """
        risk_weights = {"low": 1.2, "medium": 1.0, "high": 0.2}
        rw = risk_weights.get(risk_level, 0.5)
        uncertainty = 4.0 * prob_uplift * (1.0 - prob_uplift) if prob_uplift is not None else 1.0
        q = 0.05 + 0.20 * uncertainty * rw + 0.10 * sample_imbalance
        return max(0.05, min(0.30, q))

    def thompson_probe_rate(self, ess: float, prob_uplift: float,
                            risk_level: str = "medium") -> float:
        """Adaptive Thompson exploration rate."""
        if risk_level == "high": return 0.0
        risk_weights = {"low": 1.5, "medium": 0.8}
        rw = risk_weights.get(risk_level, 0.3)
        uncertainty = 4.0 * prob_uplift * (1.0 - prob_uplift) if prob_uplift is not None else 1.0
        n_min = {"low": 5, "medium": 8, "high": 10}.get(risk_level, 8)
        if ess >= n_min: return 0.0
        return max(0.0, min(0.10, 0.15 * uncertainty * rw))

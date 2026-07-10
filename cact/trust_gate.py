"""
C-ACT TrustGate: Adaptive counterfactual calibration gate.

Per-group adaptive thresholds selected from calibration data:
  τ_g  — uplift probability threshold per task group
  δ_g  — minimum uplift effect size per group
  h_g  — harm upper bound per group

The 4-condition C-ACT reuse decision:
  Reuse(u,c) ⟺ π_u(c) ≥ τ*_{g,l}
             ∧  h⁺(c) ≤ h*_{g,l}
             ∧  ContractSatisfied(u,c)
             ∧  InteractionSafe(u, P, c)

Only ε_harm = 0.10 (external safety budget) is fixed.
All other thresholds are learned from E2 calibration.
"""

import json, os
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.stats import beta as beta_dist


# Lifecycle state reference (for evaluate default)
CANDIDATE = "candidate"

# Task groups for adaptive thresholds
TASK_GROUPS = ["crafting", "mining", "exploration", "tech_tree",
               "failure_recovery", "interaction_stress"]

# Calibration search grids
UPLIFT_CANDIDATES = [0.80, 0.85, 0.88, 0.90, 0.92, 0.95]
DELTA_CANDIDATES = [0.02, 0.03, 0.05, 0.08, 0.10]
HARM_CANDIDATES = [0.06, 0.08, 0.10, 0.12, 0.15]
SYNERGY_THRESHOLDS = [0.70, 0.80, 0.90]
CONFLICT_THRESHOLDS = [0.70, 0.80, 0.90]

# External safety budget (fixed)
EPS_HARM = 0.10


class TrustGate:
    """Per-group adaptive counterfactual calibration gate."""

    def __init__(self):
        # Per-group thresholds (populated by calibrate)
        self.tau: Dict[str, float] = {}       # τ_g: uplift probability threshold
        self.delta: Dict[str, float] = {}     # δ_g: minimum uplift effect size
        self.harm: Dict[str, float] = {}      # h_g: harm upper bound
        self.theta_syn: float = 0.70           # synergy threshold
        self.theta_conf: float = 0.80          # conflict threshold
        self._calibrated = False
        self._calibration_results: Dict = {}

    # ── Uplift / safety helpers ──
    @staticmethod
    def uplift_certificate(use_alpha: float, use_beta: float,
                           base_alpha: float, base_beta: float,
                           delta: float = 0.05) -> float:
        """LCB[use] - UCB[base] at significance δ."""
        lcb_use = float(beta_dist.ppf(delta, use_alpha, use_beta))
        ucb_base = float(beta_dist.ppf(1 - delta, base_alpha, base_beta))
        return lcb_use - ucb_base

    @staticmethod
    def harm_ucb(harm_alpha: float, harm_beta: float,
                 q: float = 0.95) -> float:
        """h⁺ = Q_q[p_harm]"""
        return float(beta_dist.ppf(q, harm_alpha, harm_beta))

    # ── Calibration (per-group adaptive) ──
    def calibrate(self, calib_data: List[Dict],
                  task_group: str = None) -> Dict:
        """Select adaptive τ_g, δ_g, h_g from calibration data.

        Maximizes coverage subject to risk ≤ ε_harm.
        """
        if not calib_data:
            return {"tau": 0.90, "delta": 0.05, "harm": 0.10,
                    "coverage": 0.0, "risk": 0.0}

        best_config = None
        best_coverage = -1.0

        for tau in UPLIFT_CANDIDATES:
            for delta in DELTA_CANDIDATES:
                for harm in HARM_CANDIDATES:
                    coverage, risk, risk_plus = self._evaluate(
                        calib_data, tau, delta, harm)
                    if risk_plus <= EPS_HARM and coverage > best_coverage:
                        best_coverage = coverage
                        best_config = {"tau": tau, "delta": delta, "harm": harm,
                                       "coverage": round(coverage, 3),
                                       "risk": round(risk, 3),
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

    def calibrate_all_groups(self, data_by_group: Dict[str, List[Dict]]) -> Dict:
        """Calibrate per-group thresholds from grouped calibration data."""
        results = {}
        for grp in TASK_GROUPS:
            if grp in data_by_group and data_by_group[grp]:
                results[grp] = self.calibrate(data_by_group[grp], grp)
            else:
                results[grp] = {"tau": 0.90, "delta": 0.05, "harm": 0.10,
                                "coverage": 0.0, "risk": 0.0}
        # Calibrate interaction thresholds
        self._calibrate_interaction(data_by_group.get("calib", []))
        self._calibration_results = results
        return results

    def _evaluate(self, data: List[Dict], tau: float,
                  delta: float, harm: float) -> Tuple[float, float, float]:
        """Evaluate a candidate threshold config."""
        scores = []
        for d in data:
            pu = d.get("pi_uplift", d.get("prob_uplift", 0.5))
            hu = d.get("harm_ucb", d.get("harm", 0.05))
            up = d.get("uplift", 0.0)
            ok = pu >= tau and up >= delta and hu <= harm
            if "is_harmful" in d:
                scores.append((ok, int(bool(d["is_harmful"]))))
        accepted = sum(1 for ok, _ in scores if ok)
        N = len(scores)
        coverage = accepted / N if N else 0
        risk = sum(h for ok, h in scores if ok) / max(accepted, 1)
        risk_plus = self._binom_ucb(
            int(sum(h for ok, h in scores if ok)), int(accepted))
        return coverage, risk, risk_plus

    @staticmethod
    def _binom_ucb(k: int, n: int, delta: float = 0.1) -> float:
        if n == 0: return 1.0
        return float(beta_dist.ppf(1 - delta, k + 1, n - k + 1))

    def _calibrate_interaction(self, calib_data: List[Dict]):
        """Select optimal synergy/conflict thresholds from calibration data."""
        inter_data = [d for d in calib_data if d.get("interaction_state")]
        if len(inter_data) < 10: return
        best_f1 = -1
        for syn_t in SYNERGY_THRESHOLDS:
            for conf_t in CONFLICT_THRESHOLDS:
                tp = 0; fp = 0
                for d in inter_data:
                    pred_syn = d.get("pi_syn", 0) >= syn_t
                    pred_conf = d.get("pi_conf", 0) >= conf_t
                    true_syn = d.get("interaction_synergy", False)
                    true_conf = d.get("interaction_conflict", False)
                    if pred_syn == true_syn and pred_conf == true_conf:
                        tp += 1
                    else:
                        fp += 1
                f1 = tp / max(tp + fp / 2.0, 1)
                if f1 > best_f1:
                    best_f1 = f1
                    self.theta_syn = syn_t
                    self.theta_conf = conf_t

    # ── C-ACT gate decision (4 conditions) ──
    def evaluate(self, pi_uplift: float, uplift_lcb: float,
                 harm_ucb_val: float, task_group: str = None,
                 lifecycle_state: str = CANDIDATE,
                 contract_satisfied: bool = True,
                 interaction_safe: bool = True) -> Tuple[bool, Dict]:
        """C-ACT 4-condition reuse gate.

        Reuse(u,c) ⟺ π_u ≥ τ* ∧ h⁺ ≤ h* ∧ Contract ∧ Interaction

        Args:
            pi_uplift: π_u(c) = P(p_use > p_base + δ | data)
            uplift_lcb: LCB[use] - UCB[base]
            harm_ucb_val: h⁺(c) = Q_{0.95}[p_harm]
            task_group: one of TASK_GROUPS
            lifecycle_state: current lifecycle state
            contract_satisfied: pre-condition check passed
            interaction_safe: interaction_gate result

        Returns:
            (allow, info_dict)
        """
        key = task_group or "_global"
        if getattr(self, 'abl_adaptive', True) and self.tau:
            tau = self.tau.get(key, 0.90)
            delta = self.delta.get(key, 0.05)
            harm = self.harm.get(key, 0.10)
        else:
            tau, delta, harm = 0.88, 0.05, 0.10  # Global fixed defaults (doc TABLE 76/152)

        # Condition 0: Lifecycle gate
        if lifecycle_state in ("disabled", "deprecated"):
            return False, {"reason": "lifecycle_blocked",
                           "lifecycle": lifecycle_state,
                           "tau": tau, "delta": delta, "harm": harm,
                           "supervised": False}

        # Condition 1: Contract check
        if not contract_satisfied:
            return False, {"reason": "contract_violation",
                           "tau": tau, "delta": delta, "harm": harm,
                           "supervised": False}

        # Probation thresholds are applied before the certified check.
        supervised = lifecycle_state == "probation"
        if supervised:
            tau = max(0.55, tau - 0.20)
            harm = min(harm, 0.10)

        # Condition 2: Uplift + safety check
        uplift_ok = pi_uplift >= tau and uplift_lcb >= delta
        safety_ok = harm_ucb_val <= harm

        if not uplift_ok and not safety_ok:
            return False, {"reason": "both_fail", "tau": tau,
                           "delta": delta, "harm": harm,
                           "pi_uplift": pi_uplift,
                           "uplift_lcb": uplift_lcb,
                           "harm_ucb": harm_ucb_val,
                           "supervised": supervised}
        elif not uplift_ok:
            return False, {"reason": "uplift_fail", "tau": tau,
                           "delta": delta, "harm": harm,
                           "pi_uplift": pi_uplift,
                           "uplift_lcb": uplift_lcb,
                           "supervised": supervised}
        elif not safety_ok:
            return False, {"reason": "safety_fail", "tau": tau,
                           "delta": delta, "harm": harm,
                           "harm_ucb": harm_ucb_val,
                           "supervised": supervised}

        # Condition 3: Interaction check
        if not interaction_safe:
            return False, {"reason": "interaction_conflict",
                           "tau": tau, "delta": delta, "harm": harm,
                           "pi_uplift": pi_uplift,
                           "harm_ucb": harm_ucb_val,
                           "supervised": supervised}

        return True, {"reason": "gate_pass", "tau": tau,
                      "delta": delta, "harm": harm,
                      "pi_uplift": pi_uplift,
                      "uplift_lcb": uplift_lcb,
                      "harm_ucb": harm_ucb_val,
                      "supervised": supervised}

    def should_reuse(self, pi_uplift: float, uplift_lcb: float,
                     harm_ucb_val: float, task_group: str = None,
                     lifecycle_state: str = CANDIDATE,
                     contract_satisfied: bool = True,
                     interaction_safe: bool = True) -> Tuple[bool, Dict]:
        """Alias for evaluate()."""
        return self.evaluate(pi_uplift, uplift_lcb, harm_ucb_val,
                            task_group, lifecycle_state,
                            contract_satisfied, interaction_safe)

    # ── Adaptive exploration rates (for integration) ──
    @staticmethod
    def exploration_rate(ess: float, pi_uplift: float,
                         risk_level: str = "medium",
                         sample_imbalance: float = 0.0) -> float:
        """Adaptive active calibration base rate.

        High uncertainty + low risk → more exploration.
        High risk → less exploration.
        """
        risk_weights = {"low": 1.2, "medium": 1.0, "high": 0.2}
        rw = risk_weights.get(risk_level, 0.5)
        uncertainty = 4.0 * pi_uplift * (1.0 - pi_uplift) if pi_uplift else 1.0
        q = 0.05 + 0.20 * uncertainty * rw + 0.10 * sample_imbalance
        return max(0.05, min(0.30, q))

    # ── Persistence ──
    def get_config(self) -> Dict:
        return {
            "tau": self.tau,
            "delta": self.delta,
            "harm": self.harm,
            "theta_syn": self.theta_syn,
            "theta_conf": self.theta_conf,
            "eps_harm": EPS_HARM,
        }

    def save_calibration(self, path: str):
        with open(path, "w") as f:
            json.dump(self.get_config(), f, indent=2)

    def load_calibration(self, path: str):
        if os.path.exists(path):
            with open(path) as f:
                cfg = json.load(f)
            self.tau = cfg.get("tau", {})
            self.delta = cfg.get("delta", {})
            self.harm = cfg.get("harm", {})
            self.theta_syn = cfg.get("theta_syn", 0.70)
            self.theta_conf = cfg.get("theta_conf", 0.80)
            self._calibrated = True



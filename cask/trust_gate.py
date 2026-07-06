"""
TrustGate: 反事实校准的门控逻辑（升级版）

核心升级：
  1. 统一证书：所有知识统一为 counterfactual uplift
     Δ(u,c) = LCB[p_use] - UCB[p_base]
  2. 风险校准：不手调阈值，在 calibration set 上自动选 t_ε
  3. 有害复用风险跟踪

用法:
  gate = TrustGate()
  # 校准
  gate.calibrate(calib_data, epsilon=0.1)
  # 部署
  reuse = gate.should_reuse(uplift, harm_ucb, context)
"""

import numpy as np
import bisect
from typing import List, Dict, Tuple, Optional
from scipy.stats import beta as beta_dist


class TrustGate:
    """
    反事实校准门控。

    S(u,c) = LCB[p_use] - UCB[p_base] - λ·UCB[p_harm]
    t_ε = argmax Coverage(t) s.t. Risk⁺(t) ≤ ε
    """

    def __init__(self, epsilon: float = 0.1, λ_harm: float = 0.2):
        self.epsilon = epsilon  # 目标有害复用风险
        self.λ_harm = λ_harm    # 有害复用惩罚权重
        self.t_epsilon: Optional[float] = None  # 校准后的阈值

    # ──── 单条知识判断 ────

    def uplift_certificate(self, use_alpha: float, use_beta: float,
                           base_alpha: float, base_beta: float,
                           delta: float = 0.1) -> float:
        """
        保守反事实增益：
          Δ̄ = LCB[Beta(use_α,use_β)] - UCB[Beta(base_α,base_β)]
        """
        lcb_use = float(beta_dist.ppf(delta, use_alpha, use_beta))
        ucb_base = float(beta_dist.ppf(1 - delta, base_alpha, base_beta))
        return lcb_use - ucb_base

    def harm_ucb(self, harm_alpha: float, harm_beta: float,
                 delta: float = 0.1) -> float:
        """有害复用概率上界 UCB[Beta(harm_α,harm_β)]"""
        return float(beta_dist.ppf(1 - delta, harm_alpha, harm_beta))

    def trust_score(self, uplift: float, harm_ucb_val: float) -> float:
        """信任分数 S = uplift - λ·harm_UCB"""
        return uplift - self.λ_harm * harm_ucb_val

    def should_reuse(self, uplift: float, harm_ucb_val: float) -> bool:
        """基于校准阈值的复用决策"""
        if self.t_epsilon is None:
            raise RuntimeError("Gate not calibrated. Call calibrate() first.")
        score = self.trust_score(uplift, harm_ucb_val)
        return score >= self.t_epsilon

    # ──── 校准 ────

    def calibrate(self, calib_data: List[Dict]):
        """
        从校准数据自动选择阈值。

        calib_data: [{score, uplift, harm_ucb, is_harmful}, ...]
        """
        if not calib_data:
            # No data → default threshold = 0 (allow everything)
            self.t_epsilon = 0.0
            return

        # 按 score 排序，从高到低
        sorted_data = sorted(calib_data, key=lambda d: d["score"], reverse=True)
        scores = np.array([d["score"] for d in sorted_data])
        harms = np.array([d["is_harmful"] for d in sorted_data])
        N = len(scores)

        best_t, best_coverage = -float("inf"), 0

        # 扫描所有可能的阈值（每个不同的 score 值）
        for i in range(N):
            t = scores[i]
            # 通过阈值的条目
            accepted = np.sum(scores >= t)
            if accepted == 0:
                continue
            coverage = accepted / N
            # 风险 = 有害复用 / 通过数
            risk = np.sum(harms[scores >= t]) / accepted
            # 上置信界（二项比例 UCB）
            risk_plus = self._binom_ucb(
                int(np.sum(harms[scores >= t])), int(accepted)
            )

            if risk_plus <= self.epsilon and coverage > best_coverage:
                best_coverage = coverage
                best_t = t

        # 找不到满足约束的 → 设极高的门槛（几乎不通过）
        if best_t == -float("inf"):
            best_t = max(scores) + 0.01

        self.t_epsilon = float(best_t)
        return {"t_epsilon": self.t_epsilon, "coverage": best_coverage,
                "n_calib": N, "epsilon": self.epsilon}

    @staticmethod
    def _binom_ucb(successes: int, total: int, delta: float = 0.05) -> float:
        """二项比例的 UCB"""
        if total == 0:
            return 1.0
        return float(beta_dist.ppf(1 - delta, successes + 1, total - successes + 1))

    # ──── Risk-Coverage 分析 ────

    def risk_coverage_curve(self, calib_data: List[Dict]) -> Tuple[List, List]:
        """生成 risk-coverage 曲线数据点"""
        sorted_data = sorted(calib_data, key=lambda d: d["score"], reverse=True)
        scores = np.array([d["score"] for d in sorted_data])
        harms = np.array([d["is_harmful"] for d in sorted_data])
        N = len(scores)

        coverages, risks = [], []
        unique_scores = sorted(set(scores), reverse=True)
        for t in unique_scores:
            accepted = np.sum(scores >= t)
            coverage = accepted / N
            risk = np.sum(harms[scores >= t]) / accepted if accepted > 0 else 0.0
            coverages.append(coverage)
            risks.append(risk)

        return coverages, risks

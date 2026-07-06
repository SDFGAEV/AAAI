"""
Metrics: 知识使用评估指标。

KUS  = Knowledge Usage Success    — 使用知识后推进任务的比例
HRR  = Harmful Reuse Rate         — 复用知识后失败的比例
IRR  = Invalid Repair Rate        — remedy 修复无效的比例
ECE  = Expected Calibration Error — 置信度校准误差
"""

import math
import numpy as np
from typing import Dict, List, Tuple


def compute_kus(usage_log: List[Dict]) -> float:
    """
    KUS = knowledge_advanced_task / total_knowledge_used

    分母: 所有被使用（reuse）的知识次数
    分子: 使用后任务推进（subgoal 完成 / progress 增加）
    """
    total = len(usage_log)
    if total == 0:
        return 0.0
    successful = sum(1 for entry in usage_log if entry.get("advanced_task", False))
    return successful / total


def compute_hrr(usage_log: List[Dict]) -> float:
    """
    HRR = harmful_uses / total_knowledge_used

    有害复用 = 使用了知识但 subgoal 失败 或 导致不可恢复失败
    """
    total = len(usage_log)
    if total == 0:
        return 0.0
    harmful = sum(1 for entry in usage_log
                  if not entry.get("advanced_task", False))
    return harmful / total


def compute_irr(remedy_log: List[Dict]) -> float:
    """
    IRR = remedy_failed_to_recover / remedy_used

    只针对 remedy：用了 remedy 但未能修复 failure
    """
    total = len(remedy_log)
    if total == 0:
        return 0.0
    invalid = sum(1 for entry in remedy_log
                  if not entry.get("failure_resolved", False))
    return invalid / total


def compute_ece(confidence_list: List[float],
                outcome_list: List[float],
                n_bins: int = 10) -> float:
    """
    ECE = Expected Calibration Error

    将 confidence 分桶，计算每个桶内置信度均值与真实准确率的差异。
    越低说明校准越好。
    """
    if len(confidence_list) == 0:
        return 0.0

    conf = np.array(confidence_list)
    out = np.array(outcome_list)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(conf, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    ece = 0.0
    for i in range(n_bins):
        mask = bin_indices == i
        if np.sum(mask) == 0:
            continue
        bin_conf = np.mean(conf[mask])
        bin_acc = np.mean(out[mask])
        ece += np.sum(mask) * abs(bin_conf - bin_acc)

    return ece / len(conf)


def compute_risk_coverage(usage_log: List[Dict],
                          gate_decisions: List[bool]) -> Tuple[List[float], List[float]]:
    """
    计算 risk-coverage 曲线。

    coverage = reused / (reused + fallback)
    risk = harmful_reuse / reused

    返回 (coverage_list, risk_list) 用于绘图。
    """
    total_decisions = len(gate_decisions)
    if total_decisions == 0:
        return [], []

    coverages = []
    risks = []

    for threshold in np.linspace(0, 1, 20):
        accepted = sum(1 for d in gate_decisions if d >= threshold)
        harmful = sum(1 for i, d in enumerate(gate_decisions)
                      if d >= threshold and not usage_log[i].get("advanced_task", False))

        coverage = accepted / total_decisions if total_decisions > 0 else 0
        risk = harmful / accepted if accepted > 0 else 0

        coverages.append(coverage)
        risks.append(risk)

    return coverages, risks


def calibration_diagram(confidence_list: List[float],
                        outcome_list: List[float],
                        n_bins: int = 10):
    """
    生成 reliability diagram 的数据。

    返回 (bin_confidences, bin_accuracies) 用于绘制校准曲线。
    """
    conf = np.array(confidence_list)
    out = np.array(outcome_list)

    if len(conf) == 0:
        return [], []

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(conf, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    bin_confs = []
    bin_accs = []

    for i in range(n_bins):
        mask = bin_indices == i
        if np.sum(mask) == 0:
            continue
        bin_confs.append(float(np.mean(conf[mask])))
        bin_accs.append(float(np.mean(out[mask])))

    return bin_confs, bin_accs

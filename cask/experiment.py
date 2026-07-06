"""
CASK Experiment Runner

两阶段实验：
  Phase A: 知识积累 — 运行 agent 生成 skill/remedy，记录到 TrustStore
  Phase B: 信任评估 — 对比 NoTrust / MeanTrust / LCBTrust

用法：
  from cask.experiment import CaskExperiment
  exp = CaskExperiment()
  exp.run_phase_a(tasks)
  exp.run_phase_b(tasks, method="lcb_trust")
"""

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

from .trust_store import TrustStore
from .trust_gate import TrustGate
from .context_bucket import ContextBucket
from .metrics import (compute_kus, compute_hrr, compute_irr,
                      compute_ece, calibration_diagram)
from .xenon_adapter import CaskDecomposedMemoryAdapter, CaskHrgAdapter


class CaskExperiment:
    """
    CASK 实验控制中心。

    管理 TrustStore + 两种策略（NoTrust, MeanTrust, LCBTrust）+ 日志。
    """

    def __init__(self, log_dir: str = None):
        self.log_dir = log_dir or os.path.join(
            os.path.dirname(__file__), "..", "cask_logs"
        )
        os.makedirs(self.log_dir, exist_ok=True)

        self.store = TrustStore(
            store_path=os.path.join(self.log_dir, "trust_store")
        )
        self.gate = TrustGate()
        self.bucket = ContextBucket()

        self.skill_adapter = CaskDecomposedMemoryAdapter(self.store, self.gate)
        self.hrg_adapter = CaskHrgAdapter(self.store, self.gate)

        # 实验日志
        self.usage_log: List[Dict] = []
        self.gate_decisions: List[bool] = []
        self.confidence_list: List[float] = []
        self.outcome_list: List[float] = []

        self.logger = logging.getLogger("CaskExperiment")
        self._setup_logger()

    def _setup_logger(self):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
        ))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def set_method(self, method: str):
        """
        设置门控策略。

        "no_trust":     Always reuse（对应 XENON 原始行为）
        "mean_trust":   Mean > 阈值（去掉 LCB 只剩均值）
        "lcb_trust":    LCB > 阈值（完整 CASK）
        """
        assert method in ("no_trust", "mean_trust", "lcb_trust")
        self._method = method
        self.logger.info(f"Method set to: {method}")

    def decide_skill(self, skill_id: str, context: str) -> bool:
        """
        根据当前策略决定是否复用该 skill。
        """
        if self._method == "no_trust":
            return True  # 总是复现
        elif self._method == "mean_trust":
            return self.store.mean(skill_id, context) >= 0.5
        else:  # lcb_trust
            return self.gate.check_skill(self.store, skill_id, context)

    def decide_remedy(self, remedy_id: str, context: str,
                      fallback_id: str) -> bool:
        if self._method == "no_trust":
            return True
        elif self._method == "mean_trust":
            mean_remedy = self.store.mean(remedy_id, context)
            mean_fallback = self.store.mean(fallback_id, context)
            return mean_remedy - mean_fallback >= 0.05
        else:  # lcb_trust
            return self.gate.check_remedy(
                self.store, remedy_id, context, fallback_id
            )

    def record_episode(self, task_id: str, method: str, seed: int,
                       subgoal_results: List[Dict]) -> Dict:
        """
        记录一个 episode 的完整日志。

        subgoal_results: 每个子目标的使用/结果日志
        Returns: episode 统计
        """
        episode_log = {
            "task_id": task_id,
            "method": method,
            "seed": seed,
            "timestamp": time.time(),
            "subgoals": subgoal_results,
            "metrics": {},
        }

        kus = compute_kus(subgoal_results)
        hrr = compute_hrr(subgoal_results)

        remedy_results = [
            sg for sg in subgoal_results if sg.get("type") == "remedy"
        ]
        irr = compute_irr(remedy_results)

        episode_log["metrics"]["kus"] = kus
        episode_log["metrics"]["hrr"] = hrr
        episode_log["metrics"]["irr"] = irr

        self.usage_log.extend(subgoal_results)

        # 累积校准数据
        for sg in subgoal_results:
            if "confidence" in sg:
                self.confidence_list.append(sg["confidence"])
                self.outcome_list.append(
                    1.0 if sg.get("advanced_task", False) else 0.0
                )

        return episode_log

    def save_logs(self):
        """保存实验日志"""
        log_path = os.path.join(
            self.log_dir,
            f"experiment_{self._method}_{int(time.time())}.json"
        )

        report = {
            "method": self._method,
            "total_episodes": len(self.usage_log),
            "kus": compute_kus(self.usage_log),
            "hrr": compute_hrr(self.usage_log),
            "irr": compute_irr(
                [sg for sg in self.usage_log if sg.get("type") == "remedy"]
            ),
            "ece": compute_ece(self.confidence_list, self.outcome_list),
            "calibration_data": calibration_diagram(
                self.confidence_list, self.outcome_list
            ),
        }

        with open(log_path, "w") as fp:
            json.dump(report, fp, indent=2)

        self.logger.info(f"Log saved: {log_path}")
        return report

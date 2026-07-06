"""
TrustStore: 反事实证书存储（升级版）

每个知识 (u,c) 维护三组 Beta 统计：
  use  — P(Y=1 | do(u), c)      : 使用知识后的成功率
  base — P(Y=1 | do(∅), c)      : 不使用知识时的基础成功率
  harm — P(H=1 | do(u), c)      : 有害复用概率
"""

import json, os
from typing import Dict, Tuple
from scipy.stats import beta as beta_dist


class TrustStore:

    def __init__(self, store_path: str = None):
        self.store_path = store_path or os.path.join(
            os.path.dirname(__file__), "..", "ckpt", "cask_cert"
        )
        os.makedirs(self.store_path, exist_ok=True)
        self._db_file = os.path.join(self.store_path, "cert_db.json")
        self._data: Dict[str, Dict] = {}
        self._load()

    def _make_key(self, kid: str, context: str, stat: str) -> str:
        return f"{kid}|{context}|{stat}"

    def _load(self):
        if os.path.exists(self._db_file):
            try:
                with open(self._db_file) as f:
                    self._data = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                self._data = {}
        else:
            self._data = {}

    def _save(self):
        with open(self._db_file, "w") as f:
            json.dump(self._data, f, indent=2)

    def _get(self, kid: str, context: str, stat: str) -> Dict:
        k = self._make_key(kid, context, stat)
        if k not in self._data:
            self._data[k] = {"alpha": 1.0, "beta": 1.0}
        return self._data[k]

    # ──── 记录 ────

    def record_use(self, kid: str, context: str, success: float,
                   weight: float = 1.0):
        """记录使用知识后的结果"""
        e = self._get(kid, context, "use")
        e["alpha"] += success * weight
        e["beta"] += (1 - success) * weight
        self._save()

    def record_base(self, kid: str, context: str, success: float,
                    weight: float = 1.0):
        """记录不使用知识（base/fallback）的结果"""
        e = self._get(kid, context, "base")
        e["alpha"] += success * weight
        e["beta"] += (1 - success) * weight
        self._save()

    def record_harm(self, kid: str, context: str, is_harmful: float,
                    weight: float = 1.0):
        """记录有害复用"""
        e = self._get(kid, context, "harm")
        e["alpha"] += is_harmful * weight
        e["beta"] += (1 - is_harmful) * weight
        self._save()

    def record_episode(self, kid: str, context: str,
                       used: bool, success: float, is_harmful: float,
                       weight: float = 1.0):
        """一次 episode 的完整记录"""
        if used:
            self.record_use(kid, context, success, weight)
        else:
            self.record_base(kid, context, success, weight)
        if is_harmful > 0:
            self.record_harm(kid, context, is_harmful, weight)

    # ──── 查询 ────

    def get_stats(self, kid: str, context: str, stat: str) -> Tuple[float, float]:
        e = self._get(kid, context, stat)
        return e["alpha"], e["beta"]

    def mean(self, kid: str, context: str, stat: str = "use") -> float:
        a, b = self.get_stats(kid, context, stat)
        total = a + b
        return a / total if total > 0 else 0.5

    def lcb(self, kid: str, context: str, stat: str = "use",
            delta: float = 0.1) -> float:
        a, b = self.get_stats(kid, context, stat)
        return float(beta_dist.ppf(delta, a, b))

    def ucb(self, kid: str, context: str, stat: str = "use",
            delta: float = 0.1) -> float:
        a, b = self.get_stats(kid, context, stat)
        return float(beta_dist.ppf(1 - delta, a, b))

    def total_count(self, kid: str, context: str, stat: str = "use") -> float:
        a, b = self.get_stats(kid, context, stat)
        return a + b - 2.0

    def uplift(self, kid: str, context: str, base_kid: str = None,
               delta: float = 0.1) -> float:
        """保守反事实增益 Δ̄ = LCB[use] - UCB[base]"""
        bk = base_kid or kid.replace("use:", "base:")
        return self.lcb(kid, context, "use", delta) - self.ucb(bk, context, "base", delta)

    def harm_ucb(self, kid: str, context: str, delta: float = 0.1) -> float:
        """有害复用概率上界"""
        return self.ucb(kid, context, "harm", delta)

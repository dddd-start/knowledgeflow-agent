"""主题漂移检测 —— 用句向量余弦判断用户是否切换了话题。

用途:话题切换时给对话打一个分段标记(前端展示 + 记忆分段依据)。
比关键词方案鲁棒:换说法("报销"→"出差费用")语义相近,余弦仍然高,不会误判。
"""

from __future__ import annotations

import numpy as np


class TopicTracker:
    def __init__(self, embedder, threshold: float = 0.45) -> None:
        self.embedder = embedder
        self.threshold = threshold
        self._topic_vec: np.ndarray | None = None
        self.topic_changes = 0

    def _cos(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    def check_and_update(self, text: str) -> bool:
        """返回是否检测到话题切换;同时用 EMA 平滑更新主题向量。"""
        try:
            vec = self.embedder.encode([text])[0]
        except Exception:  # noqa: BLE001
            return False
        if self._topic_vec is None:
            self._topic_vec = vec
            return False
        changed = self._cos(self._topic_vec, vec) < self.threshold
        merged = 0.7 * self._topic_vec + 0.3 * vec
        norm = float(np.linalg.norm(merged))
        self._topic_vec = merged / (norm + 1e-9) if norm else merged
        if changed:
            self.topic_changes += 1
        return changed

    def reset(self) -> None:
        self._topic_vec = None

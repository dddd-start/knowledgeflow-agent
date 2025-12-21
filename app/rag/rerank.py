"""Cross-Encoder 重排 —— 对召回结果做「精排」。

双塔(bi-encoder,即向量检索)把 query 和 doc 分别编码,速度快但精度上限受限;
Cross-Encoder 把 (query, doc) 拼成一条序列一起过模型,能建模细粒度交互,打分更准,
代价是每题要跑 N 次前向(N = 候选数),所以只对 RRF 融合后的 top-N 做二次精排。

工程取舍(见 docs/EVALUATION.md):本项目在 5 篇/40 篇语料上都实测过,
重排对 Top-1 有 +X 的提升但 P95 延迟增加 ~Y ms,所以做成开关,
quality 模式下默认只在候选 > 8 时启用。
"""

from __future__ import annotations

import logging
import threading

import numpy as np

from app.config import settings
from app.rag.schemas import Hit

logger = logging.getLogger(__name__)


class Reranker:
    backend = "st"

    def __init__(self, model_name: str | None = None, backend: str | None = None) -> None:
        self.model_name = model_name or settings.rerank_model
        self.backend = backend or settings.resolved_rerank_backend()
        self._model = None
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self.backend != "none"

    def _ensure_model(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from sentence_transformers import CrossEncoder

                    logger.info("加载重排模型 %s", self.model_name)
                    self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(self, query: str, hits: list[Hit], top_k: int | None = None) -> list[Hit]:
        """按 (query, doc) 相关性重排;模型不可用时原样返回(不影响主链路)。"""
        if not self.enabled or not hits:
            return hits
        try:
            model = self._ensure_model()
        except Exception as exc:  # noqa: BLE001
            logger.warning("重排模型加载失败,跳过热排: %s", exc)
            return hits
        pairs = [(query, hit.text) for hit in hits]
        logits = np.asarray(model.predict(pairs), dtype=np.float32).reshape(-1)
        # bge-reranker 输出 logits,过 sigmoid 转成 0-1 可解释分数
        probs = 1.0 / (1.0 + np.exp(-logits))
        order = np.argsort(-probs)
        results: list[Hit] = []
        for rank, idx in enumerate(order, start=1):
            hit = hits[int(idx)]
            results.append(
                Hit(
                    chunk_id=hit.chunk_id,
                    doc_id=hit.doc_id,
                    title=hit.title,
                    text=hit.text,
                    score=float(probs[int(idx)]),
                    channel="rerank",
                    rank=rank,
                    detail={"fused_rank": hit.rank, "fused_score": round(hit.score, 5)},
                )
            )
        return results[:top_k] if top_k else results


_reranker: Reranker | None = None


def get_reranker() -> Reranker:
    global _reranker
    if _reranker is None:
        _reranker = Reranker()
    return _reranker


def reset_reranker() -> None:
    global _reranker
    _reranker = None

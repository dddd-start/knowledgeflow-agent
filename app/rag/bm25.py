"""BM25 关键词检索 —— 中文必须配 jieba 分词。

为什么向量检索之外还要 BM25?
- 向量擅长语义相近(「出差住宿怎么报」↔「差旅住宿标准」),
  但对专有名词/编号/精确术语不敏感(「VPN-2024」「SLA 99.9%」「工号」);
- BM25 恰好相反:词面命中极准,但对同义改写无能为力。
两者互补,再用 RRF 融合,是中文企业知识库最划算的组合(见 docs/EVALUATION.md)。

实现:rank_bm25(纯 Python,小语料下足够快);索引 pickle 持久化,避免每次启动重新分词。
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

from app.config import settings
from app.rag.schemas import Chunk, Hit

logger = logging.getLogger(__name__)


class BM25Index:
    def __init__(self) -> None:
        self._bm25 = None
        self.docs: list[dict] = []
        self.tokenized: list[list[str]] = []

    # ------------------------------------------------------------------ 分词
    @staticmethod
    def tokenize(text: str) -> list[str]:
        import jieba

        # 搜索引擎模式:长词再切一次,提高「费用报销流程」这类复合词的召回
        return [w.strip() for w in jieba.lcut_for_search(text) if w.strip()]

    # ------------------------------------------------------------------ 构建
    def build(self, chunks: list[Chunk]) -> None:
        if not chunks:
            self._bm25 = None
            self.docs = []
            self.tokenized = []
            return
        from rank_bm25 import BM25Okapi

        self.docs = [
            {
                "chunk_id": c.chunk_id,
                "doc_id": c.doc_id,
                "text": c.text,
                "metadata": c.metadata,
            }
            for c in chunks
        ]
        self.tokenized = [self.tokenize(c.text) for c in chunks]
        self._bm25 = BM25Okapi(self.tokenized)

    # ------------------------------------------------------------------ 检索
    def search(self, query: str, top_k: int = 5, where: dict | None = None) -> list[Hit]:
        if self._bm25 is None or not self.docs:
            return []
        tokens = self.tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        order = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)
        # 小语料(如单测只有 2 篇)下 IDF 可能为负,此时仍按分数排序返回 Top-K,
        # 否则会"明明有命中却返回空";真实语料里过滤负分能减少噪声。
        positive = [i for i in order if float(scores[i]) > 0]
        candidates = positive or order
        hits: list[Hit] = []
        for idx in candidates:
            doc = self.docs[idx]
            if where and any(doc["metadata"].get(k) != v for k, v in where.items()):
                continue
            hits.append(
                Hit(
                    chunk_id=doc["chunk_id"],
                    doc_id=doc["doc_id"],
                    title=str(doc["metadata"].get("title", doc["doc_id"])),
                    text=doc["text"],
                    score=float(scores[idx]),
                    channel="bm25",
                    rank=len(hits) + 1,
                    detail={"category": doc["metadata"].get("category", "")},
                )
            )
            if len(hits) >= top_k:
                break
        return hits

    # ------------------------------------------------------------------ 持久化
    def save(self, path: Path | str | None = None) -> None:
        path = Path(path or settings.index_dir / "bm25.pkl")
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".pkl.tmp")
        with open(tmp, "wb") as f:
            pickle.dump({"docs": self.docs, "tokenized": self.tokenized}, f, protocol=4)
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path | str | None = None) -> BM25Index | None:
        path = Path(path or settings.index_dir / "bm25.pkl")
        if not path.exists():
            return None
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            from rank_bm25 import BM25Okapi

            index = cls()
            index.docs = data["docs"]
            index.tokenized = data["tokenized"]
            index._bm25 = BM25Okapi(index.tokenized) if index.tokenized else None
            return index
        except Exception as exc:  # noqa: BLE001
            logger.warning("BM25 索引加载失败,将重建: %s", exc)
            return None

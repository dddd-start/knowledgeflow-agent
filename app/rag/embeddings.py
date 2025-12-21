"""Embedding —— 默认本地 bge 中文小模型,无 ML 依赖时退化为哈希向量。

两个后端:
- `st`  : sentence-transformers 加载 BAAI/bge-small-zh-v1.5(512 维,CPU ~10ms/条),
          query 侧加 bge 官方推荐的中文检索指令前缀,与文档侧向量空间对齐;
- `hash`: 字符 n-gram + 分词的确定性特征哈希(512 维),零依赖、跨进程稳定,
          用于 CI / 无网演示 / 单测。它只能做"词面相似",这正是降级模式的边界,
          /api/meta 里会明确标注,不假装是语义检索。

模型懒加载:进程启动不加载权重,第一次 encode 才加载,避免拖慢接口启动。
"""

from __future__ import annotations

import hashlib
import logging
import threading
from typing import Any

import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)

# bge 中文检索模型官方建议的 query 指令前缀(文档侧不加)
_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："


class Embedder:
    """线程安全的向量化器(单进程内共享模型)。"""

    def __init__(
        self,
        backend: str | None = None,
        model_name: str | None = None,
        hash_dim: int | None = None,
    ) -> None:
        self.backend = backend or settings.resolved_embedding_backend()
        self.model_name = model_name or settings.embedding_model
        self.hash_dim = hash_dim or settings.hash_dim
        self._model: Any = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    @property
    def dim(self) -> int:
        if self.backend == "st":
            model = self._ensure_model()
            # sentence-transformers 6.x 改了方法名,兼容两种
            fn = (
                getattr(model, "get_embedding_dimension", None)
                or model.get_sentence_embedding_dimension
            )
            return int(fn())
        return self.hash_dim

    def _ensure_model(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from sentence_transformers import SentenceTransformer

                    logger.info("加载 embedding 模型 %s", self.model_name)
                    self._model = SentenceTransformer(self.model_name)
        return self._model

    # ------------------------------------------------------------------
    def encode(self, texts: list[str], *, is_query: bool = False) -> np.ndarray:
        """返回 L2 归一化后的向量矩阵 (n, dim)。"""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        if self.backend == "st":
            model = self._ensure_model()
            inputs = [f"{_QUERY_INSTRUCTION}{t}" for t in texts] if is_query else list(texts)
            vectors = model.encode(
                inputs, normalize_embeddings=True, show_progress_bar=False, batch_size=32
            )
            return np.asarray(vectors, dtype=np.float32)
        return np.vstack([self._hash_vector(t) for t in texts])

    def encode_query(self, text: str) -> np.ndarray:
        return self.encode([text], is_query=True)[0]

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return self.encode(texts, is_query=False)

    # ------------------------------------------------------------------ 哈希后端
    def _hash_vector(self, text: str) -> np.ndarray:
        import jieba

        vec = np.zeros(self.hash_dim, dtype=np.float32)
        tokens = [w.strip() for w in jieba.lcut(text) if w.strip()]
        # 中文加字 bigram,缓解分词误差(如"报销流程"vs"报销 流程")
        cjk = "".join(ch for ch in text if "\u4e00" <= ch <= "\u9fff")
        tokens.extend(cjk[i : i + 2] for i in range(len(cjk) - 1))
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "big")
            idx = value % self.hash_dim
            sign = 1.0 if (value >> 63) & 1 else -1.0
            vec[idx] += sign
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 0 else vec

    # ------------------------------------------------------------------
    def describe(self) -> dict:
        if self.backend == "st":
            return {"backend": "st", "model": self.model_name, "dim": self.dim}
        return {"backend": "hash", "model": f"hash-{self.hash_dim}", "dim": self.hash_dim}


_embedder: Embedder | None = None
_lock = threading.Lock()


def get_embedder(backend: str | None = None, model_name: str | None = None) -> Embedder:
    """进程级单例(模型只加载一次)。"""
    global _embedder
    if _embedder is None or (backend and backend != _embedder.backend):
        with _lock:
            if _embedder is None or (backend and backend != _embedder.backend):
                _embedder = Embedder(backend=backend, model_name=model_name)
    return _embedder


def reset_embedder() -> None:
    global _embedder
    _embedder = None

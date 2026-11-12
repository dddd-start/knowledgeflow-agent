"""向量库 —— 默认 numpy 精确检索,可选 ChromaDB 后端。

为什么默认不用 ChromaDB?
    本项目语料规模在 1~5 万 chunk 量级,512 维 float32 精确余弦的代价是
    向量内积一次,实测 2 万 chunk 单次检索 < 15ms,而 ANN 索引(HNSW)
    在这个规模下召回损失换来的延迟收益可以忽略。所以默认用 numpy 精确检索:
    零依赖、结果确定、便于评测复现;当语料上到百万级再切换 Chroma/Qdrant,
    `VectorStore` 接口保持不变(两个后端实现同一套方法)。

持久化:向量与 payload 分两个文件原子写入(data/index/vectors.npy + chunks.json),
启动时加载,进程重启不丢。
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path

import numpy as np

from app.config import Settings, settings
from app.rag.schemas import Chunk, Hit

logger = logging.getLogger(__name__)


class BaseVectorStore:
    backend = "base"

    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        raise NotImplementedError

    def search(self, embedding: np.ndarray, top_k: int = 5, where: dict | None = None) -> list[Hit]:
        raise NotImplementedError

    def all_chunks(self) -> list[Chunk]:
        raise NotImplementedError

    def get_document(self, doc_id: str) -> list[Chunk]:
        raise NotImplementedError

    def count(self) -> int:
        raise NotImplementedError

    def documents(self) -> list[dict]:
        """按文档聚合(标题 / 分类 / 块数 / 字数),供知识库页面展示。"""
        grouped: dict[str, dict] = {}
        for chunk in self.all_chunks():
            meta = chunk.metadata
            doc_id = chunk.doc_id
            item = grouped.setdefault(
                doc_id,
                {
                    "doc_id": doc_id,
                    "title": meta.get("title", doc_id),
                    "category": meta.get("category", ""),
                    "chunks": 0,
                    "chars": 0,
                },
            )
            item["chunks"] += 1
            item["chars"] += len(chunk.text)
        return sorted(grouped.values(), key=lambda d: (d["category"], d["title"]))

    def delete_all(self) -> None:
        raise NotImplementedError

    def persist(self) -> None:
        return None


class NumpyVectorStore(BaseVectorStore):
    backend = "numpy"

    def __init__(self, index_dir: Path | str | None = None) -> None:
        self.index_dir = Path(index_dir or settings.index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.vectors_path = self.index_dir / "vectors.npy"
        self.payload_path = self.index_dir / "chunks.json"
        self.meta_path = self.index_dir / "index_meta.json"
        self._vectors: np.ndarray = np.zeros((0, 0), dtype=np.float32)
        self._payload: list[dict] = []
        self._load()

    # ------------------------------------------------------------------ 元信息
    def save_meta(self, meta: dict) -> None:
        """记录索引构建信息(embedding 后端/模型/策略),用于启动时一致性校验。"""
        self.index_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=self.index_dir, suffix=".json", delete=False, mode="w", encoding="utf-8"
        ) as tmp:
            json.dump(meta, tmp, ensure_ascii=False, indent=2)
            os.replace(tmp.name, self.meta_path)

    def load_meta(self) -> dict:
        if not self.meta_path.exists():
            return {}
        try:
            return json.loads(self.meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    # ------------------------------------------------------------------ 持久化
    def _load(self) -> None:
        if not self.vectors_path.exists() or not self.payload_path.exists():
            return
        try:
            self._vectors = np.load(self.vectors_path).astype(np.float32)
            self._payload = json.loads(self.payload_path.read_text(encoding="utf-8"))
            logger.info(
                "向量库已加载: %d chunks, dim=%d", len(self._payload), self._vectors.shape[1]
            )
        except Exception as exc:  # noqa: BLE001  文件损坏时不应阻止启动
            logger.warning("向量库加载失败,按空库启动: %s", exc)
            self._vectors = np.zeros((0, 0), dtype=np.float32)
            self._payload = []

    def persist(self) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        if len(self._payload) == 0:
            if self.vectors_path.exists():
                os.remove(self.vectors_path)
            if self.payload_path.exists():
                os.remove(self.payload_path)
            return
        # 原子写:先写临时文件再 rename,避免进程被杀留下半个文件
        with tempfile.NamedTemporaryFile(
            dir=self.index_dir, suffix=".npy", delete=False
        ) as tmp_vec:
            np.save(tmp_vec.name, self._vectors)
            os.replace(tmp_vec.name, self.vectors_path)
        with tempfile.NamedTemporaryFile(
            dir=self.index_dir, suffix=".json", delete=False, mode="w", encoding="utf-8"
        ) as tmp_json:
            json.dump(self._payload, tmp_json, ensure_ascii=False)
            os.replace(tmp_json.name, self.payload_path)

    # ------------------------------------------------------------------ 写
    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        if not chunks:
            return
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.shape[0] != len(chunks):
            raise ValueError("chunks 与 embeddings 数量不一致")
        if self._vectors.size and self._vectors.shape[1] != embeddings.shape[1]:
            raise ValueError(
                f"向量维度不一致: 库={self._vectors.shape[1]} 新增={embeddings.shape[1]}"
                "(换了 embedding 模型需要重建索引)"
            )
        existing = {item["chunk_id"] for item in self._payload}
        keep_new = [i for i, c in enumerate(chunks) if c.chunk_id not in existing]
        if not keep_new:
            return
        new_vectors = embeddings[keep_new]
        self._vectors = (
            new_vectors if not self._vectors.size else np.vstack([self._vectors, new_vectors])
        )
        self._payload.extend(
            {
                "chunk_id": chunks[i].chunk_id,
                "doc_id": chunks[i].doc_id,
                "text": chunks[i].text,
                "metadata": chunks[i].metadata,
            }
            for i in keep_new
        )
        self.persist()

    # ------------------------------------------------------------------ 读
    def search(self, embedding: np.ndarray, top_k: int = 5, where: dict | None = None) -> list[Hit]:
        if not self._payload or top_k <= 0:
            return []
        query = np.asarray(embedding, dtype=np.float32).reshape(-1)
        scores = self._vectors @ query
        candidates = np.argsort(-scores)
        hits: list[Hit] = []
        for idx in candidates:
            item = self._payload[int(idx)]
            if where and not _match(item["metadata"], where):
                continue
            hits.append(
                Hit(
                    chunk_id=item["chunk_id"],
                    doc_id=item["doc_id"],
                    title=str(item["metadata"].get("title", item["doc_id"])),
                    text=item["text"],
                    score=float(scores[int(idx)]),
                    channel="vector",
                    rank=len(hits) + 1,
                    detail={"category": item["metadata"].get("category", "")},
                )
            )
            if len(hits) >= top_k:
                break
        return hits

    def all_chunks(self) -> list[Chunk]:
        return [
            Chunk(
                chunk_id=item["chunk_id"],
                doc_id=item["doc_id"],
                text=item["text"],
                metadata=item.get("metadata", {}),
            )
            for item in self._payload
        ]

    def get_document(self, doc_id: str) -> list[Chunk]:
        chunks = [c for c in self.all_chunks() if c.doc_id == doc_id]
        chunks.sort(key=lambda c: int(c.metadata.get("chunk_index", 0)))
        return chunks

    def count(self) -> int:
        return len(self._payload)

    def delete_all(self) -> None:
        self._vectors = np.zeros((0, 0), dtype=np.float32)
        self._payload = []
        self.persist()


class ChromaVectorStore(BaseVectorStore):
    """可选后端:数据量上到十万级后切换,接口与 numpy 版完全一致。"""

    backend = "chroma"

    def __init__(self, collection_name: str | None = None, persist_dir: Path | str | None = None):
        import chromadb

        path = Path(persist_dir or settings.index_dir) / "chroma"
        path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(path))
        self.name = collection_name or settings.collection_name
        self.collection = self.client.get_or_create_collection(
            name=self.name, metadata={"hnsw:space": "cosine"}
        )

    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        if not chunks:
            return
        self.collection.upsert(
            ids=[c.chunk_id for c in chunks],
            documents=[c.text for c in chunks],
            embeddings=[np.asarray(e, dtype=np.float32).tolist() for e in embeddings],
            metadatas=[_scalar_meta(c.metadata) for c in chunks],
        )

    def search(self, embedding: np.ndarray, top_k: int = 5, where: dict | None = None) -> list[Hit]:
        res = self.collection.query(
            query_embeddings=[np.asarray(embedding, dtype=np.float32).tolist()],
            n_results=top_k,
            where=where or None,
        )
        ids = (res.get("ids") or [[]])[0]
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        return [
            Hit(
                chunk_id=ids[i],
                doc_id=str(metas[i].get("doc_id", "")),
                title=str(metas[i].get("title", ids[i])),
                text=docs[i],
                score=1.0 - float(dists[i]),  # 余弦距离 -> 相似度
                channel="vector",
                rank=i + 1,
                detail={"category": metas[i].get("category", "")},
            )
            for i in range(len(ids))
        ]

    def all_chunks(self) -> list[Chunk]:
        res = self.collection.get(include=["documents", "metadatas"])
        ids = res.get("ids") or []
        docs = res.get("documents") or []
        metas = res.get("metadatas") or []
        return [
            Chunk(
                chunk_id=ids[i],
                doc_id=str(metas[i].get("doc_id", "")),
                text=docs[i],
                metadata=metas[i],
            )
            for i in range(len(ids))
        ]

    def get_document(self, doc_id: str) -> list[Chunk]:
        chunks = [c for c in self.all_chunks() if c.doc_id == doc_id]
        chunks.sort(key=lambda c: int(c.metadata.get("chunk_index", 0)))
        return chunks

    def count(self) -> int:
        return int(self.collection.count())

    def delete_all(self) -> None:
        self.client.delete_collection(self.name)
        self.collection = self.client.create_collection(
            name=self.name, metadata={"hnsw:space": "cosine"}
        )


def _match(metadata: dict, where: dict) -> bool:
    for key, expected in where.items():
        value = metadata.get(key)
        if isinstance(expected, dict):  # 支持 {"$in": [...]}
            allowed = expected.get("$in", [])
            if value not in allowed:
                return False
        elif value != expected:
            return False
    return True


def _scalar_meta(metadata: dict) -> dict:
    """Chroma 只接受标量元数据,统一做一次收敛。"""
    out: dict = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)):
            out[key] = value
        else:
            out[key] = json.dumps(value, ensure_ascii=False)
    return out


_store: BaseVectorStore | None = None


def create_vector_store(cfg: Settings | None = None) -> BaseVectorStore:
    cfg = cfg or settings
    backend = cfg.resolved_vector_backend()
    if backend == "chroma":
        try:
            return ChromaVectorStore(cfg.collection_name)
        except Exception as exc:  # noqa: BLE001  依赖缺失时降级
            logger.warning("Chroma 不可用,降级 numpy: %s", exc)
    return NumpyVectorStore(cfg.index_dir)


def get_vector_store(cfg: Settings | None = None) -> BaseVectorStore:
    global _store
    if _store is None:
        _store = create_vector_store(cfg)
    return _store


def reset_vector_store() -> None:
    global _store
    _store = None

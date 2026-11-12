"""检索编排 —— 整个 RAG 的"心脏"。

两种模式:
- `fast`    : query → 向量召回 + BM25 召回 → RRF 融合 → top-k
              (0 次 LLM 调用,用于意图路由的 kb 快路径与 Agent 的 kb_search 工具)
- `quality`  : query → 查询改写 → HyDE → 三路召回(原 query / HyDE / BM25)
              → RRF 融合 → Cross-Encoder 重排 → top-k
              (2 次 LLM 调用,用于"高质量问答"模式)

每次检索都会产出 `RetrievalTrace`:每个阶段的耗时、候选数量、命中片段与分数。
Web 前端把它渲染成"检索链路时间轴",评测脚本用它做消融实验 —— 同一份数据,
既给面试官看,也给评测用。
"""

from __future__ import annotations

import logging
import time

from app.config import Settings, settings
from app.llm.base import BaseLLM
from app.rag.bm25 import BM25Index
from app.rag.embeddings import Embedder, get_embedder
from app.rag.hybrid import rrf_fuse
from app.rag.rerank import Reranker, get_reranker
from app.rag.rewrite import hyde, rewrite_query
from app.rag.schemas import Chunk, Document, Hit, RetrievalTrace, Stage
from app.rag.vectorstore import BaseVectorStore, get_vector_store

logger = logging.getLogger(__name__)


class RetrievalPipeline:
    def __init__(
        self,
        llm: BaseLLM | None = None,
        *,
        embedder: Embedder | None = None,
        store: BaseVectorStore | None = None,
        bm25: BM25Index | None = None,
        reranker: Reranker | None = None,
        cfg: Settings | None = None,
    ) -> None:
        self.cfg = cfg or settings
        self.llm = llm
        self.embedder = embedder or get_embedder()
        self.store = store or get_vector_store(self.cfg)
        self.bm25 = bm25
        self.reranker = reranker if reranker is not None else get_reranker()
        self._ensure_bm25()

    # ------------------------------------------------------------------ 构建
    @classmethod
    def build(cls, llm: BaseLLM | None = None, cfg: Settings | None = None) -> RetrievalPipeline:
        cfg = cfg or settings
        return cls(
            llm,
            embedder=get_embedder(
                backend=cfg.resolved_embedding_backend(), model_name=cfg.embedding_model
            ),
            store=get_vector_store(cfg),
            reranker=Reranker(model_name=cfg.rerank_model, backend=cfg.resolved_rerank_backend()),
            cfg=cfg,
        )

    def _ensure_bm25(self) -> None:
        if self.bm25 is not None:
            return
        if self.store.count() == 0:
            return
        loaded = BM25Index.load(self.cfg.index_dir / "bm25.pkl")
        if loaded is not None:
            self.bm25 = loaded
            return
        logger.info("BM25 索引缺失,从向量库重建")
        self.rebuild_bm25()

    def rebuild_bm25(self) -> None:
        index = BM25Index()
        index.build(self.store.all_chunks())
        index.save(self.cfg.index_dir / "bm25.pkl")
        self.bm25 = index

    # ------------------------------------------------------------------ 检索
    def retrieve(
        self,
        query: str,
        *,
        mode: str = "fast",
        top_k: int | None = None,
        where: dict | None = None,
        history: list[dict] | None = None,
        enable_rerank: bool | None = None,
        llm: BaseLLM | None = None,
    ) -> tuple[list[Hit], RetrievalTrace]:
        started = time.perf_counter()
        top_k = top_k or self.cfg.top_k
        trace = RetrievalTrace(query=query, mode=mode)
        llm = llm if llm is not None else self.llm

        if self.store.count() == 0:
            trace.add(Stage(name="empty", ms=0, info={"message": "知识库为空,请先入库"}))
            trace.total_ms = int((time.perf_counter() - started) * 1000)
            return [], trace

        search_query = query
        quality = mode == "quality" and llm is not None

        # ---- 0. 查询改写 + HyDE(quality):两者互不依赖,并发执行省一轮 RTT ----
        rewritten = query
        hypothetical = ""
        if quality and (self.cfg.rewrite_enabled or self.cfg.hyde_enabled):
            import concurrent.futures

            t0 = time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                rewrite_future = (
                    pool.submit(rewrite_query, llm, query, history)
                    if self.cfg.rewrite_enabled
                    else None
                )
                hyde_future = pool.submit(hyde, llm, query) if self.cfg.hyde_enabled else None
                rewritten = rewrite_future.result() if rewrite_future else query
                hypothetical = hyde_future.result() if hyde_future else ""
            trace.add(
                Stage(
                    name="rewrite",
                    ms=int((time.perf_counter() - t0) * 1000),
                    info={"origin": query, "rewritten": rewritten},
                )
            )
            search_query = rewritten

        # ---- 1. 向量召回 ----
        t0 = time.perf_counter()
        try:
            query_vec = self.embedder.encode_query(search_query)
            vector_hits = self.store.search(query_vec, top_k=self.cfg.candidate_k, where=where)
        except Exception as exc:  # noqa: BLE001
            logger.warning("向量召回失败: %s", exc)
            vector_hits = []
        trace.add(
            Stage(
                name="vector",
                ms=int((time.perf_counter() - t0) * 1000),
                info={"candidates": len(vector_hits), "backend": self.store.backend},
                hits=vector_hits,
            )
        )

        # ---- 2. BM25 召回 ----
        t0 = time.perf_counter()
        bm25_hits = (
            self.bm25.search(search_query, top_k=self.cfg.candidate_k, where=where)
            if self.bm25
            else []
        )
        trace.add(
            Stage(
                name="bm25",
                ms=int((time.perf_counter() - t0) * 1000),
                info={"candidates": len(bm25_hits)},
                hits=bm25_hits,
            )
        )

        ranked: dict[str, list[Hit]] = {"vector": vector_hits, "bm25": bm25_hits}
        weights: dict[str, float] = {"vector": 1.0, "bm25": 1.0}

        # ---- 3. HyDE 召回(quality)----
        if quality and self.cfg.hyde_enabled:
            t0 = time.perf_counter()
            hyde_hits: list[Hit] = []
            if hypothetical:
                try:
                    hyde_vec = self.embedder.encode_documents([hypothetical])[0]
                    hyde_hits = self.store.search(hyde_vec, top_k=self.cfg.candidate_k, where=where)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("HyDE 召回失败: %s", exc)
            trace.add(
                Stage(
                    name="hyde",
                    ms=int((time.perf_counter() - t0) * 1000),
                    info={"text": hypothetical[:160], "candidates": len(hyde_hits)},
                    hits=hyde_hits,
                )
            )
            if hyde_hits:
                ranked["hyde"] = hyde_hits
                weights["hyde"] = 0.8  # 假设文档有噪声,权重略低于原 query

        # ---- 4. RRF 融合 ----
        t0 = time.perf_counter()
        fused = rrf_fuse(ranked, k=self.cfg.rrf_k, weights=weights)
        trace.add(
            Stage(
                name="fuse",
                ms=int((time.perf_counter() - t0) * 1000),
                info={"channels": list(ranked), "k": self.cfg.rrf_k, "candidates": len(fused)},
                hits=fused,
            )
        )

        # ---- 5. 重排(可选)----
        hits = fused
        do_rerank = (
            (self.cfg.rerank_enabled if enable_rerank is None else enable_rerank)
            and self.reranker.enabled
            and len(fused) > 3
        )
        if do_rerank:
            t0 = time.perf_counter()
            hits = self.reranker.rerank(query, fused[: self.cfg.rerank_top_n], top_k=top_k)
            trace.add(
                Stage(
                    name="rerank",
                    ms=int((time.perf_counter() - t0) * 1000),
                    info={
                        "model": self.reranker.model_name,
                        "candidates": min(len(fused), self.cfg.rerank_top_n),
                    },
                    hits=hits,
                )
            )
        else:
            hits = fused[:top_k]

        trace.hits = hits
        trace.total_ms = int((time.perf_counter() - started) * 1000)
        return hits, trace

    # ------------------------------------------------------------------ 上下文
    @staticmethod
    def format_context(hits: list[Hit], text_limit: int = 1200) -> str:
        """把命中片段渲染成带编号的参考资料(MockLLM 与前端都依赖这个格式)。"""
        if not hits:
            return ""
        blocks = []
        for i, hit in enumerate(hits, start=1):
            header = (
                f"[{i}] 《{hit.title}》 "
                f"(doc_id={hit.doc_id}, 相关度={hit.score:.3f}, 通道={hit.channel})"
            )
            blocks.append(f"{header}\n{hit.text[:text_limit]}")
        return "\n\n".join(blocks)

    # ------------------------------------------------------------------ 入库
    def remove_document(self, doc_id: str) -> int:
        """删除某篇文档的所有块并重建索引(增量更新 = 先删后加)。"""
        remaining = [c for c in self.store.all_chunks() if c.doc_id != doc_id]
        removed = self.store.count() - len(remaining)
        if removed <= 0:
            return 0
        self.store.delete_all()
        if remaining:
            vectors = self.embedder.encode_documents([c.text for c in remaining])
            self.store.add(remaining, vectors)
        self.rebuild_bm25()
        return removed

    def ingest(
        self,
        documents: list[Document],
        *,
        strategy: str = "recursive",
        reset: bool = False,
    ) -> dict:
        from app.rag.chunking import chunk_document

        if reset:
            self.store.delete_all()
        chunks: list[Chunk] = []
        for doc in documents:
            encode = self.embedder.encode_documents if strategy == "semantic" else None
            chunks.extend(chunk_document(doc, strategy=strategy, encode=encode))

        if chunks:
            vectors = self.embedder.encode_documents([c.text for c in chunks])
            self.store.add(chunks, vectors)
            if hasattr(self.store, "save_meta"):
                try:
                    self.store.save_meta(
                        {
                            "embedding": self.embedder.describe(),
                            "strategy": strategy,
                            "documents": len(documents),
                            "chunks": self.store.count(),
                        }
                    )
                except Exception:  # noqa: BLE001 - 元信息写失败不影响检索
                    logger.warning("写入索引元信息失败", exc_info=True)
        self.rebuild_bm25()
        logger.info("入库完成: %d 篇文档 / %d 个块", len(documents), len(chunks))
        return {
            "documents": len(documents),
            "chunks": len(chunks),
            "total_chunks": self.store.count(),
            "strategy": strategy,
        }

    # ------------------------------------------------------------------ 元信息
    def stats(self) -> dict:
        return {
            "documents": len(self.store.documents()),
            "chunks": self.store.count(),
            "embedding": self.embedder.describe(),
            "vector_store": self.store.backend,
            "bm25": bool(self.bm25 and self.bm25.docs),
            "rerank": self.reranker.enabled,
        }

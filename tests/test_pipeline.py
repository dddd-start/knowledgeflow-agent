"""检索编排测试:fast/quality 两条路径、trace 结构、上下文格式、增量更新。"""

from __future__ import annotations

import re

from app.llm.mock import MockLLM
from app.rag.pipeline import RetrievalPipeline
from app.rag.rerank import Reranker
from app.rag.schemas import Document


def test_fast_retrieve_returns_hits_and_trace(pipeline):
    hits, trace = pipeline.retrieve("报销打款是每月几号", mode="fast")
    assert hits, "应当命中费用报销流程"
    assert hits[0].doc_id.startswith("doc-")
    names = [stage.name for stage in trace.stages]
    assert names[:3] == ["vector", "bm25", "fuse"]
    assert trace.total_ms >= 0
    assert trace.hits == hits


def test_quality_retrieve_uses_rewrite_and_hyde(pipeline):
    hits, trace = pipeline.retrieve("住宿标准是多少", mode="quality")
    names = [stage.name for stage in trace.stages]
    assert "rewrite" in names and "hyde" in names
    assert hits
    # 改写阶段记录了原查询与改写结果
    rewrite_stage = next(s for s in trace.stages if s.name == "rewrite")
    assert rewrite_stage.info["origin"] == "住宿标准是多少"


def test_trace_serializes_for_frontend(pipeline):
    _hits, trace = pipeline.retrieve("出差住宿", mode="fast")
    data = trace.to_dict()
    assert data["mode"] == "fast"
    assert isinstance(data["stages"], list)
    first = data["stages"][0]
    assert {"name", "ms", "info", "hits"} <= set(first)
    hit = first["hits"][0]
    assert {"chunk_id", "doc_id", "title", "text", "score", "channel"} <= set(hit)


def test_format_context_is_parseable(pipeline):
    hits, _ = pipeline.retrieve("报销", mode="fast")
    context = pipeline.format_context(hits)
    # 与 MockLLM、前端引用解析共用的固定格式
    assert re.search(
        r"^\[1\] 《.+》 \(doc_id=doc-.+, 相关度=[\d.]+, 通道=.+\)", context, re.MULTILINE
    )
    assert context.count("[1]") == 1


def test_empty_knowledge_base(cfg, embedder):
    from app.rag.vectorstore import NumpyVectorStore

    pipe = RetrievalPipeline(
        MockLLM(),
        embedder=embedder,
        store=NumpyVectorStore(cfg.index_dir / "empty"),
        reranker=Reranker(backend="none"),
        cfg=cfg,
    )
    hits, trace = pipe.retrieve("报销")
    assert hits == []
    assert trace.stages[0].name == "empty"


def test_where_filter_by_category(pipeline):
    hits, _ = pipeline.retrieve("报销", mode="fast", where={"category": "人事制度"})
    assert all(h.detail.get("category") == "人事制度" for h in hits) or hits == []


def test_remove_document(pipeline):
    before = pipeline.store.count()
    removed = pipeline.remove_document("doc-IT服务-VPN与远程接入")
    assert removed > 0
    assert pipeline.store.count() == before - removed
    assert pipeline.remove_document("doc-不存在") == 0


def test_ingest_stats(pipeline):
    doc = Document(
        doc_id="doc-新制度-测试",
        title="测试制度",
        category="新制度",
        text="# 测试\n\n## 条款\n这是新制度内容。" * 3,
    )
    stats = pipeline.ingest([doc], strategy="fixed")
    assert stats["documents"] == 1
    assert stats["chunks"] >= 1
    assert stats["total_chunks"] == pipeline.store.count()

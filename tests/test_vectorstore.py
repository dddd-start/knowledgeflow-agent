"""向量库测试:增删查、持久化、文档聚合。"""

from __future__ import annotations

import numpy as np
import pytest

from app.rag.embeddings import Embedder
from app.rag.schemas import Chunk
from app.rag.vectorstore import NumpyVectorStore


@pytest.fixture()
def embedder() -> Embedder:
    return Embedder(backend="hash", hash_dim=32)


def add_docs(store: NumpyVectorStore, embedder: Embedder) -> None:
    chunks = [
        Chunk(
            "a#0",
            "doc-a",
            "报销流程与打款时间",
            {"title": "报销流程", "category": "财务", "chunk_index": 0},
        ),
        Chunk(
            "a#1",
            "doc-a",
            "发票要求与抬头",
            {"title": "报销流程", "category": "财务", "chunk_index": 1},
        ),
        Chunk(
            "b#0",
            "doc-b",
            "VPN 申请与使用规范",
            {"title": "VPN", "category": "IT", "chunk_index": 0},
        ),
    ]
    store.add(chunks, embedder.encode_documents([c.text for c in chunks]))


def test_add_and_search(store: NumpyVectorStore, embedder: Embedder):
    add_docs(store, embedder)
    assert store.count() == 3
    hits = store.search(embedder.encode_query("报销打款"), top_k=2)
    assert len(hits) == 2
    assert hits[0].channel == "vector"
    assert hits[0].title == "报销流程"
    assert hits[0].score >= hits[1].score


def test_get_document_sorted(store: NumpyVectorStore, embedder: Embedder):
    add_docs(store, embedder)
    chunks = store.get_document("doc-a")
    assert [c.chunk_id for c in chunks] == ["a#0", "a#1"]
    assert store.get_document("missing") == []


def test_documents_aggregation(store: NumpyVectorStore, embedder: Embedder):
    add_docs(store, embedder)
    docs = store.documents()
    assert len(docs) == 2
    doc_a = next(d for d in docs if d["doc_id"] == "doc-a")
    assert doc_a["chunks"] == 2
    assert doc_a["title"] == "报销流程"


def test_persistence_roundtrip(store: NumpyVectorStore, embedder: Embedder):
    add_docs(store, embedder)
    reloaded = NumpyVectorStore(store.index_dir)
    assert reloaded.count() == 3
    assert reloaded.search(embedder.encode_query("VPN"), top_k=1)[0].doc_id == "doc-b"


def test_delete_and_dim_mismatch(store: NumpyVectorStore, embedder: Embedder):
    add_docs(store, embedder)
    store.delete_all()
    assert store.count() == 0
    assert store.search(embedder.encode_query("报销")) == []
    # 维度不一致必须显式报错,避免把两套模型向量混在一起
    add_docs(store, embedder)
    with pytest.raises(ValueError):
        store.add([Chunk("c#0", "doc-c", "x", {})], np.zeros((1, 8), dtype=np.float32))


def test_duplicate_chunk_not_added_twice(store: NumpyVectorStore, embedder: Embedder):
    chunks = [Chunk("a#0", "doc-a", "报销", {"title": "t", "category": "c"})]
    vectors = embedder.encode_documents(["报销"])
    store.add(chunks, vectors)
    store.add(chunks, vectors)
    assert store.count() == 1

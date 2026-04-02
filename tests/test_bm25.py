"""BM25 关键词检索测试。"""

from __future__ import annotations

from app.rag.bm25 import BM25Index
from app.rag.schemas import Chunk


def make_chunk(cid: str, text: str, title: str = "t") -> Chunk:
    return Chunk(
        chunk_id=cid, doc_id="doc-" + cid, text=text, metadata={"title": title, "category": "c"}
    )


def test_chinese_tokenization_required():
    # 「费用报销」不分词会被当成一个生词,分词后「报销」才能命中
    tokens = BM25Index.tokenize("费用报销流程")
    assert "报销" in tokens or "费用" in tokens


def test_search_ranks_matching_chunk_first():
    index = BM25Index()
    index.build(
        [
            make_chunk("a", "费用报销需在 30 天内提交，报销单附发票。"),
            make_chunk("b", "VPN 申请由主管审批。"),
            make_chunk("c", "年假每年 5 天起。"),
        ]
    )
    hits = index.search("报销要几天内提交", top_k=2)
    assert hits and hits[0].chunk_id == "a"
    assert hits[0].channel == "bm25"
    assert hits[0].score > 0


def test_search_empty_index_and_query():
    index = BM25Index()
    assert index.search("报销") == []
    index.build([make_chunk("a", "报销流程")])
    assert index.search("") == []


def test_where_filter():
    index = BM25Index()
    chunks = [
        make_chunk("a", "报销流程"),
        Chunk(
            chunk_id="b",
            doc_id="doc-b",
            text="报销标准",
            metadata={"title": "t", "category": "人事"},
        ),
    ]
    index.build(chunks)
    hits = index.search("报销", where={"category": "人事"})
    assert [h.chunk_id for h in hits] == ["b"]


def test_persistence_roundtrip(tmp_path):
    index = BM25Index()
    index.build([make_chunk("a", "报销流程说明")])
    path = tmp_path / "bm25.pkl"
    index.save(path)
    loaded = BM25Index.load(path)
    assert loaded is not None
    assert loaded.search("报销")[0].chunk_id == "a"
    assert BM25Index.load(tmp_path / "missing.pkl") is None

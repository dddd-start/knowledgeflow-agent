"""RRF 融合测试。"""

from __future__ import annotations

from app.rag.hybrid import rrf_fuse
from app.rag.schemas import Hit


def make_hit(chunk_id: str, score: float = 1.0, channel: str = "vector") -> Hit:
    return Hit(
        chunk_id=chunk_id,
        doc_id="doc-" + chunk_id,
        title=chunk_id,
        text="text",
        score=score,
        channel=channel,
    )


def test_common_item_wins():
    vector = [make_hit("a"), make_hit("b"), make_hit("c")]
    bm25 = [make_hit("c"), make_hit("d")]
    fused = rrf_fuse({"vector": vector, "bm25": bm25})
    # c 同时出现在两路,融合分最高
    assert fused[0].chunk_id == "c"
    assert fused[0].detail["ranks"] == {"vector": 3, "bm25": 1}
    assert set(fused[0].detail["channels"]) == {"vector", "bm25"}


def test_scores_are_rank_based():
    fused = rrf_fuse({"vector": [make_hit(f"x{i}") for i in range(5)]}, k=60)
    # 第 1 名得分 = 1/61,第 2 名 = 1/62
    assert abs(fused[0].score - 1 / 61) < 1e-9
    assert abs(fused[1].score - 1 / 62) < 1e-9


def test_weights_shift_order():
    vector = [make_hit("a"), make_hit("b")]
    bm25 = [make_hit("b"), make_hit("a")]
    equal = rrf_fuse({"vector": vector, "bm25": bm25})
    vector_heavy = rrf_fuse({"vector": vector, "bm25": bm25}, weights={"vector": 3.0, "bm25": 0.1})
    assert equal[0].chunk_id in {"a", "b"}
    assert vector_heavy[0].chunk_id == "a"


def test_limit_and_empty():
    assert rrf_fuse({"vector": [make_hit("a"), make_hit("b")]}, limit=1)[0].chunk_id == "a"
    assert rrf_fuse({"vector": []}) == []
    assert rrf_fuse({}) == []

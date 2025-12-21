"""RRF(Reciprocal Rank Fusion)—— 多路检索结果融合。

公式:
    score(d) = Σ_r  w_r / (k + rank_r(d))

只使用「名次」而不是原始分数,因为向量余弦相似度与 BM25 分数量纲完全不同,
直接加权求和需要归一化,而不同查询的分数分布差别很大,归一化本身不可靠。
RRF 用名次天然规避了这个问题,k=60 是原论文经验值(本文档实测 k∈[10,100] 差异 <1%)。

除以 (k+rank) 而不是 1/rank,是为了压低"第一名通吃"的权重:
排第一和第二的差距不能太大,否则相关性证据会被单路结果绑架。
"""

from __future__ import annotations

from app.rag.schemas import Hit

DEFAULT_K = 60


def rrf_fuse(
    ranked_lists: dict[str, list[Hit]],
    *,
    k: int = DEFAULT_K,
    weights: dict[str, float] | None = None,
    limit: int | None = None,
) -> list[Hit]:
    """融合多路已排序结果,返回按融合分降序的 Hit 列表。

    参数:
        ranked_lists: {"vector": [...], "bm25": [...], "hyde": [...]}
        weights:      每路的权重(默认 1.0),用于突出某一路
        limit:        截断数量
    """
    weights = weights or {}
    fused: dict[str, Hit] = {}
    scores: dict[str, float] = {}
    ranks: dict[str, dict[str, int]] = {}

    for channel, hits in ranked_lists.items():
        weight = float(weights.get(channel, 1.0))
        for rank, hit in enumerate(hits, start=1):
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + weight / (k + rank)
            ranks.setdefault(hit.chunk_id, {})[channel] = rank
            if hit.chunk_id not in fused:
                fused[hit.chunk_id] = Hit(
                    chunk_id=hit.chunk_id,
                    doc_id=hit.doc_id,
                    title=hit.title,
                    text=hit.text,
                    channel="fused",
                    detail=dict(hit.detail),  # 保留 category 等元信息
                )

    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    if limit is not None:
        ordered = ordered[:limit]

    results: list[Hit] = []
    for rank, (chunk_id, score) in enumerate(ordered, start=1):
        hit = fused[chunk_id]
        hit.score = score
        hit.rank = rank
        hit.channel = "fused"
        # 保留来源 hit 的元信息(如 category),同时补上融合解释字段
        hit.detail = {
            **hit.detail,
            "rrf": round(score, 5),
            "ranks": ranks.get(chunk_id, {}),
            "channels": sorted(ranks.get(chunk_id, {})),
        }
        results.append(hit)
    return results

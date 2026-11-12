"""检索消融评测 —— 用同一份评测集,量化每一级优化到底带来了多少收益。

六组配置(逐级叠加):
    vector          纯向量(bge-small-zh)
    bm25            纯关键词(jieba + BM25)
    hybrid          向量 + BM25 + RRF
    hybrid_rewrite  再加查询改写
    quality         再加 HyDE(三路召回)
    quality_rerank  再加 Cross-Encoder 重排

指标:文档级 Top1 / Top3 / MRR(以 gold_doc 为准,命中即算,不要求命中具体 chunk)。
另外输出每类 query(keyword/paraphrase/multi_hop/noisy)的分组指标,
用来解释"语义检索补的正是关键词检索的短板"。

用法:
    python scripts/eval_retrieval.py            # 全量
    python scripts/eval_retrieval.py --chunking # 额外做三种分块策略对比
    python scripts/eval_retrieval.py --limit 20 # 快速验证
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import PROJECT_ROOT as ROOT  # noqa: E402
from app.config import settings  # noqa: E402
from app.llm.factory import get_llm  # noqa: E402
from app.obs.logging import setup_logging  # noqa: E402
from app.rag.embeddings import get_embedder  # noqa: E402
from app.rag.hybrid import rrf_fuse  # noqa: E402
from app.rag.pipeline import RetrievalPipeline  # noqa: E402
from app.rag.rerank import Reranker  # noqa: E402
from app.rag.rewrite import hyde, rewrite_query  # noqa: E402
from app.rag.schemas import Hit  # noqa: E402

TOP_K = 20


# ----------------------------------------------------------------------
def load_eval_set(limit: int | None = None) -> list[dict]:
    path = ROOT / "data" / "eval" / "eval_queries.json"
    items = json.loads(path.read_text(encoding="utf-8"))
    return items[:limit] if limit else items


def doc_ranking(hits: list[Hit]) -> list[str]:
    """把 chunk 级结果压成文档级排序(同一文档取最高排名)。"""
    seen: list[str] = []
    for hit in hits:
        if hit.doc_id not in seen:
            seen.append(hit.doc_id)
    return seen


def score_one(ranking: list[str], gold: str) -> tuple[int, float]:
    """返回 (gold 排名, 倒数排名);未命中排名为 0、倒数为 0。"""
    if gold in ranking:
        rank = ranking.index(gold) + 1
        return rank, 1.0 / rank
    return 0, 0.0


def summarize(results: list[dict]) -> dict:
    n = len(results)
    top1 = sum(1 for r in results if r["rank"] == 1) / n
    top3 = sum(1 for r in results if 1 <= r["rank"] <= 3) / n
    mrr = sum(r["rr"] for r in results) / n
    latencies = [r["ms"] for r in results]
    by_type: dict[str, dict] = {}
    for r in results:
        bucket = by_type.setdefault(r["type"], {"n": 0, "top1": 0, "top3": 0, "mrr": 0.0})
        bucket["n"] += 1
        bucket["top1"] += r["rank"] == 1
        bucket["top3"] += 1 <= r["rank"] <= 3
        bucket["mrr"] += r["rr"]
    for bucket in by_type.values():
        bucket["top1"] = round(bucket["top1"] / bucket["n"], 3)
        bucket["top3"] = round(bucket["top3"] / bucket["n"], 3)
        bucket["mrr"] = round(bucket["mrr"] / bucket["n"], 3)
    return {
        "n": n,
        "top1": round(top1, 3),
        "top3": round(top3, 3),
        "mrr": round(mrr, 3),
        "latency_p50_ms": round(statistics.median(latencies), 1),
        "latency_p95_ms": round(sorted(latencies)[int(len(latencies) * 0.95) - 1], 1),
        "by_type": by_type,
    }


# ----------------------------------------------------------------------
class Configs:
    """六种检索配置的抽取,复用同一套组件,保证变量只有一个。"""

    def __init__(self, pipeline: RetrievalPipeline, llm) -> None:
        self.p = pipeline
        # 评测脚本不走 FallbackLLM 的降级,否则网关抖动会让消融数据失真
        self.llm = getattr(llm, "primary", llm)
        self._rewrite_cache: dict[str, str] = {}
        self._hyde_cache: dict[str, str] = {}

    def rewrite(self, query: str) -> str:
        if query not in self._rewrite_cache:
            self._rewrite_cache[query] = rewrite_query(self.llm, query)
        return self._rewrite_cache[query]

    def hyde(self, query: str) -> str:
        if query not in self._hyde_cache:
            self._hyde_cache[query] = hyde(self.llm, query)
        return self._hyde_cache[query]

    def _vector(self, query: str) -> list[Hit]:
        vec = self.p.embedder.encode_query(query)
        return self.p.store.search(vec, top_k=TOP_K)

    def _bm25(self, query: str) -> list[Hit]:
        return self.p.bm25.search(query, top_k=TOP_K) if self.p.bm25 else []

    def _hybrid(self, query: str, rewritten: str | None = None) -> list[Hit]:
        q = rewritten or query
        return rrf_fuse({"vector": self._vector(q), "bm25": self._bm25(q)}, k=self.p.cfg.rrf_k)[
            :TOP_K
        ]

    # ------------------------------------------------------------------
    def run(self, name: str, query: str) -> list[Hit]:
        if name == "vector":
            return self._vector(query)
        if name == "bm25":
            return self._bm25(query)
        if name == "hybrid":
            return self._hybrid(query)
        if name == "hybrid_rewrite":
            rewritten = self.rewrite(query)
            return self._hybrid(query, rewritten)
        if name in {"quality", "quality_rerank"}:
            rewritten = self.rewrite(query)
            hypothetical = self.hyde(query)
            ranked = {"vector": self._vector(rewritten), "bm25": self._bm25(rewritten)}
            if hypothetical:
                vec = self.p.embedder.encode_documents([hypothetical])[0]
                ranked["hyde"] = self.p.store.search(vec, top_k=TOP_K)
            fused = rrf_fuse(
                ranked, k=self.p.cfg.rrf_k, weights={"vector": 1.0, "bm25": 1.0, "hyde": 0.8}
            )[:TOP_K]
            if name == "quality_rerank":
                return self.p.reranker.rerank(query, fused[:10], top_k=TOP_K)
            return fused
        raise ValueError(name)


# ----------------------------------------------------------------------
def evaluate_config(name: str, configs: Configs, items: list[dict]) -> dict:
    results = []
    for item in items:
        started = time.perf_counter()
        hits = configs.run(name, item["query"])
        ms = (time.perf_counter() - started) * 1000
        ranking = doc_ranking(hits)
        rank, rr = score_one(ranking, item["gold_doc"])
        results.append(
            {
                "query": item["query"],
                "gold": item["gold_doc"],
                "type": item.get("type", "other"),
                "rank": rank,
                "rr": rr,
                "ms": round(ms, 1),
            }
        )
    summary = summarize(results)
    summary["config"] = name
    summary["misses"] = [r["query"] for r in results if r["rank"] != 1][:8]
    return summary


def chunking_ablation(limit: int | None = None) -> list[dict]:
    """同一份语料、三种分块策略,只比较 hybrid 检索的 Top1/Top3。"""
    from app.rag.bm25 import BM25Index
    from app.rag.ingest import load_documents
    from app.rag.vectorstore import NumpyVectorStore

    docs = load_documents()
    embedder = get_embedder()
    items = load_eval_set(limit)
    rows = []
    for strategy in ("fixed", "recursive", "semantic"):
        store = NumpyVectorStore(ROOT / "data" / "index" / f"_chunk_{strategy}")
        store.delete_all()
        chunks = []
        for doc in docs:
            from app.rag.chunking import chunk_document

            encode = embedder.encode_documents if strategy == "semantic" else None
            chunks.extend(chunk_document(doc, strategy=strategy, encode=encode))
        store.add(chunks, embedder.encode_documents([c.text for c in chunks]))
        bm25 = BM25Index()
        bm25.build(store.all_chunks())

        ranks = []
        for item in items:
            vec = embedder.encode_query(item["query"])
            fused = rrf_fuse(
                {
                    "vector": store.search(vec, top_k=TOP_K),
                    "bm25": bm25.search(item["query"], top_k=TOP_K),
                },
                k=settings.rrf_k,
            )
            ranking = doc_ranking(fused)
            rank, rr = score_one(ranking, item["gold_doc"])
            ranks.append({"rank": rank, "rr": rr, "type": item.get("type", "other"), "ms": 0.0})
        summary = summarize(ranks)
        summary["config"] = f"chunking:{strategy}"
        summary["chunks"] = len(chunks)
        rows.append(summary)
        import shutil

        shutil.rmtree(store.index_dir, ignore_errors=True)
    return rows


# ----------------------------------------------------------------------
def to_markdown(rows: list[dict]) -> str:
    lines = [
        "| 配置 | Top-1 | Top-3 | MRR | P50 延迟 | P95 延迟 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['config']} | {row['top1']:.0%} | {row['top3']:.0%} | {row['mrr']:.3f} "
            f"| {row['latency_p50_ms']} ms | {row['latency_p95_ms']} ms |"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="检索消融评测")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--chunking", action="store_true", help="额外做分块策略对比")
    parser.add_argument("--all", action="store_true", help="等价于 --chunking")
    args = parser.parse_args()
    setup_logging("WARNING")

    llm = get_llm()
    pipeline = RetrievalPipeline.build(llm)
    if pipeline.store.count() == 0:
        print("知识库为空,请先运行 python scripts/ingest.py")
        return 1
    pipeline.reranker = Reranker(backend="st")  # 评测里强制测重排,不受 .env 开关影响

    items = load_eval_set(args.limit)
    print(f"评测集: {len(items)} 条 | 知识库: {pipeline.store.count()} chunks\n")
    configs = Configs(pipeline, llm)
    names = ["vector", "bm25", "hybrid", "hybrid_rewrite", "quality", "quality_rerank"]
    rows = []
    for name in names:
        started = time.perf_counter()
        row = evaluate_config(name, configs, items)
        rows.append(row)
        print(
            f"{name:>16}: Top1={row['top1']:.3f} Top3={row['top3']:.3f} "
            f"MRR={row['mrr']:.3f} P50={row['latency_p50_ms']}ms "
            f"({time.perf_counter() - started:.0f}s)"
        )
    if args.chunking or args.all:
        print("\n分块策略对比(hybrid 检索):")
        rows += chunking_ablation(args.limit)
        for row in rows[-3:]:
            print(
                f"  {row['config']:>22}: chunks={row['chunks']} Top1={row['top1']:.3f} "
                f"Top3={row['top3']:.3f} MRR={row['mrr']:.3f}"
            )

    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "eval_set_size": len(items),
        "kb_chunks": pipeline.store.count(),
        "embedding": pipeline.embedder.describe(),
        "llm_model": getattr(llm, "model", "unknown"),
        "results": rows,
    }
    out_dir = ROOT / "eval"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "report-retrieval.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n" + to_markdown(rows))
    print(f"\n报告已写入 {out_dir / 'report-retrieval.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

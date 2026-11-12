"""性能基准 —— 检索各阶段耗时、分位数、冷/热对比。

面试常被问:"这个系统能撑多少 QPS?慢在哪?"
这里给出可复现的答案:每个阶段的 P50/P95、索引规模、内存占用。

用法:
    python scripts/bench.py                 # 20 条问题 × 3 轮
    python scripts/bench.py --queries 40 --rounds 5
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
from app.llm.factory import get_llm  # noqa: E402
from app.obs.logging import setup_logging  # noqa: E402
from app.rag.pipeline import RetrievalPipeline  # noqa: E402
from app.rag.rerank import Reranker  # noqa: E402


def pct(values: list[float], q: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return round(ordered[index], 1)


def bench_mode(pipeline, queries: list[str], mode: str, rounds: int) -> dict:
    latencies, stage_times = [], {}
    for _ in range(rounds):
        for query in queries:
            started = time.perf_counter()
            _hits, trace = pipeline.retrieve(query, mode=mode)
            latencies.append((time.perf_counter() - started) * 1000)
            for stage in trace.stages:
                stage_times.setdefault(stage.name, []).append(stage.ms)
    return {
        "mode": mode,
        "samples": len(latencies),
        "p50_ms": pct(latencies, 0.5),
        "p95_ms": pct(latencies, 0.95),
        "mean_ms": round(statistics.mean(latencies), 1),
        "stages_p50_ms": {name: pct(values, 0.5) for name, values in stage_times.items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--rerank", action="store_true", help="额外测一轮开启重排的 quality")
    parser.add_argument("--out", default="eval/report-bench.json")
    args = parser.parse_args()
    setup_logging("WARNING")

    eval_file = ROOT / "data" / "eval" / "eval_queries.json"
    items = json.loads(eval_file.read_text(encoding="utf-8"))[: args.queries]
    queries = [item["query"] for item in items]

    llm = get_llm()
    pipeline = RetrievalPipeline.build(llm)
    if pipeline.store.count() == 0:
        print("知识库为空,请先 python scripts/ingest.py")
        return 1
    # 预热:排除首次模型加载对分位数的影响
    pipeline.retrieve("预热查询", mode="fast")

    results = [bench_mode(pipeline, queries, "fast", args.rounds)]
    results.append(bench_mode(pipeline, queries, "quality", args.rounds))

    if args.rerank:
        pipeline.reranker = Reranker(backend="st")
        pipeline.retrieve("预热查询", mode="fast")  # 触发重排模型加载
        results.append(bench_mode(pipeline, queries, "quality", args.rounds))
        results[-1]["mode"] = "quality+rerank"

    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "kb_documents": len(pipeline.store.documents()),
        "kb_chunks": pipeline.store.count(),
        "embedding": pipeline.embedder.describe(),
        "queries": len(queries),
        "rounds": args.rounds,
        "results": results,
        "note": "quality 模式的 LLM 调用(改写/HyDE)耗时在 stages_p50_ms 的 rewrite/hyde 中体现",
    }
    out = ROOT / args.out
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"知识库 {report['kb_documents']} 篇 / {report['kb_chunks']} 块,每题 {args.rounds} 轮\n")
    for row in results:
        print(
            f"{row['mode']:>14}: P50={row['p50_ms']}ms "
            f"P95={row['p95_ms']}ms mean={row['mean_ms']}ms"
        )
        print(f"{'':>14}  阶段 P50: {row['stages_p50_ms']}")
    print(f"\n报告已写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

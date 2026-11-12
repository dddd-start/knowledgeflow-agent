"""端到端评测 —— 忠实度(引用命中)、拒答率、延迟与成本。

与 eval_retrieval.py 的区别:
- 检索评测只看"检索有没有找到正确文档";
- 端到端评测看"最终回答有没有引用正确文档 + 不确定时会不会拒答 + 花了多少时间和钱"。
  这是面试官最关心的三件事:准、稳、便宜。

指标:
- citation_hit@1 / @3:回答的引用列表里,第 1 / 前 3 条是否包含 gold 文档;
- refusal_rate:知识库覆盖不到的问题中,明确拒答的比例;
- latency p50 / p95、平均 token、单次成本;
- 按 query 类型(keyword / paraphrase / multi_hop / noisy)分组。

用法:
    python scripts/eval_e2e.py                 # 全量(40 条 in-scope + 12 条 oos)
    python scripts/eval_e2e.py --limit 10      # 快速抽样
    python scripts/eval_e2e.py --mode fast     # 换模式(默认 quality)
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
from app.obs.logging import setup_logging  # noqa: E402
from app.services import ChatService, collect_events  # noqa: E402

# 拒答识别:不只看固定话术,模型实际会用多种"我没有依据"的表述。
# 判定优先看自检结论,其次看文本里是否同时出现"否定 + 知识库/依据/数据"信号。
REFUSAL_HINTS = (
    "没有找到足够依据",
    "知识库中没有",
    "知识库中未",
    "知识库没有",
    "未收录",
    "没有收录",
    "无法回答",
    "无法给出",
    "无法查询",
    "不做推测",
    "不能提供",
    "不适合通过问答",
    "超出",
)


def load_json(path: Path, limit: int | None = None) -> list[dict]:
    if not path.exists():
        return []
    items = json.loads(path.read_text(encoding="utf-8"))
    return items[:limit] if limit else items


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return round(ordered[index], 1)


def is_refusal(answer: str, self_check: dict | None) -> bool:
    if self_check and self_check.get("verdict") == "unsupported":
        return True
    return any(hint in answer for hint in REFUSAL_HINTS)


def main() -> int:
    parser = argparse.ArgumentParser(description="端到端评测")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--mode", default="quality", choices=["fast", "quality", "auto"])
    parser.add_argument("--strategy", default="recursive")
    args = parser.parse_args()
    setup_logging("WARNING")

    service = ChatService()
    service.ensure_index_ready()
    eval_items = load_json(ROOT / "data" / "eval" / "eval_queries.json", args.limit)
    oos_items = load_json(ROOT / "data" / "eval" / "oos_queries.json", args.limit)
    if not eval_items:
        print("缺少 data/eval/eval_queries.json")
        return 1
    meta = service.meta()
    print(
        f"端到端评测 | 模式={args.mode} | LLM={meta['llm']['model']} | "
        f"知识库={meta['kb']['documents']} 篇 / {meta['kb']['chunks']} 块\n"
    )

    rows = []
    latencies, tokens, costs = [], [], []
    for item in eval_items:
        started = time.perf_counter()
        result = collect_events(
            service.answer(item["query"], mode=args.mode, stream=False, session_id="eval")
        )
        elapsed = int((time.perf_counter() - started) * 1000)
        citations = result.get("citations", [])
        top1 = [c["doc_id"] for c in citations[:1]]
        top3 = [c["doc_id"] for c in citations[:3]]
        hit1 = item["gold_doc"] in top1
        hit3 = item["gold_doc"] in top3
        rows.append(
            {
                "query": item["query"],
                "type": item.get("type", ""),
                "gold": item["gold_doc"],
                "hit@1": hit1,
                "hit@3": hit3,
                "citations": len(citations),
                "self_check": (result.get("self_check") or {}).get("verdict"),
                "latency_ms": elapsed,
                "error": bool(result.get("error")),
            }
        )
        latencies.append(elapsed)
        usage = result.get("usage", {})
        tokens.append(usage.get("total_tokens", 0))
        costs.append(usage.get("cost_usd", 0.0))
        print(
            f"{'✓' if hit1 else ('~' if hit3 else '✗')} [{item.get('type', ''):<10}] "
            f"{item['query'][:28]:<30} {elapsed:>5}ms"
        )

    # ---- OOS 拒答 ----
    oos_rows = []
    for item in oos_items:
        result = collect_events(
            service.answer(item["query"], mode=args.mode, stream=False, session_id="eval-oos")
        )
        oos_rows.append(
            {
                "query": item["query"],
                "refused": is_refusal(result.get("answer", ""), result.get("self_check")),
                "answer_head": result.get("answer", "")[:80],
            }
        )

    by_type: dict[str, dict] = {}
    for row in rows:
        bucket = by_type.setdefault(row["type"], {"n": 0, "hit1": 0, "hit3": 0})
        bucket["n"] += 1
        bucket["hit1"] += row["hit@1"]
        bucket["hit3"] += row["hit@3"]
    for bucket in by_type.values():
        bucket["hit1"] = round(bucket["hit1"] / bucket["n"], 3)
        bucket["hit3"] = round(bucket["hit3"] / bucket["n"], 3)

    summary = {
        "mode": args.mode,
        "n": len(rows),
        "citation_hit_at_1": round(sum(r["hit@1"] for r in rows) / len(rows), 3),
        "citation_hit_at_3": round(sum(r["hit@3"] for r in rows) / len(rows), 3),
        "refusal_rate_oos": round(sum(r["refused"] for r in oos_rows) / max(len(oos_rows), 1), 3),
        "latency_p50_ms": percentile(latencies, 0.5),
        "latency_p95_ms": percentile(latencies, 0.95),
        "avg_total_tokens": round(statistics.mean(tokens), 1) if tokens else 0,
        "avg_cost_usd": round(statistics.mean(costs), 6) if costs else 0,
        "errors": sum(1 for r in rows if r["error"]),
        "by_type": by_type,
    }
    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "meta": service.meta(),
        "summary": summary,
        "details": rows,
        "oos": oos_rows,
    }
    out = ROOT / "eval"
    out.mkdir(exist_ok=True)
    (out / "report-e2e.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n===== 汇总 =====")
    for key, value in summary.items():
        if key != "by_type":
            print(f"{key:>20}: {value}")
    for name, bucket in summary["by_type"].items():
        print(
            f"  {name:<12} n={bucket['n']:<3} hit@1={bucket['hit1']:.0%} hit@3={bucket['hit3']:.0%}"
        )
    print(f"\n报告已写入 {out / 'report-e2e.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""知识库入库脚本 —— 把 data/raw 下的文档切块、向量化、建索引。

用法:
    python scripts/ingest.py                      # 递归分块 + 全量重建
    python scripts/ingest.py --strategy fixed     # 固定窗口分块(对比实验用)
    python scripts/ingest.py --strategy semantic  # 语义分块(需要 ML 依赖)
    python scripts/ingest.py --dry-run            # 只统计不写索引
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.llm.factory import get_llm  # noqa: E402
from app.obs.logging import setup_logging  # noqa: E402
from app.rag.ingest import load_documents  # noqa: E402
from app.rag.pipeline import RetrievalPipeline  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="KnowledgeFlow 知识库入库")
    parser.add_argument(
        "--strategy", default="recursive", choices=["fixed", "recursive", "semantic"]
    )
    parser.add_argument("--raw-dir", default=None, help="文档目录,默认 data/raw")
    parser.add_argument("--dry-run", action="store_true", help="只统计分块,不写入索引")
    args = parser.parse_args()

    setup_logging("INFO")
    documents = load_documents(args.raw_dir)
    if not documents:
        print(f"[错误] {args.raw_dir or 'data/raw'} 下没有可入库的文档(.md/.txt/.html)")
        return 1

    print(f"读取到 {len(documents)} 篇文档,共 {sum(d.chars for d in documents)} 字")
    if args.dry_run:
        from app.rag.chunking import chunk_document

        total = sum(len(chunk_document(doc, args.strategy)) for doc in documents)
        print(f"[dry-run] 策略={args.strategy} 预计产生 {total} 个分块")
        return 0

    pipeline = RetrievalPipeline.build(get_llm())
    started = time.perf_counter()
    stats = pipeline.ingest(documents, strategy=args.strategy, reset=True)
    elapsed = time.perf_counter() - started
    stats["elapsed_s"] = round(elapsed, 2)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"索引已写入 {pipeline.cfg.index_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

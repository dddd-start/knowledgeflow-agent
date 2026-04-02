"""镜像构建期预下载 embedding / 重排模型,保证容器启动后无需联网。

Dockerfile 里单独成文件,是为了避免在 RUN 中嵌 heredoc(BuildKit 解析会出错)。
"""

from __future__ import annotations

import os

from sentence_transformers import SentenceTransformer


def main() -> None:
    model = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
    print(f"==> 预下载 embedding 模型: {model}")
    SentenceTransformer(model)
    if os.environ.get("PRELOAD_RERANK", "0") == "1":
        from sentence_transformers import CrossEncoder

        rerank = os.environ.get("RERANK_MODEL", "BAAI/bge-reranker-base")
        print(f"==> 预下载重排模型: {rerank}")
        CrossEncoder(rerank)


if __name__ == "__main__":
    main()

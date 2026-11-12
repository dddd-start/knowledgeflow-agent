# ---------- 构建参数 ----------
#   INSTALL_ML=1 时安装本地 embedding/重排依赖(torch CPU + sentence-transformers),
#   镜像约 1.8GB;INSTALL_ML=0 时只有核心依赖(~300MB),自动使用哈希向量离线演示。
ARG INSTALL_ML=1

FROM python:3.12-slim AS base
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_TELEMETRY=1
WORKDIR /app

# uv:快速、可复现的依赖安装
COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /usr/local/bin/uv

FROM base AS deps
ARG INSTALL_ML=1
COPY pyproject.toml uv.lock README.md ./
RUN if [ "$INSTALL_ML" = "1" ]; then \
      uv sync --frozen --no-dev --extra ml; \
    else \
      uv sync --frozen --no-dev; \
    fi

FROM base AS runtime
ARG INSTALL_ML=1
ENV PATH="/app/.venv/bin:$PATH" \
    HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com} \
    LLM_BACKEND=auto \
    VECTOR_BACKEND=numpy \
    HOST=0.0.0.0 \
    PORT=8000
COPY --from=deps /app/.venv /app/.venv
COPY app ./app
COPY web ./web
COPY scripts ./scripts
COPY data/raw ./data/raw
COPY data/intent_model ./data/intent_model
COPY Makefile run.sh start_server.sh ./

# 预下载 embedding / 重排模型(镜像内离线可用),避免首次请求等待
RUN if [ "$INSTALL_ML" = "1" ]; then \
      python - <<'PY'
import os
from sentence_transformers import SentenceTransformer
SentenceTransformer("BAAI/bge-small-zh-v1.5")
if os.getenv("PRELOAD_RERANK", "0") == "1":
    from sentence_transformers import CrossEncoder
    CrossEncoder("BAAI/bge-reranker-base")
PY
    fi

# 首次启动自动建索引;有 data/index 卷则跳过
RUN mkdir -p data/index logs && \
    if [ "$INSTALL_ML" = "1" ]; then python scripts/ingest.py; fi || true

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
  CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/api/health')" || exit 1
CMD ["python", "-m", "uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

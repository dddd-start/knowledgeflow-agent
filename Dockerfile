# syntax=docker/dockerfile:1
# ---------- 构建参数 ----------
#   INSTALL_ML=1  安装本地 embedding/重排依赖(torch CPU + sentence-transformers),镜像约 1.8GB
#   INSTALL_ML=0  仅核心依赖(~300MB),自动使用哈希向量离线演示(CI 用这个)
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
    HF_ENDPOINT=https://hf-mirror.com \
    HOST=0.0.0.0 \
    PORT=8000 \
    LOG_LEVEL=INFO
COPY --from=deps /app/.venv /app/.venv
COPY app ./app
COPY web ./web
COPY scripts ./scripts
COPY data/raw ./data/raw
COPY data/intent_model ./data/intent_model
COPY deploy/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
COPY Makefile run.sh start_server.sh ./
RUN chmod +x /usr/local/bin/docker-entrypoint.sh && mkdir -p data/index logs

# 预下载模型(构建期完成,运行时不再联网);重排模型体积大,默认不预下载
RUN if [ "$INSTALL_ML" = "1" ]; then python scripts/preload_models.py; fi

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/api/health')" || exit 1

# 首次启动自动建索引(用当前 embedding 后端,避免维度/后端不一致)
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["python", "-m", "uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

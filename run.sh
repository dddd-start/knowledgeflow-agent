#!/usr/bin/env bash
# KnowledgeFlow 一键启动:装依赖 -> 建索引 -> 起服务
# 没有 API Key 也能跑(离线 Mock 演示模式);有 Key 则自动走真实模型。
set -euo pipefail
cd "$(dirname "$0")"

UV="${UV:-uv}"
if ! command -v "$UV" >/dev/null 2>&1; then
  if [ -x "$HOME/.local/bin/uv" ]; then
    UV="$HOME/.local/bin/uv"
  else
    echo "未找到 uv。安装:curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    exit 1
  fi
fi

if [ ! -d .venv ]; then
  echo "==> 首次运行:安装依赖(约 3~5 分钟,含 CPU 版 torch)"
  "$UV" sync --python 3.12 --extra ml --group dev
fi

if [ ! -f .env ]; then
  cp .env.example .env
  echo "==> 已生成 .env:未配置 LLM_API_KEY,将以离线 Mock 模式启动"
fi

if [ ! -f data/index/vectors.npy ]; then
  echo "==> 首次运行:构建知识库索引"
  .venv/bin/python scripts/ingest.py
fi

PORT="${PORT:-8000}"
echo "==> 启动: http://127.0.0.1:${PORT}"
exec .venv/bin/python -m uvicorn app.api.main:app --host 0.0.0.0 --port "$PORT"

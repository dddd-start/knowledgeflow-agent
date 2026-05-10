#!/bin/sh
# 容器启动入口:索引缺失时先建索引(用当前 embedding 后端,保证与运行期一致),再启动服务。
set -e

if [ ! -f /app/data/index/vectors.npy ]; then
  echo "==> 首次启动:构建知识库索引"
  python scripts/ingest.py || echo "!! 索引构建失败,服务将以空知识库启动(可稍后手动重建)"
fi

exec "$@"

# KnowledgeFlow —— 常用命令
SHELL := /bin/bash
PY := .venv/bin/python
UV := uv
PORT ?= 8000

.PHONY: setup seed run dev cli test lint fmt eval bench train-intent demo clean help

help:
	@echo "make setup        安装依赖(uv sync --extra ml)"
	@echo "make seed         构建知识库索引(data/raw -> data/index)"
	@echo "make run          启动 Web 服务( http://127.0.0.1:8000 )"
	@echo "make dev          开发模式(热重载)"
	@echo "make cli          命令行交互"
	@echo "make test         离线测试(不依赖 API Key)"
	@echo "make lint         代码检查(ruff + mypy)"
	@echo "make eval         检索消融评测(生成 docs/EVALUATION.md 数据)"
	@echo "make bench        延迟基准测试"
	@echo "make train-intent 微调意图分类模型"

setup:
	$(UV) sync --python 3.12 --extra ml --group dev
	@test -f .env || cp .env.example .env
	@echo "依赖已安装。没有 API Key 也能跑(离线 Mock 模式)。"

seed:
	$(PY) scripts/ingest.py

run:
	$(PY) -m uvicorn app.api.main:app --host 0.0.0.0 --port $(PORT)

dev:
	$(PY) -m uvicorn app.api.main:app --host 127.0.0.1 --port $(PORT) --reload

cli:
	$(PY) -m app.cli.main

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check app scripts tests
	$(PY) -m mypy app

fmt:
	$(PY) -m ruff format app scripts tests
	$(PY) -m ruff check --fix app scripts tests

eval:
	$(PY) scripts/eval_retrieval.py --all
	$(PY) scripts/eval_e2e.py

bench:
	$(PY) scripts/bench.py

train-intent:
	$(PY) scripts/train_intent.py

demo: seed run

clean:
	rm -rf data/index data/chroma data/memory .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

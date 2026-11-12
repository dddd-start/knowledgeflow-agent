"""HTTP API —— 把服务层封装成 SSE 流式接口 + 知识库管理接口。

设计要点:
- `POST /api/chat`:默认 SSE 流式(边检索边生成边推事件),`stream=false` 时返回完整 JSON;
- 重对象(模型 / 索引 / 服务)在 lifespan 里初始化一次,不在 import 时加载;
- 静态页面挂 `web/`,与 API 同源部署,面试演示只需一个端口;
- `/metrics` 暴露 Prometheus 文本指标,便于讲"可观测性"。
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from app import __version__
from app.config import settings
from app.obs.logging import setup_logging
from app.services import ChatService

logger = logging.getLogger(__name__)


class AppState:
    """进程级单例:一次加载,所有请求复用。"""

    def __init__(self, service: ChatService | None = None) -> None:
        self.started_at = time.time()
        self.service = service or ChatService()


def create_app(service_factory=None) -> FastAPI:
    """应用工厂:默认用全局配置构建服务;测试可注入自定义 ChatService。"""

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        setup_logging()
        logger.info("启动 %s v%s", settings.app_name, __version__)
        state = AppState(service_factory() if service_factory else None)
        application.state.kf = state
        state.service.ensure_index_ready()
        if settings.warmup:
            state.service.warmup()
        meta = state.service.meta()
        logger.info(
            "就绪: LLM=%s | embedding=%s | 向量库=%s | 知识库=%d chunks",
            meta["llm"],
            meta["embedding"],
            meta["vector_store"],
            state.service.pipeline.store.count(),
        )
        yield
        state.service.registry.close()
        logger.info("已停止")

    application = FastAPI(
        title="KnowledgeFlow · 企业知识库智能问答 Agent",
        version=__version__,
        description="混合检索 + 意图路由 + Agent 决策循环 + 幻觉自检的可评测 RAG 服务",
        lifespan=lifespan,
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    from app.api.routers import chat, kb, meta

    application.include_router(chat.router)
    application.include_router(kb.router)
    application.include_router(meta.router)

    web_dir = Path(settings.web_dir)
    web_dir.mkdir(parents=True, exist_ok=True)
    application.mount("/static", StaticFiles(directory=str(web_dir)), name="static")

    @application.get("/", include_in_schema=False)
    def index():
        index_file = web_dir / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
        return HTMLResponse(
            "<h1>KnowledgeFlow API</h1><p>演示前端尚未安装,接口文档见 <a href='/docs'>/docs</a></p>"
        )

    return application


app = create_app()

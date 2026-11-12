"""系统信息接口 —— 前后端联调、演示讲解、监控都从这里取"当前到底跑的是什么"。"""

from __future__ import annotations

import json
import time

from fastapi import APIRouter, Request
from fastapi.responses import PlainTextResponse

from app import __version__
from app.obs.metrics import metrics
from app.prompts.manager import get_prompt_manager

router = APIRouter(tags=["meta"])


@router.get("/api/meta")
async def meta(request: Request):
    return request.app.state.kf.service.meta()


@router.get("/api/health")
async def health(request: Request):
    state = request.app.state.kf
    return {
        "status": "ok",
        "version": __version__,
        "uptime_s": round(time.time() - state.started_at, 1),
        "kb_chunks": state.service.pipeline.store.count(),
        "llm_backend": state.service.meta()["llm"]["backend"],
    }


@router.get("/metrics", response_class=PlainTextResponse)
async def prometheus(request: Request):
    service = request.app.state.kf.service
    text = metrics.render_prometheus()
    stats = service.pipeline.stats()
    text += (
        "# HELP kf_kb_chunks 知识库文本块数量\n"
        "# TYPE kf_kb_chunks gauge\n"
        f"kf_kb_chunks {stats['chunks']}\n"
        "# HELP kf_kb_documents 知识库文档数量\n"
        "# TYPE kf_kb_documents gauge\n"
        f"kf_kb_documents {stats['documents']}\n"
    )
    return PlainTextResponse(text)


@router.get("/api/demo/questions")
async def demo_questions(request: Request):
    """演示问题从评测集里取,保证"点了就能答上来"(不会现场翻车)。"""
    eval_file = request.app.state.kf.service.cfg.data_dir / "eval" / "eval_queries.json"
    if not eval_file.exists():
        return {"questions": []}
    try:
        items = json.loads(eval_file.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {"questions": []}
    keywords = [q for q in items if q.get("type") == "keyword"][:3]
    paraphrases = [q for q in items if q.get("type") == "paraphrase"][:4]
    multi = [q for q in items if q.get("type") == "multi_hop"][:2]
    picked = keywords + paraphrases + multi
    return {
        "questions": [
            {"text": q["query"], "type": q.get("type", ""), "note": q.get("note", "")}
            for q in picked
        ]
    }


@router.get("/api/prompts")
async def prompts():
    manager = get_prompt_manager()
    names = ["answer", "rewrite", "hyde", "self_check", "agent_system", "chat"]
    return {
        "versions": {name: manager.get_version(name) for name in names},
        "templates_dir": str(manager.templates_dir),
    }

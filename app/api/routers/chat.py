"""对话接口 —— SSE 流式 + JSON 一次性返回。

SSE 事件协议(前端按 `event` 字段分发):
    meta       本次请求的模式/意图/模型/知识库规模
    retrieval  检索链路:各阶段耗时、每路候选、融合与重排结果
    delta      回答文本增量
    replace    自检修订:前端用新文本替换已渲染内容
    citations  引用列表(doc_id / 标题 / 分数 / 原文片段)
    tools      Agent 工具调用记录(输入/输出/耗时/是否失败)
    self_check 幻觉自检结论
    usage      token / 成本 / 调用次数 / 总耗时
    done       最终答案 + 完整 trace
    error      错误信息
"""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.api.schemas import ChatRequest
from app.services import Event, collect_events

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["chat"])

_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",  # 关掉 nginx 缓冲,保证逐字输出
}


def _encode(event: Event) -> str:
    return f"event: {event.type}\ndata: {json.dumps(event.data, ensure_ascii=False)}\n\n"


@router.post("/chat")
async def chat(req: ChatRequest, request: Request):
    service = request.app.state.kf.service
    if not req.stream:
        events = service.answer(
            req.message,
            session_id=req.session_id,
            user_id=req.user_id,
            mode=req.mode,
            stream=False,
        )
        result = collect_events(events)
        if result.get("error"):
            raise HTTPException(status_code=500, detail=result["error"])
        return result

    def gen():
        try:
            for event in service.answer(
                req.message,
                session_id=req.session_id,
                user_id=req.user_id,
                mode=req.mode,
                stream=True,
            ):
                yield _encode(event)
        except Exception as exc:  # noqa: BLE001  流已开始,只能以事件形式报错
            logger.exception("SSE 生成失败")
            yield _encode(Event("error", {"message": f"{type(exc).__name__}: {exc}"}))

    return StreamingResponse(gen(), media_type="text/event-stream", headers=_SSE_HEADERS)


@router.get("/memory/{user_id}/{session_id}")
async def memory(user_id: str, session_id: str, request: Request):
    service = request.app.state.kf.service
    return service.memory(session_id, user_id).status()


@router.delete("/memory/{user_id}")
async def clear_memory(user_id: str, request: Request):
    service = request.app.state.kf.service
    service.memory("default", user_id).long.delete_user(user_id)
    return {"deleted": user_id}

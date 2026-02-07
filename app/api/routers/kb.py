"""知识库接口 —— 文档列表 / 详情 / 入库 / 上传 / 重建索引。

真实企业里入库是异步任务(大文件解析慢),本项目为演示做成同步 + 小语料秒级完成,
但保留了"单篇增量入库"与"全量重建"两个入口,接口形态与生产一致。
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from starlette.concurrency import run_in_threadpool

from app.api.schemas import IngestRequest, IngestResponse
from app.rag.ingest import build_document, load_documents

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/kb", tags=["knowledge-base"])


@router.get("/documents")
async def list_documents(request: Request):
    store = request.app.state.kf.service.pipeline.store
    documents = store.documents()
    categories: dict[str, int] = {}
    for doc in documents:
        categories[doc["category"]] = categories.get(doc["category"], 0) + 1
    return {
        "total": len(documents),
        "chunks": store.count(),
        "categories": categories,
        "documents": documents,
    }


@router.get("/documents/{doc_id}")
async def document_detail(doc_id: str, request: Request):
    store = request.app.state.kf.service.pipeline.store
    chunks = store.get_document(doc_id)
    if not chunks:
        raise HTTPException(status_code=404, detail=f"文档不存在: {doc_id}")
    meta = chunks[0].metadata
    return {
        "doc_id": doc_id,
        "title": meta.get("title", doc_id),
        "category": meta.get("category", ""),
        "chunks": [
            {"chunk_id": c.chunk_id, "index": c.metadata.get("chunk_index", i), "text": c.text}
            for i, c in enumerate(chunks)
        ],
    }


@router.post("/ingest", response_model=IngestResponse)
async def ingest(req: IngestRequest, request: Request):
    service = request.app.state.kf.service
    pipeline = service.pipeline

    def _do() -> dict:
        if req.rebuild:
            docs = load_documents()
            return pipeline.ingest(docs, strategy=req.strategy, reset=True)
        doc = build_document(req.title, req.category, req.text)
        pipeline.remove_document(doc.doc_id)  # 同 ID 覆盖,避免残留旧块
        return pipeline.ingest([doc], strategy=req.strategy, reset=False)

    try:
        stats = await run_in_threadpool(_do)
    except Exception as exc:  # noqa: BLE001
        logger.exception("入库失败")
        raise HTTPException(status_code=500, detail=f"入库失败: {exc}") from exc
    return IngestResponse(**stats)


@router.post("/upload", response_model=IngestResponse)
async def upload(
    request: Request,
    file: UploadFile = File(...),
    title: str = Form(""),
    category: str = Form("未分类"),
    strategy: str = Form("recursive"),
):
    raw = await file.read()
    if len(raw) > 5 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="文件超过 5MB,请拆分后上传")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=400, detail="仅支持 UTF-8 编码的 .md / .txt 文件") from exc
    if len(text.strip()) < 20:
        raise HTTPException(status_code=400, detail="文档内容过短(少于 20 字)")

    service = request.app.state.kf.service
    doc_title = title or (file.filename or "未命名文档").rsplit(".", 1)[0]

    def _do() -> dict:
        doc = build_document(doc_title, category, text, source=file.filename or "")
        service.pipeline.remove_document(doc.doc_id)
        return service.pipeline.ingest([doc], strategy=strategy, reset=False)

    try:
        stats = await run_in_threadpool(_do)
    except Exception as exc:  # noqa: BLE001
        logger.exception("上传入库失败")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return IngestResponse(**stats)


@router.post("/reindex", response_model=IngestResponse)
async def reindex(request: Request, strategy: str = "recursive"):
    service = request.app.state.kf.service

    def _do() -> dict:
        docs = load_documents()
        if not docs:
            raise RuntimeError("data/raw 下没有可入库的文档")
        return service.pipeline.ingest(docs, strategy=strategy, reset=True)

    try:
        stats = await run_in_threadpool(_do)
    except Exception as exc:  # noqa: BLE001
        logger.exception("重建索引失败")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return IngestResponse(**stats)

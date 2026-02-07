"""API 请求/响应模型。"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000, description="用户问题")
    session_id: str = Field("default", max_length=64, description="会话 ID(短期记忆隔离)")
    user_id: str = Field("default", max_length=64, description="用户 ID(长期偏好隔离)")
    mode: str = Field("auto", description="auto | fast | quality | agent | agentic")
    stream: bool = Field(True, description="是否 SSE 流式返回")


class IngestRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=100)
    category: str = Field("未分类", max_length=30)
    text: str = Field(..., min_length=20, description="文档正文(支持 Markdown)")
    strategy: str = Field("recursive", description="fixed | recursive | semantic")
    rebuild: bool = Field(False, description="是否连同 data/raw 全量重建索引")


class IngestResponse(BaseModel):
    documents: int
    chunks: int
    total_chunks: int
    strategy: str


class MemoryResponse(BaseModel):
    user_id: str
    session_id: str
    turns: int
    summary: str
    estimated_tokens: int
    preferences: list[str]

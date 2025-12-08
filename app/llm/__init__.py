"""LLM 适配层:协议 + OpenAI 兼容实现 + 离线 Mock + 工厂。"""

from app.llm.base import (
    BaseLLM,
    ChatResult,
    LLMError,
    StreamEvent,
    ToolCall,
    Usage,
    estimate_tokens,
)
from app.llm.factory import FallbackLLM, build_llm, get_llm, reset_llm_cache
from app.llm.mock import MockLLM
from app.llm.openai_compat import OpenAICompatLLM

__all__ = [
    "BaseLLM",
    "ChatResult",
    "FallbackLLM",
    "LLMError",
    "MockLLM",
    "OpenAICompatLLM",
    "StreamEvent",
    "ToolCall",
    "Usage",
    "build_llm",
    "estimate_tokens",
    "get_llm",
    "reset_llm_cache",
]

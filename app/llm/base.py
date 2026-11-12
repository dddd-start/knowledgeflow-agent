"""LLM 抽象层 —— 全项目唯一的模型调用协议。

这里刻意**不绑定任何 SDK**:只定义协议 + 数据结构,具体实现有两个:
- `openai_compat.OpenAICompatLLM`:任何 OpenAI 兼容网关(DeepSeek 官方 / one-api /
  Ollama / vLLM)都能直接接,换模型只改 .env;
- `mock.MockLLM`:确定性离线模型,保证没有 Key、没有网络时整套系统仍可演示与单测。

关键设计:
1. `purpose` 参数标明这次调用在业务里的用途(answer / rewrite / hyde / judge / ...),
   它同时服务于可观测性(统计每次问答的 LLM 调用构成)和离线 mock 的确定性行为;
2. 统一返回 `ChatResult`,带 token 用量、延迟、finish_reason,便于成本与性能核算;
3. 流式返回 `StreamEvent`,调用方只关心 delta 文本,末尾事件带 usage。
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

# 一次调用的用途;只用于埋点与 mock 行为分派,不影响真实模型语义
Purpose = str


class LLMError(RuntimeError):
    """模型调用失败(网络 / 鉴权 / 限流 / 响应格式),上层据此降级。"""


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            self.prompt_tokens + other.prompt_tokens,
            self.completion_tokens + other.completion_tokens,
            self.total_tokens + other.total_tokens,
        )

    def to_dict(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass
class ToolCall:
    """OpenAI 风格的函数调用请求。"""

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)

    def to_message(self) -> dict:
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": json.dumps(self.arguments, ensure_ascii=False),
            },
        }


@dataclass
class ChatResult:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    model: str = ""
    latency_ms: int = 0
    finish_reason: str = "stop"

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


@dataclass
class StreamEvent:
    """流式事件:delta 文本 或 结尾的 usage。"""

    delta: str = ""
    usage: Usage | None = None
    done: bool = False


@runtime_checkable
class BaseLLM(Protocol):
    """模型客户端协议。所有实现必须满足,测试里可直接用假实现替换。"""

    @property
    def backend(self) -> str: ...

    @property
    def model(self) -> str: ...

    def chat(
        self,
        messages: list[dict],
        *,
        system: str | None = None,
        tools: list[dict] | None = None,
        temperature: float = 0.3,
        max_tokens: int = 1024,
        model: str | None = None,
        thinking: bool = False,
        purpose: Purpose = "answer",
    ) -> ChatResult: ...

    def generate_text(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.2,
        thinking: bool = False,
        purpose: Purpose = "answer",
    ) -> str: ...

    def stream(
        self,
        messages: list[dict],
        *,
        system: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 1024,
        purpose: Purpose = "answer",
    ) -> Iterator[StreamEvent]: ...

    def close(self) -> None: ...


def estimate_tokens(text: str) -> int:
    """中文近似 token 估算(不引 tiktoken:中文不精确且增加依赖)。

    中文 1 字 ≈ 0.6 token,英文/符号按 4 字符 ≈ 1 token,取保守混合值。
    """
    if not text:
        return 0
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    other = len(text) - cjk
    return max(1, int(cjk * 0.6 + other / 4))


def messages_to_prompt(messages: list[dict]) -> str:
    """把消息列表压成单条 prompt(仅用于统计/调试)。"""
    parts = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content") or ""
        if isinstance(content, list):
            content = json.dumps(content, ensure_ascii=False)
        parts.append(f"[{role}] {content}")
    return "\n".join(parts)

"""LLM 工厂 —— 按配置构建客户端,带降级与单例缓存。"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Iterator

from app.config import Settings, settings
from app.llm.base import BaseLLM, ChatResult, StreamEvent
from app.llm.mock import MockLLM
from app.llm.openai_compat import OpenAICompatLLM

logger = logging.getLogger(__name__)


class FallbackLLM:
    """真实模型不可用(无 Key / 鉴权失败 / 网络断)时自动降级到 Mock。

    这样"配置了 Key 但现场网络抽风"的面试事故不会让 Demo 崩溃,
    同时页面顶栏会明确标注当前是离线模式,不误导面试官。
    """

    def __init__(self, primary, fallback: MockLLM) -> None:
        self.primary = primary
        self.fallback = fallback
        self.backend = primary.backend
        self.model = primary.model
        self.degraded = False
        self.last_error: str | None = None

    def _switch(self, exc: Exception) -> None:
        self.degraded = True
        self.last_error = str(exc)[:200]
        logger.warning("LLM 降级为离线模式: %s", exc)

    def chat(self, *args, **kwargs) -> ChatResult:
        if self.degraded:
            return self.fallback.chat(*args, **kwargs)
        try:
            return self.primary.chat(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            self._switch(exc)
            return self.fallback.chat(*args, **kwargs)

    def generate_text(self, *args, **kwargs) -> str:
        if self.degraded:
            return self.fallback.generate_text(*args, **kwargs)
        try:
            return self.primary.generate_text(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            self._switch(exc)
            return self.fallback.generate_text(*args, **kwargs)

    def stream(self, *args, **kwargs) -> Iterator[StreamEvent]:
        if self.degraded:
            yield from self.fallback.stream(*args, **kwargs)
            return
        try:
            got = False
            for event in self.primary.stream(*args, **kwargs):
                got = got or bool(event.delta)
                yield event
            if not got:  # 流式返回空:回退一次非流式,避免空回答
                yield from self.fallback.stream(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            self._switch(exc)
            yield from self.fallback.stream(*args, **kwargs)

    def close(self) -> None:
        self.primary.close()
        self.fallback.close()


def build_llm(cfg: Settings | None = None) -> BaseLLM:
    cfg = cfg or settings
    backend = cfg.resolved_llm_backend()
    if backend == "mock":
        logger.info("LLM 使用 MockLLM(未配置 LLM_API_KEY 或显式指定)")
        return MockLLM()
    primary = OpenAICompatLLM(
        base_url=cfg.llm_base_url,
        api_key=cfg.llm_api_key,
        model=cfg.llm_model,
        model_pro=cfg.llm_model_pro,
        model_fast=cfg.llm_model_fast,
        timeout=cfg.llm_timeout,
        max_retries=cfg.llm_max_retries,
        temperature=cfg.llm_temperature,
        proxy=cfg.llm_proxy,
    )
    logger.info("LLM 使用 %s (%s)", cfg.llm_base_url, cfg.llm_model)
    return FallbackLLM(primary, MockLLM())


_llm_cache: dict[int, BaseLLM] = {}


def get_llm(cfg: Settings | None = None):
    """进程级单例(httpx 连接池复用)。"""
    cfg = cfg or settings
    key = id(cfg)
    if key not in _llm_cache:
        _llm_cache[key] = build_llm(cfg)
    return _llm_cache[key]


def reset_llm_cache() -> None:
    for llm in _llm_cache.values():
        with contextlib.suppress(Exception):
            llm.close()
    _llm_cache.clear()

"""意图路由 —— 先分类,再决定"这次请求值不值得调用 LLM 生成"。

路由表:
    意图                 路径      LLM 调用
    weather/calculator/
    datetime (高置信)    fast      0 次(直接执行工具)
    small_talk           chat      1 次
    kb_query             rag       1 次(快速混合检索 + 生成)
    complex_agent / 低置信 agent    ≤3 次(Agent 循环)

低置信度永远回退 Agent,这是"规则不要拦死复杂问题"的兜底设计。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from app.agent.registry import ToolRegistry, ToolResult
from app.config import Settings, settings
from app.nlu.intent import IntentClassifier, get_classifier
from app.nlu.slots import extract_city, extract_expression

logger = logging.getLogger(__name__)

FAST_INTENTS = {"weather", "calculator", "datetime"}


@dataclass
class RoutePlan:
    intent: str
    confidence: float
    path: str  # fast | rag | chat | agent
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "intent": self.intent,
            "confidence": round(self.confidence, 3),
            "path": self.path,
            "reason": self.reason,
        }


@dataclass
class FastAnswer:
    answer: str
    tool_calls: list[ToolResult] = field(default_factory=list)
    intent: str = ""
    confidence: float = 0.0


class IntentRouter:
    def __init__(
        self,
        classifier: IntentClassifier | None = None,
        cfg: Settings | None = None,
    ) -> None:
        self.cfg = cfg or settings
        self.classifier = classifier or get_classifier()
        self.threshold = self.cfg.intent_threshold

    # ------------------------------------------------------------------
    def plan(self, text: str) -> RoutePlan:
        intent, confidence = self.classifier.predict(text)
        if not self.cfg.intent_enabled:
            return RoutePlan(intent, confidence, "agent", "意图路由已关闭")
        if confidence < self.threshold:
            return RoutePlan(
                intent, confidence, "agent", f"置信度 {confidence:.2f} < 阈值 {self.threshold}"
            )
        if intent == "complex_agent":
            return RoutePlan(intent, confidence, "agent", "复杂任务")
        if intent in FAST_INTENTS:
            return RoutePlan(intent, confidence, "fast", "确定性工具直达")
        if intent == "small_talk":
            return RoutePlan(intent, confidence, "chat", "闲聊")
        return RoutePlan(intent, confidence, "rag", "知识库问答")

    # ------------------------------------------------------------------
    def fast_answer(self, text: str, registry: ToolRegistry) -> FastAnswer | None:
        """执行 0-LLM 快路径。槽位抽不到时返回 None,由调用方回退。"""
        intent, confidence = self.classifier.predict(text)
        if intent == "weather":
            city = extract_city(text) or "北京"
            result = registry.execute("get_weather", {"city": city})
        elif intent == "calculator":
            expression = extract_expression(text)
            if not expression:
                return None
            result = registry.execute("calculator", {"expression": expression})
        elif intent == "datetime":
            result = registry.execute("get_datetime", {})
        else:
            return None
        return FastAnswer(
            answer=result.text, tool_calls=[result], intent=intent, confidence=confidence
        )


_router: IntentRouter | None = None


def get_router() -> IntentRouter:
    global _router
    if _router is None:
        _router = IntentRouter()
    return _router


def reset_router() -> None:
    global _router
    _router = None

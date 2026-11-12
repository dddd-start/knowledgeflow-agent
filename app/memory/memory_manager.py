"""记忆管理器 —— 把短期窗口、摘要、长期偏好、话题检测粘成一个门面。

上层(服务层 / API / CLI)只依赖这一个类:

    mem = MemoryManager(session_id, user_id)
    mem.before_user_turn(text)          # 必要时压缩 + 检测话题
    system_extra = mem.system_extra()   # 渲染进 system 的偏好与摘要
    history = mem.history()             # 滑动窗口内的消息
    mem.record_turn(user, assistant)    # 落一轮
    mem.extract_preferences(llm, conv)  # 异步抽取长期偏好
"""

from __future__ import annotations

import logging

from app.config import settings
from app.memory.long_term import LongTermMemory
from app.memory.short_term import ShortTermMemory
from app.memory.topic import TopicTracker

logger = logging.getLogger(__name__)


class MemoryManager:
    def __init__(self, session_id: str = "default", user_id: str = "default") -> None:
        self.session_id = session_id
        self.user_id = user_id
        self.short = ShortTermMemory()
        self.long = LongTermMemory()
        self.topic: TopicTracker | None = None

    # ------------------------------------------------------------------
    def _ensure_topic(self) -> TopicTracker:
        if self.topic is None:
            from app.rag.embeddings import get_embedder

            self.topic = TopicTracker(get_embedder())
        return self.topic

    def before_user_turn(self, text: str, llm=None) -> dict:
        """一轮开始前的记忆维护:压缩 + 话题检测。返回本次维护的事件信息。"""
        events: dict = {}
        if llm is not None:
            compressed = self.short.maybe_summarize(llm)
            if compressed:
                events["compressed_messages"] = compressed
        try:
            if self._ensure_topic().check_and_update(text):
                events["topic_changed"] = True
        except Exception as exc:  # noqa: BLE001
            logger.debug("话题检测跳过: %s", exc)
        return events

    def system_extra(self) -> dict:
        return {
            "user_preferences": self.long.load(self.user_id),
            "memory_summary": self.short.summary,
        }

    def history(self) -> list[dict]:
        return self.short.history()

    def record_turn(self, user: str, assistant: str) -> None:
        self.short.add_turn(user, assistant)

    def extract_preferences(self, llm, conversation: str) -> list[str]:
        return self.long.extract_and_store(llm, self.user_id, conversation)

    # ------------------------------------------------------------------
    def status(self) -> dict:
        return {
            "user_id": self.user_id,
            "session_id": self.session_id,
            "window_turns": self.short.window_turns,
            "turns": len(self.short.turns) // 2,
            "summary": self.short.summary,
            "estimated_tokens": self.short.estimated_tokens(),
            "compressed_messages": self.short.compressed_turns,
            "preferences": self.long.load(self.user_id),
            "topic_changes": self.topic.topic_changes if self.topic else 0,
            "threshold_tokens": settings.summary_threshold_tokens,
        }

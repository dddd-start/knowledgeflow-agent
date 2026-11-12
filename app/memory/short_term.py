"""短期记忆 —— 滑动窗口 + 超阈值自动摘要压缩。

只保留最近 N 轮完整消息;更早的轮次在超过 token 阈值时被 LLM 压缩成一段摘要,
摘要放回 system。这样长对话的 token 占用有上界,又不会"忘掉前面说过什么"。

为什么不用向量记忆存所有历史?
    对话历史需要**顺序和因果**,向量召回出来的是碎片;摘要 + 窗口是更符合对话语义的做法。
    跨会话的长期事实另有 LongTermMemory(偏好库)负责。
"""

from __future__ import annotations

import logging

from app.config import settings

logger = logging.getLogger(__name__)


def estimate_tokens(text: str) -> int:
    """中文近似 token 估算:1 汉字 ≈ 0.6 token(不引 tiktoken,避免重依赖)。"""
    if not text:
        return 0
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    return max(1, int(cjk * 0.6 + (len(text) - cjk) / 4))


class ShortTermMemory:
    def __init__(
        self,
        window_turns: int | None = None,
        summary_threshold_tokens: int | None = None,
    ) -> None:
        self.window_turns = window_turns or settings.memory_window_turns
        self.threshold = summary_threshold_tokens or settings.summary_threshold_tokens
        self.turns: list[dict] = []  # [{"role": "user"/"assistant", "content": str}]
        self.summary: str = ""
        self.compressed_turns = 0

    # ------------------------------------------------------------------
    def add_turn(self, user: str, assistant: str) -> None:
        self.turns.append({"role": "user", "content": user})
        self.turns.append({"role": "assistant", "content": assistant})

    def estimated_tokens(self) -> int:
        return estimate_tokens(self.summary) + sum(
            estimate_tokens(str(m.get("content", ""))) for m in self.turns
        )

    def maybe_summarize(self, llm) -> int:
        """超过阈值时把最旧的对话折叠进摘要,返回被折叠的消息条数。"""
        if self.estimated_tokens() <= self.threshold:
            return 0
        keep = self.window_turns * 2
        if len(self.turns) <= keep:
            return 0
        old, recent = self.turns[:-keep], self.turns[-keep:]
        transcript = "\n".join(f"[{m['role']}] {m['content']}" for m in old)
        try:
            from app.prompts.manager import get_prompt_manager

            prompt = get_prompt_manager().render(
                "summarizer", transcript=transcript, previous=self.summary
            )
            new_summary = llm.generate_text(prompt, max_tokens=256, purpose="summarize").strip()
            if new_summary:
                self.summary = new_summary[:500]
        except Exception as exc:  # noqa: BLE001  摘要失败不影响主流程
            logger.warning("对话摘要失败,直接丢弃最旧轮次: %s", exc)
        self.turns = recent
        self.compressed_turns += len(old)
        return len(old)

    def history(self) -> list[dict]:
        return list(self.turns)

    def reset(self) -> None:
        self.turns = []
        self.summary = ""

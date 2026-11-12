"""记忆层测试:滑动窗口、摘要压缩、长期偏好、话题漂移。"""

from __future__ import annotations

from app.memory.long_term import LongTermMemory
from app.memory.memory_manager import MemoryManager
from app.memory.short_term import ShortTermMemory, estimate_tokens
from app.memory.topic import TopicTracker


class EchoLLM:
    backend = "fake"
    model = "fake"

    def generate_text(self, prompt, **kwargs) -> str:
        return "上一轮聊了报销与差旅标准" if kwargs.get("purpose") == "summarize" else "[]"

    def chat(self, *args, **kwargs):
        raise AssertionError("不应调用 chat")


def test_estimate_tokens():
    assert estimate_tokens("") == 0
    assert estimate_tokens("你好") >= 1
    assert estimate_tokens("hello world") >= 1


def test_short_term_window_and_summary():
    mem = ShortTermMemory(window_turns=2, summary_threshold_tokens=10)
    for i in range(5):
        mem.add_turn(f"用户问题{i}", f"助手回答{i}" * 5)
    assert len(mem.turns) == 10
    compressed = mem.maybe_summarize(EchoLLM())
    assert compressed > 0
    assert len(mem.turns) == 4  # 只保留最近 2 轮
    assert mem.summary
    assert mem.compressed_turns == compressed


def test_short_term_no_compress_under_threshold():
    mem = ShortTermMemory(window_turns=5, summary_threshold_tokens=10000)
    mem.add_turn("问", "答")
    assert mem.maybe_summarize(EchoLLM()) == 0


def test_long_term_preferences(tmp_path):
    mem = LongTermMemory(tmp_path / "memory.db")
    mem.add("u1", "部门", "市场部")
    mem.add("u1", "常驻城市", "上海")
    assert mem.load("u1") == ["常驻城市: 上海", "部门: 市场部"]
    mem.add("u1", "部门", "销售部")  # 覆盖
    assert "部门: 销售部" in mem.load("u1")
    mem.delete_user("u1")
    assert mem.load("u1") == []
    mem.close()


def test_manager_status_and_history(tmp_path):
    manager = MemoryManager(session_id="s1", user_id="u1")
    manager.long.close()
    manager.long = LongTermMemory(tmp_path / "m.db")
    manager.record_turn("问题", "回答")
    status = manager.status()
    assert status["turns"] == 1
    assert status["session_id"] == "s1"
    assert manager.history()[0]["content"] == "问题"
    assert manager.system_extra()["memory_summary"] == ""


def test_topic_tracker_detects_change():
    class FixedEmbedder:
        def __init__(self):
            self.n = 0

        def encode(self, texts):
            import numpy as np

            self.n += 1
            # 第一次与主题同向,之后明显反向
            vec = np.array([1.0, 0.0]) if self.n == 1 else np.array([0.0, 1.0])
            return vec.reshape(1, -1)

    tracker = TopicTracker(FixedEmbedder(), threshold=0.5)
    assert tracker.check_and_update("报销") is False
    assert tracker.check_and_update("服务器部署") is True
    assert tracker.topic_changes == 1
    tracker.reset()
    assert tracker.check_and_update("新话题") is False

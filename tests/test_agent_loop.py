"""Agent 决策循环测试:工具协议、轮数上限、失败降级。"""

from __future__ import annotations

from app.agent.loop import AgentLoop
from app.agent.registry import ToolRegistry
from app.agent.tools import build_tools
from app.llm.base import ChatResult, ToolCall, Usage


class AlwaysToolLLM:
    """每轮都请求调用工具,用于验证轮数上限强制收敛。"""

    backend = "fake"
    model = "fake"

    def __init__(self) -> None:
        self.calls = 0

    def chat(self, messages, *, system=None, tools=None, **kwargs) -> ChatResult:
        self.calls += 1
        if tools is None:  # 收敛调用
            return ChatResult(text="最终答案", usage=Usage(1, 1, 2))
        return ChatResult(
            text="",
            tool_calls=[ToolCall(id=f"c{self.calls}", name="get_datetime", arguments={})],
            usage=Usage(1, 1, 2),
        )

    def generate_text(self, prompt, **kwargs) -> str:
        return "text"


class FailingLLM:
    backend = "fake"
    model = "fake"

    def chat(self, messages, **kwargs) -> ChatResult:
        raise RuntimeError("模型服务 502")

    def generate_text(self, prompt, **kwargs) -> str:
        raise RuntimeError("模型服务 502")


def test_agent_uses_mock_tool_flow(pipeline, mock_llm):
    registry = ToolRegistry(build_tools(pipeline))
    loop = AgentLoop(mock_llm, registry, system_prompt="你是助手", max_rounds=2)
    run = loop.run("上海天气怎么样")
    assert run.answer
    # 第 1 轮决定调用工具,第 2 轮基于工具结果给出答案
    assert run.rounds == 2
    assert [tc.name for tc in run.tool_calls] == ["get_weather"]
    assert "上海" in run.answer
    assert run.stopped_by == "final"
    registry.close()


def test_agent_max_rounds_forces_answer(pipeline):
    registry = ToolRegistry(build_tools(pipeline))
    llm = AlwaysToolLLM()
    loop = AgentLoop(llm, registry, system_prompt="你是助手", max_rounds=2)
    run = loop.run("现在几点")
    assert run.stopped_by == "max_rounds"
    assert run.rounds == 2
    assert run.answer == "最终答案"
    # 2 轮工具 + 1 轮收敛
    assert llm.calls == 3
    registry.close()


def test_agent_handles_llm_failure(pipeline):
    registry = ToolRegistry(build_tools(pipeline))
    loop = AgentLoop(FailingLLM(), registry, system_prompt="", max_rounds=2)
    run = loop.run("现在几点")
    assert run.stopped_by == "error"
    assert "不可用" in run.answer
    registry.close()


def test_agent_records_unknown_tool_error(pipeline):
    class UnknownToolLLM:
        backend = "fake"
        model = "fake"

        def __init__(self):
            self.used = False

        def chat(self, messages, *, tools=None, **kwargs):
            if not self.used and tools is not None:
                self.used = True
                return ChatResult(
                    text="",
                    tool_calls=[ToolCall(id="x", name="not_exist", arguments={})],
                    usage=Usage(),
                )
            return ChatResult(text="好的", usage=Usage())

    registry = ToolRegistry(build_tools(pipeline))
    run = AgentLoop(UnknownToolLLM(), registry, max_rounds=2).run("随便问问")
    assert run.tool_calls[0].is_error
    assert "未知工具" in run.tool_calls[0].text
    assert run.answer == "好的"
    registry.close()

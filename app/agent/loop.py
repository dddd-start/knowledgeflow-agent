"""Agent 决策循环 —— Function Calling / ReAct 的工程实现。

协议要点(最常见的 Agent bug 都出在这里):
1. 模型返回 tool_calls 后,assistant 消息必须**带原样的 tool_calls** 追加进历史;
2. 每个 tool_result 用 `tool_call_id` 与调用一一对应,role 固定为 `tool`;
3. 循环直到模型不再请求工具,或达到轮数上限。

防死循环与降级:
- `max_rounds` 硬上限(默认 3);
- 达到上限仍在调工具时,最后一轮**禁用工具**强制模型收敛出答案;
- 工具异常在 Registry 已转成错误文本,模型可以自我修正;
- 整轮失败(模型不可用)由上层 FallbackLLM 接管,不影响服务可用性。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from app.agent.registry import ToolRegistry, ToolResult
from app.config import settings
from app.llm.base import BaseLLM, Usage

logger = logging.getLogger(__name__)


@dataclass
class AgentRun:
    answer: str = ""
    rounds: int = 0
    tool_calls: list[ToolResult] = field(default_factory=list)
    messages: list[dict] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    stopped_by: str = "final"  # final | max_rounds | error

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "rounds": self.rounds,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "stopped_by": self.stopped_by,
        }


class AgentLoop:
    def __init__(
        self,
        llm: BaseLLM,
        registry: ToolRegistry,
        system_prompt: str = "",
        max_rounds: int | None = None,
    ) -> None:
        self.llm = llm
        self.registry = registry
        self.system_prompt = system_prompt
        self.max_rounds = max_rounds or settings.max_agent_rounds

    # ------------------------------------------------------------------
    def run(
        self,
        user_input: str,
        history: list[dict] | None = None,
        system: str | None = None,
    ) -> AgentRun:
        messages: list[dict] = list(history or [])
        messages.append({"role": "user", "content": user_input})
        system_prompt = system if system is not None else self.system_prompt
        run = AgentRun(messages=messages)

        for round_no in range(1, self.max_rounds + 1):
            run.rounds = round_no
            try:
                result = self.llm.chat(
                    messages,
                    system=system_prompt,
                    tools=self.registry.specs(),
                    temperature=0.2,
                    max_tokens=2048,
                    purpose="agent",
                )
            except Exception as exc:  # noqa: BLE001  模型不可用:返回可解释的降级答案
                logger.warning("Agent 第 %d 轮模型调用失败: %s", round_no, exc)
                run.answer = "抱歉,模型服务暂时不可用,请稍后再试。"
                run.stopped_by = "error"
                return run

            run.usage = run.usage + result.usage

            if not result.tool_calls:
                run.answer = result.text.strip()
                messages.append({"role": "assistant", "content": result.text})
                run.stopped_by = "final"
                return run

            # 1) assistant 消息原样带上 tool_calls
            messages.append(
                {
                    "role": "assistant",
                    "content": result.text or "",
                    "tool_calls": [call.to_message() for call in result.tool_calls],
                }
            )

            # 2) 逐个执行工具,回填 tool 消息
            for call in result.tool_calls:
                tool_result = self.registry.execute(call.name, call.arguments)
                run.tool_calls.append(tool_result)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": tool_result.text,
                    }
                )

        # 3) 轮数上限:禁用工具,强制收敛
        run.stopped_by = "max_rounds"
        try:
            final = self.llm.chat(
                messages,
                system=system_prompt
                + "\n\n[系统提示] 工具调用轮数已达上限,请基于已有信息直接给出最终回答。",
                tools=None,
                temperature=0.2,
                max_tokens=1024,
                purpose="agent",
            )
            run.answer = final.text.strip()
            run.usage = run.usage + final.usage
            messages.append({"role": "assistant", "content": run.answer})
        except Exception as exc:  # noqa: BLE001
            logger.warning("Agent 收敛失败: %s", exc)
            run.answer = "工具调用轮数已达上限,未能收敛出最终答案,请简化问题后重试。"
        return run

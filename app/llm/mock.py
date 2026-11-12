"""MockLLM —— 确定性离线模型(无 Key / 无网络 / CI 单测 / 现场演示兜底)。

它不是"返回固定字符串"的假实现,而是一个**规则驱动的降级引擎**:
- 能看懂 RAG 上下文里的编号证据,生成带引用编号的回答;
- 能按 agent 的工具协议返回 tool_calls(kb_search / calculator / get_weather / get_datetime),
  让 Agent 循环、工具执行、多轮编排在离线时也能跑通;
- 能按 `purpose` 区分业务调用,给 judge / self_check 返回结构化 JSON。

价值:面试官在自己电脑上 `git clone && ./run.sh` 就能看到完整链路,不需要 API Key;
同时 CI 里所有测试都不依赖外部服务。
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Iterator

from app.llm.base import ChatResult, StreamEvent, ToolCall, Usage, estimate_tokens

_WEATHER = re.compile(r"天气|气温|下雨|温度|几度|冷不冷|热不热")
_CALC = re.compile(r"(\d+(?:\.\d+)?\s*[+\-*/×÷]\s*\d+(?:\.\d+)?)|计算|等于多少|算一下")
_TIME = re.compile(r"几点|现在时间|日期|星期几|周几|今天几号|几号了")
_GREET = re.compile(r"^(你好|您好|hi|hello|嗨|在吗|早上好|晚上好)[!！。,. ]*$", re.I)
_KB_HINT = re.compile(
    r"报销|差旅|发票|考勤|请假|年假|加班|入职|转正|培训|绩效|采购|预算|"
    r"账号|权限|VPN|密码|邮箱|打印机|会议室|资产|发版|上线|部署|回滚|"
    r"数据|安全|合规|审计|应急|客服|产品|版本|定价|SLA|流程|制度|规定"
)
_CITY = re.compile(
    r"(北京|上海|广州|深圳|杭州|成都|武汉|西安|南京|苏州|长沙|重庆|天津|青岛|郑州|合肥|厦门)"
)

_CONTEXT_LINE = re.compile(r"^\[(\d+)\]\s*《([^》]+)》", re.MULTILINE)


class MockLLM:
    """实现 BaseLLM 协议;backend 字段供 /api/meta 展示。"""

    backend = "mock"
    model = "mock-deterministic"

    def __init__(self, *, chunk_delay: float = 0.008) -> None:
        self.chunk_delay = chunk_delay
        self.calls: list[dict] = []  # 便于测试断言调用构成

    # ------------------------------------------------------------------ 工具判断
    @staticmethod
    def _last_user(messages: list[dict]) -> str:
        for msg in reversed(messages):
            if msg.get("role") == "user" and isinstance(msg.get("content"), str):
                return msg["content"]
        return ""

    def _decide_tool(self, text: str) -> tuple[str, dict] | None:
        if _GREET.match(text or ""):
            return None
        if _WEATHER.search(text):
            city = _CITY.search(text)
            return "get_weather", {"city": city.group(1) if city else "北京"}
        if _CALC.search(text):
            m = re.search(r"[\d.()\s+\-*/×÷]+", text)
            expr = (m.group(0) if m else "0").replace("×", "*").replace("÷", "/")
            return "calculator", {"expression": expr.strip()}
        if _TIME.search(text):
            return "get_datetime", {}
        if _KB_HINT.search(text) or len(text) > 8:
            return "kb_search", {"query": text}
        return None

    # ------------------------------------------------------------------ 回答合成
    @staticmethod
    def _context_blocks(system: str | None) -> list[tuple[str, str]]:
        """从 RAG system 里抽出 [(标题, 正文)] 证据块。"""
        if not system:
            return []
        blocks: list[tuple[str, str]] = []
        matches = list(_CONTEXT_LINE.finditer(system))
        for i, m in enumerate(matches):
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(system)
            body = system[start:end].strip()
            body = re.sub(r"^\s*(doc_id|来源|分类)[:：].*$", "", body, flags=re.MULTILINE).strip()
            blocks.append((m.group(2), body))
        return blocks

    def _compose_answer(self, text: str, system: str | None) -> str:
        blocks = self._context_blocks(system)
        if not blocks:
            if _GREET.match(text or ""):
                return (
                    "你好,我是 KnowledgeFlow 知识库助手。可以问我公司制度、流程类问题,"
                    "比如「出差住宿标准是多少」「VPN 怎么申请」。"
                )
            return (
                "我目前运行在离线演示模式(未配置 LLM Key),只能基于知识库片段作答。"
                "如果你问的是公司制度/流程类问题,请补充更具体的说法。"
            )
        head = "根据知识库中的资料:\n\n"
        lines = []
        for idx, (title, body) in enumerate(blocks[:3], start=1):
            snippet = re.split(r"(?<=[。;!?])", body.strip())[0].strip()
            snippet = snippet[:160] + ("…" if len(snippet) > 160 else "")
            lines.append(f"{idx}. 《{title}》指出:{snippet}[{idx}]")
        tail = "\n\n以上结论均可在右侧「检索证据」中查看原文片段。"
        return head + "\n".join(lines) + tail

    @staticmethod
    def _tool_answer(tool_msgs: list[dict]) -> str:
        if not tool_msgs:
            return ""
        content = str(tool_msgs[-1].get("content", ""))
        return content.strip()

    # ------------------------------------------------------------------ 协议实现
    def chat(
        self,
        messages: list[dict],
        *,
        system: str | None = None,
        tools: list[dict] | None = None,
        temperature: float | None = None,
        max_tokens: int = 1024,
        model: str | None = None,
        thinking: bool = False,
        purpose: str = "answer",
    ) -> ChatResult:
        started = time.perf_counter()
        text = self._last_user(messages)
        usage = Usage(
            prompt_tokens=estimate_tokens(system or "") + estimate_tokens(text),
            completion_tokens=0,
        )
        usage.total_tokens = usage.prompt_tokens

        # ---- 结构化用途:查询改写 / 证据判断 / 自检 / 摘要 / 偏好抽取 ----
        if purpose == "rewrite":
            answer = re.sub(r"^(请问|你好,?|我想问一下|麻烦问一下)", "", text).strip() or text
        elif purpose == "hyde":
            q = self._last_user(messages)
            answer = (
                f"关于「{q[:60]}」,公司制度和操作流程通常包含适用范围、申请入口、"
                "审批层级、时限要求与责任人等要素。"
            )
        elif purpose == "judge":
            answer = json.dumps(
                {"sufficient": True, "missing": "", "reason": "离线模式默认证据充分"},
                ensure_ascii=False,
            )
        elif purpose == "self_check":
            answer = json.dumps(
                {
                    "verdict": "supported",
                    "reason": "回答均可在证据中找到依据",
                    "revised_answer": "",
                },
                ensure_ascii=False,
            )
        elif purpose == "summarize":
            answer = f"(摘要) {text[:120]}"
        elif purpose == "preferences":
            answer = "[]"
        elif purpose == "agentic_query":
            answer = text
        elif purpose == "agent" and tools:
            # 已经执行过工具 -> 给出最终答案;否则决定是否调用工具
            tool_msgs = [m for m in messages if m.get("role") == "tool"]
            if tool_msgs:
                answer = self._compose_from_tool(tool_msgs, system)
            else:
                decision = self._decide_tool(text)
                if decision:
                    name, args = decision
                    call = ToolCall(
                        id=f"call_{abs(hash((name, json.dumps(args, sort_keys=True)))) % 10**8}",
                        name=name,
                        arguments=args,
                    )
                    usage.completion_tokens = 12
                    usage.total_tokens += 12
                    self.calls.append({"purpose": purpose, "tool": name})
                    return ChatResult(
                        text="",
                        tool_calls=[call],
                        usage=usage,
                        model=self.model,
                        latency_ms=int((time.perf_counter() - started) * 1000),
                        finish_reason="tool_calls",
                    )
                answer = self._compose_answer(text, system)
        else:
            answer = self._compose_answer(text, system)

        usage.completion_tokens = estimate_tokens(answer)
        usage.total_tokens = usage.prompt_tokens + usage.completion_tokens
        self.calls.append({"purpose": purpose})
        return ChatResult(
            text=answer,
            usage=usage,
            model=self.model,
            latency_ms=int((time.perf_counter() - started) * 1000),
            finish_reason="stop",
        )

    def _compose_from_tool(self, tool_msgs: list[dict], system: str | None) -> str:
        """工具结果 -> 最终回答:知识库结果统一转成带引用的口吻。"""
        content = str(tool_msgs[-1].get("content", "")).strip()
        if not content or content.startswith("知识库中没有找到"):
            return (
                "知识库中没有检索到与这个问题相关的内容。"
                "可以换一种问法,或确认该制度是否已上传到知识库。" + (content or "")
            )
        blocks = self._context_blocks(content)
        if blocks:
            return self._compose_answer("", content)
        # 计算器 / 天气 / 时间等确定性工具:直接回填结果
        return f"查询结果:{content}"

    def generate_text(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.2,
        thinking: bool = False,
        purpose: str = "answer",
    ) -> str:
        return self.chat(
            [{"role": "user", "content": prompt}],
            system=system,
            max_tokens=max_tokens,
            purpose=purpose,
        ).text.strip()

    def stream(
        self,
        messages: list[dict],
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int = 1024,
        purpose: str = "answer",
    ) -> Iterator[StreamEvent]:
        result = self.chat(messages, system=system, max_tokens=max_tokens, purpose=purpose)
        text = result.text
        step = 8
        for i in range(0, len(text), step):
            if self.chunk_delay:
                time.sleep(self.chunk_delay)
            yield StreamEvent(delta=text[i : i + step])
        yield StreamEvent(usage=result.usage, done=True)

    def close(self) -> None:  # pragma: no cover - 无资源
        return None


# 常见问题:mock 产生的 KB 片段来自 kb_search 的返回格式,
# 这里把 `[来源:标题 | 文档ID:xxx]` 也视为证据块,便于统一引用。
_CONTEXT_LINE_FALLBACK = re.compile(r"\[来源:([^|]+)\|", re.MULTILINE)


def context_titles(text: str) -> list[str]:
    """工具输出 -> 被引用文档标题列表(测试/评测复用)。"""
    return [m.group(1).strip() for m in _CONTEXT_LINE_FALLBACK.finditer(text)]

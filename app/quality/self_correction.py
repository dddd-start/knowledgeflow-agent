"""幻觉自检 —— 用"独立 judge + 证据"给草稿做事实核查。

为什么不能"让模型自己检查"?
    模型生成草稿时已经形成了错误信念,再问它"你确定吗"通常得到"确定";
    而把【检索到的证据】和【草稿】一起交给一次全新的调用,让它逐条核对,
    才能发现"证据里没有的数字/流程"。

三种判定:
- supported   : 结论都有依据 -> 直接输出;
- partial     : 部分无依据 -> 采纳 judge 给出的修订稿;
- unsupported : 基本是编造 -> 拒答并说明知识库未覆盖(宁可不答,不可错答)。
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from app.rag.schemas import Hit

logger = logging.getLogger(__name__)

_VERDICTS = {"supported", "partial", "unsupported"}

REFUSAL = (
    "知识库中没有找到足够依据来回答这个问题,为避免误导,我不做推测。\n"
    "建议:换一种问法、补充上下文,或联系对应的责任部门确认。"
)


@dataclass
class CheckResult:
    verdict: str
    reason: str = ""
    revised_answer: str = ""

    def to_dict(self) -> dict:
        return {"verdict": self.verdict, "reason": self.reason}


class SelfChecker:
    def __init__(self, llm, enabled: bool | None = None) -> None:
        self.llm = llm
        self.enabled = enabled

    # ------------------------------------------------------------------
    def check(self, question: str, draft: str, evidence: list[Hit]) -> CheckResult:
        if not evidence:
            return CheckResult("unsupported", "没有任何检索证据")
        try:
            from app.prompts.manager import get_prompt_manager

            prompt = get_prompt_manager().render(
                "self_check",
                question=question,
                draft=draft,
                evidence=[{"title": h.title, "text": h.text[:1200]} for h in evidence],
            )
            raw = self.llm.generate_text(
                prompt, max_tokens=400, temperature=0.0, purpose="self_check"
            )
            data = _parse_json_object(raw) or {}
        except Exception as exc:  # noqa: BLE001
            logger.warning("自检调用失败,放行草稿: %s", exc)
            return CheckResult("supported", "自检不可用,按放行处理")

        verdict = str(data.get("verdict", "supported")).strip().lower()
        if verdict not in _VERDICTS:
            verdict = "partial"
        return CheckResult(
            verdict=verdict,
            reason=str(data.get("reason", ""))[:200],
            revised_answer=str(data.get("revised_answer", "") or "").strip(),
        )

    def apply(self, question: str, draft: str, evidence: list[Hit]) -> tuple[str, CheckResult]:
        result = self.check(question, draft, evidence)
        if result.verdict == "supported":
            return draft, result
        if result.verdict == "partial" and result.revised_answer:
            return result.revised_answer, result
        return REFUSAL, result


def _parse_json_object(raw: str) -> dict | None:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", (raw or "").strip(), flags=re.DOTALL)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None

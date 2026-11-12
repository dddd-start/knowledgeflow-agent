"""Agentic RAG —— 让检索由 Agent 主动驱动(CRAG 风格迭代检索)。

朴素 RAG:  问题 → 检索一次 → 生成            (检索是被动的、一次性的)
Agentic:   问题 → 生成查询 → 检索 → 判断证据够不够
                  ├─ 不够 → 带着"缺什么"的反馈改写查询 → 再检索 → 再判断 …
                  └─ 够   → 基于证据生成回答

适用场景:多跳问题("出差住酒店的标准是多少?超过标准怎么审批?"),
一次检索往往只覆盖其中一半,需要按"缺口"再查一轮。

停止条件:证据充分 或 达到轮数上限(默认 3);上限时用最后一轮证据诚实作答,
不会无限检索。
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

from app.config import settings
from app.rag.pipeline import RetrievalPipeline
from app.rag.schemas import Hit

logger = logging.getLogger(__name__)


@dataclass
class AgenticRound:
    index: int
    query: str
    hits: list[Hit] = field(default_factory=list)
    sufficient: bool = False
    missing: str = ""
    ms: int = 0

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "query": self.query,
            "hits": [h.to_dict(text_limit=200) for h in self.hits[:5]],
            "sufficient": self.sufficient,
            "missing": self.missing,
            "ms": self.ms,
        }


@dataclass
class AgenticResult:
    answer: str = ""
    rounds: list[AgenticRound] = field(default_factory=list)
    hits: list[Hit] = field(default_factory=list)
    sufficient: bool = False

    @property
    def sources(self) -> list[Hit]:
        return self.hits


class AgenticRAG:
    def __init__(self, llm, pipeline: RetrievalPipeline, max_rounds: int | None = None) -> None:
        self.llm = llm
        self.pipeline = pipeline
        self.max_rounds = max_rounds or settings.agentic_max_rounds

    # ------------------------------------------------------------------
    def run(self, question: str) -> AgenticResult:
        import time

        result = AgenticResult()
        feedback = ""
        for index in range(1, self.max_rounds + 1):
            started = time.perf_counter()
            query = self._generate_query(question, feedback)
            hits, _trace = self.pipeline.retrieve(query, mode="fast", top_k=5)
            round_info = AgenticRound(index=index, query=query, hits=hits)

            if not hits:
                round_info.missing = "知识库没有检索到任何相关内容"
                feedback = round_info.missing
            else:
                judge = self._judge(question, hits)
                round_info.sufficient = bool(judge.get("sufficient"))
                round_info.missing = str(judge.get("missing", ""))
                feedback = round_info.missing
                if round_info.sufficient:
                    result.sufficient = True
            round_info.ms = int((time.perf_counter() - started) * 1000)
            result.rounds.append(round_info)
            if round_info.sufficient:
                break

        # 合并所有轮次的证据(按 chunk 去重),用最后一批做生成
        merged: dict[str, Hit] = {}
        for rnd in result.rounds:
            for hit in rnd.hits:
                merged.setdefault(hit.chunk_id, hit)
        result.hits = list(merged.values())[:5]

        if not result.hits:
            result.answer = "知识库中没有覆盖这个问题,建议换一种问法或联系对应部门。"
            return result

        from app.prompts.manager import get_prompt_manager

        context = self.pipeline.format_context(result.hits)
        system = get_prompt_manager().render("answer", context=context)
        try:
            result.answer = self.llm.generate_text(
                question, system=system, max_tokens=1024, purpose="answer"
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Agentic 生成失败: %s", exc)
            result.answer = "生成回答失败,请稍后再试。"
        return result

    # ------------------------------------------------------------------
    def _generate_query(self, question: str, feedback: str) -> str:
        from app.prompts.manager import get_prompt_manager

        if not feedback:
            return question
        try:
            prompt = get_prompt_manager().render(
                "agentic_query", question=question, feedback=feedback
            )
            text = self.llm.generate_text(
                prompt, max_tokens=128, temperature=0.2, purpose="agentic_query"
            ).strip()
            return text.splitlines()[0][:120] if text else question
        except Exception:  # noqa: BLE001
            return question

    def _judge(self, question: str, hits: list[Hit]) -> dict:
        from app.prompts.manager import get_prompt_manager

        try:
            prompt = get_prompt_manager().render(
                "agentic_judge",
                question=question,
                evidence=[{"title": h.title, "text": h.text[:1200]} for h in hits],
            )
            raw = self.llm.generate_text(prompt, max_tokens=200, purpose="judge")
            data = _parse_json_object(raw) or {}
        except Exception as exc:  # noqa: BLE001
            logger.warning("证据判断失败,默认充分: %s", exc)
            return {"sufficient": True, "missing": ""}
        return {
            "sufficient": _to_bool(data.get("sufficient")),
            "missing": str(data.get("missing", ""))[:120],
        }


def _to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1", "是"}
    return bool(value)


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

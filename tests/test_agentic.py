"""Agentic RAG 测试:迭代检索与停止条件。"""

from __future__ import annotations

import json

from app.agentic.agentic_rag import AgenticRAG


class ScriptedLLM:
    """按 purpose 返回脚本化结果,用于验证多轮检索逻辑。"""

    backend = "fake"
    model = "fake"

    def __init__(self, judgments: list[dict]) -> None:
        self.judgments = judgments
        self.judge_calls = 0
        self.queries: list[str] = []

    def generate_text(self, prompt, **kwargs) -> str:
        purpose = kwargs.get("purpose")
        if purpose == "judge":
            payload = self.judgments[min(self.judge_calls, len(self.judgments) - 1)]
            self.judge_calls += 1
            return json.dumps(payload, ensure_ascii=False)
        if purpose == "agentic_query":
            self.queries.append(prompt[:30])
            return "报销审批层级是什么" if self.judge_calls else "出差住宿标准是多少"
        return "根据知识库资料，出差住宿按城市分级执行 [1]。"

    def chat(self, *args, **kwargs):
        raise AssertionError("不应调用 chat")


def test_agentic_stops_when_sufficient(pipeline):
    llm = ScriptedLLM([{"sufficient": True, "missing": ""}])
    result = AgenticRAG(llm, pipeline, max_rounds=2).run("出差住宿标准是多少")
    assert len(result.rounds) == 1
    assert result.sufficient is True
    assert result.hits
    assert result.answer


def test_agentic_iterates_when_insufficient(pipeline):
    llm = ScriptedLLM(
        [
            {"sufficient": False, "missing": "缺少审批层级"},
            {"sufficient": True, "missing": ""},
        ]
    )
    result = AgenticRAG(llm, pipeline, max_rounds=3).run("出差住宿标准与超标审批")
    assert len(result.rounds) == 2
    assert result.rounds[0].sufficient is False
    assert result.rounds[1].sufficient is True
    assert result.rounds[0].missing == "缺少审批层级"
    assert llm.judge_calls == 2


def test_agentic_max_rounds(pipeline):
    llm = ScriptedLLM([{"sufficient": False, "missing": "还不够"}])
    result = AgenticRAG(llm, pipeline, max_rounds=2).run("复杂多跳问题")
    assert len(result.rounds) == 2
    assert result.sufficient is False
    assert result.answer  # 上限时仍要用最后证据作答


def test_agentic_empty_kb(cfg, embedder):
    from app.llm.mock import MockLLM
    from app.rag.pipeline import RetrievalPipeline
    from app.rag.rerank import Reranker
    from app.rag.vectorstore import NumpyVectorStore

    pipe = RetrievalPipeline(
        MockLLM(),
        embedder=embedder,
        store=NumpyVectorStore(cfg.index_dir / "empty2"),
        reranker=Reranker(backend="none"),
        cfg=cfg,
    )
    result = AgenticRAG(MockLLM(), pipe, max_rounds=2).run("知识库里没有的问题")
    assert result.hits == []
    assert "没有覆盖" in result.answer

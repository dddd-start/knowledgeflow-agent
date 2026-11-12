"""幻觉自检测试:三种判定与拒答。"""

from __future__ import annotations

import json

from app.quality.self_correction import REFUSAL, SelfChecker
from app.rag.schemas import Hit


class FakeJudge:
    backend = "fake"
    model = "fake"

    def __init__(self, payload: dict | str) -> None:
        self.payload = payload
        self.calls = 0

    def generate_text(self, prompt, **kwargs) -> str:
        self.calls += 1
        return (
            self.payload
            if isinstance(self.payload, str)
            else json.dumps(self.payload, ensure_ascii=False)
        )

    def chat(self, *args, **kwargs):
        raise AssertionError("不应该调用 chat")


EVIDENCE = [
    Hit(
        chunk_id="a#0",
        doc_id="doc-a",
        title="费用报销流程",
        text="费用发生后 30 天内提交报销。",
        score=0.8,
    )
]


def test_supported_keeps_draft():
    checker = SelfChecker(FakeJudge({"verdict": "supported", "reason": "有依据"}))
    final, result = checker.apply("报销时限", "30 天内提交 [1]", EVIDENCE)
    assert result.verdict == "supported"
    assert final == "30 天内提交 [1]"


def test_partial_uses_revised_answer():
    checker = SelfChecker(
        FakeJudge(
            {"verdict": "partial", "reason": "金额缺依据", "revised_answer": "30 天内提交 [1]"}
        )
    )
    final, result = checker.apply("报销时限", "30 天内提交，限额 5000 [1]", EVIDENCE)
    assert result.verdict == "partial"
    assert final == "30 天内提交 [1]"


def test_unsupported_refuses():
    checker = SelfChecker(FakeJudge({"verdict": "unsupported", "reason": "证据里没有"}))
    final, result = checker.apply("公司股价", "大约 100 元", EVIDENCE)
    assert result.verdict == "unsupported"
    assert final == REFUSAL
    assert "没有找到足够依据" in final


def test_no_evidence_short_circuits():
    checker = SelfChecker(FakeJudge({"verdict": "supported"}))
    final, result = checker.apply("问", "答", [])
    assert result.verdict == "unsupported"
    assert final == REFUSAL


def test_invalid_json_passes_through():
    # judge 输出不可解析时保守放行(避免把服务搞挂),但会记录 supported
    checker = SelfChecker(FakeJudge("模型跑偏了，没有 JSON"))
    final, result = checker.apply("报销时限", "30 天内", EVIDENCE)
    assert result.verdict == "supported"
    assert final == "30 天内"


def test_judge_exception_passes_through():
    class Boom(FakeJudge):
        def generate_text(self, prompt, **kwargs):
            raise RuntimeError("超时")

    final, result = SelfChecker(Boom({})).apply("问", "草稿", EVIDENCE)
    assert result.verdict == "supported"
    assert final == "草稿"

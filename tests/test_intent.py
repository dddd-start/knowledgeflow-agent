"""意图识别与路由测试。"""

from __future__ import annotations

from app.agent.registry import ToolRegistry
from app.agent.tools import build_tools
from app.nlu.intent import RuleClassifier
from app.nlu.router import IntentRouter


def test_rule_classifier_intents():
    clf = RuleClassifier()
    cases = {
        "北京天气怎么样": "weather",
        "帮我算一下 128*7": "calculator",
        "现在几点了": "datetime",
        "你好": "small_talk",
        "补卡一个月能补几次": "kb_query",
        "对比一下两版部署方案并给出优化建议": "complex_agent",
    }
    for text, expected in cases.items():
        intent, confidence = clf.predict(text)
        assert intent == expected, f"{text} -> {intent}, 期望 {expected}"
        assert 0 < confidence <= 1


def test_router_threshold_fallback(cfg, router):
    # 低置信度问题回退 Agent
    plan = router.plan("嗯……那个东西怎么弄")
    assert plan.path == "agent"
    assert plan.confidence < cfg.intent_threshold or plan.reason


def test_router_paths(router):
    assert router.plan("上海天气怎么样").path == "fast"
    assert router.plan("帮我算一下 99*3").path == "fast"
    assert router.plan("现在几点").path == "fast"
    assert router.plan("你好").path == "chat"
    assert router.plan("报销打款是每月几号").path == "rag"
    assert router.plan("对比两套部署方案并给出优化建议").path == "agent"


def test_router_disabled_goes_agent(cfg):
    cfg.intent_enabled = False
    router = IntentRouter(classifier=RuleClassifier(), cfg=cfg)
    assert router.plan("现在几点").path == "agent"


def test_fast_answer_slots(pipeline, router):
    registry = ToolRegistry(build_tools(pipeline))
    weather = router.fast_answer("上海天气怎么样", registry)
    assert weather is not None and weather.intent == "weather"
    assert "上海" in weather.answer
    assert weather.tool_calls[0].name == "get_weather"

    calc = router.fast_answer("帮我算一下 (1200+300)*0.8", registry)
    assert calc is not None and "1200" in calc.answer

    now = router.fast_answer("现在几点", registry)
    assert now is not None and ":" in now.answer

    # 不是快路径 -> None
    assert router.fast_answer("报销流程是什么", registry) is None
    assert router.fast_answer("帮我算一下这个", registry) is None
    registry.close()

"""工具层测试:计算器安全、工具协议、注册表校验/超时/异常隔离。"""

from __future__ import annotations

import time

from app.agent.registry import ToolRegistry, _validate
from app.agent.tools import (
    build_tools,
    calculator,
    get_datetime,
    get_weather,
)


def test_calculator_basic():
    assert calculator("1+2*3") == "1+2*3 = 7"
    assert calculator("(1200+300)*0.8") == "(1200+300)*0.8 = 1200"
    assert calculator("10/4") == "10/4 = 2.5"


def test_calculator_rejects_unsafe_input():
    assert "错误" in calculator("__import__('os').system('ls')")
    assert "错误" in calculator("2**10")
    assert "错误" in calculator("abc")
    assert "错误" in calculator("")


def test_datetime_and_weather():
    now = get_datetime()
    assert any(day in now for day in ("周一", "周二", "周三", "周四", "周五", "周六", "周日"))
    assert "上海" in get_weather("上海")
    assert "暂无" in get_weather("不存在市")


def test_build_tools_specs(pipeline):
    tools = build_tools(pipeline)
    assert {t.name for t in tools} == {
        "kb_search",
        "kb_read",
        "calculator",
        "get_datetime",
        "get_weather",
    }
    for tool in tools:
        spec = tool.spec()
        assert spec["type"] == "function"
        assert spec["function"]["name"] == tool.name
        assert "parameters" in spec["function"]


def test_kb_search_and_read(pipeline):
    registry = ToolRegistry(build_tools(pipeline))
    result = registry.execute("kb_search", {"query": "报销打款时间"})
    assert not result.is_error
    assert result.name == "kb_search"
    assert "《" in result.text
    # 从搜索结果里取出 doc_id,再用 kb_read 读全文
    import re

    match = re.search(r"doc_id=(doc-[^,]+)", result.text)
    assert match
    read = registry.execute("kb_read", {"doc_id": match.group(1)})
    assert not read.is_error
    assert len(read.text) > len("《")


def test_registry_validation(pipeline):
    registry = ToolRegistry(build_tools(pipeline))
    missing = registry.execute("get_weather", {})
    assert missing.is_error and "缺少必填参数" in missing.text
    wrong_type = registry.execute("get_weather", {"city": 123})
    assert wrong_type.is_error and "应为字符串" in wrong_type.text
    unknown = registry.execute("nope", {})
    assert unknown.is_error and "未知工具" in unknown.text


def test_registry_timeout(pipeline):
    from app.agent.tools import Tool

    slow = Tool(
        name="slow",
        description="sleep",
        parameters={"type": "object", "properties": {}},
        func=lambda: (time.sleep(0.5), "done")[1],
        timeout_s=0.05,
    )
    registry = ToolRegistry([slow])
    result = registry.execute("slow", {})
    assert result.is_error
    assert "超时" in result.text
    registry.close()


def test_registry_truncates_long_output(pipeline):
    from app.agent.tools import Tool

    tool = Tool(
        name="long",
        description="long",
        parameters={"type": "object", "properties": {}},
        func=lambda: "x" * 10000,
        max_output_chars=100,
    )
    result = ToolRegistry([tool]).execute("long", {})
    assert len(result.text) < 300
    assert "截断" in result.text


def test_validate_schema_types():
    schema = {
        "type": "object",
        "properties": {
            "n": {"type": "integer"},
            "flag": {"type": "boolean"},
            "mode": {"type": "string", "enum": ["a", "b"]},
        },
        "required": ["n"],
    }
    assert _validate(schema, {"n": 1}) is None
    assert "应为整数" in _validate(schema, {"n": "1"})
    assert "布尔值" in _validate(schema, {"n": 1, "flag": "yes"})
    assert "取值" in _validate(schema, {"n": 1, "mode": "c"})

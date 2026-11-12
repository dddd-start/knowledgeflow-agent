"""Agent 工具 —— 5 个标准化工具,统一 JSON Schema 描述(OpenAI Function Calling 协议)。

工具设计原则(面试常被追问"为什么是这几个工具"):
1. `kb_search` / `kb_read` 成对出现:先搜索拿片段,片段不够再看全文 —— 模仿人类查文档;
   这一步把"检索"从 RAG 的固定流程变成 Agent 的主动动作;
2. `calculator` / `get_datetime` 是确定性能力,交给 LLM 自己算容易错,工具化更可靠;
3. `get_weather` 是外部 API 的代表(演示用本地假数据,接口形态与真实 API 一致)。

安全:
- calculator 用 AST 白名单求值,不用 eval(简历里"输入护轨"的具体体现);
- 所有工具输出统一截断,防止一次工具调用把上下文打爆。
"""

from __future__ import annotations

import ast
import datetime
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict  # JSON Schema
    func: Callable[..., str]
    max_output_chars: int = 2000
    timeout_s: float = 8.0
    meta: dict = field(default_factory=dict)

    def spec(self) -> dict:
        """转成 OpenAI tools 格式。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


# ----------------------------------------------------------------------
# 知识库工具
# ----------------------------------------------------------------------
def _kb_search(pipeline, query: str) -> str:
    if not query.strip():
        return "错误:query 不能为空"
    hits, _trace = pipeline.retrieve(query, mode="fast", top_k=3)
    if not hits:
        return "知识库中没有找到相关内容。"
    return pipeline.format_context(hits, text_limit=600)


def _kb_read(store, doc_id: str) -> str:
    chunks = store.get_document(doc_id)
    if not chunks:
        return f"未找到文档 {doc_id}(请先用 kb_search 获取正确的文档ID)"
    title = chunks[0].metadata.get("title", doc_id)
    body = "\n\n".join(c.text for c in chunks)
    return f"《{title}》(doc_id={doc_id},共 {len(chunks)} 个片段)\n{body[:3000]}"


# ----------------------------------------------------------------------
# 确定性工具
# ----------------------------------------------------------------------
_ALLOWED_NODES = {
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Mod,
    ast.FloorDiv,
    ast.USub,
    ast.UAdd,
    ast.Constant,
    ast.Load,
    ast.Pow,
}
_CN_OP = {"×": "*", "÷": "/", "＋": "+", "－": "-", "（": "(", "）": ")"}


def calculator(expression: str) -> str:
    expr = expression or ""
    for cn, en in _CN_OP.items():
        expr = expr.replace(cn, en)
    expr = re.sub(r"[^0-9.+\-*/%() ]", "", expr).strip()
    if not expr:
        return "错误:表达式为空"
    if "**" in expr or expr.count("/") > 5:
        return "错误:表达式过于复杂或包含不支持的运算"
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return f"错误:无法解析表达式 {expr}"
    for node in ast.walk(tree):
        if type(node) not in _ALLOWED_NODES:
            return f"错误:不支持的语法 {type(node).__name__}"
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
            return "错误:只支持数字常量"
        if isinstance(node, ast.Pow):
            return "错误:不支持幂运算"
    try:
        value = eval(compile(tree, "<calc>", "eval"), {"__builtins__": {}}, {})
    except Exception as exc:  # noqa: BLE001
        return f"错误:计算失败 {exc}"
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return f"{expr} = {value}"


def get_datetime() -> str:
    now = datetime.datetime.now().astimezone()
    weekdays = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
    return f"{now.strftime('%Y-%m-%d %H:%M:%S')} {weekdays[now.weekday()]} ({now.tzname()})"


_WEATHER_TABLE = {
    "北京": "晴,26°C,微风,湿度 40%",
    "上海": "多云转阴,28°C,东南风 3 级,湿度 65%",
    "广州": "阵雨,30°C,南风 2 级,湿度 80%",
    "深圳": "雷阵雨,29°C,南风 3 级,湿度 78%",
    "杭州": "多云,27°C,东风 2 级,湿度 60%",
    "成都": "阴,24°C,微风,湿度 70%",
    "武汉": "晴转多云,31°C,北风 2 级,湿度 55%",
}


def get_weather(city: str = "北京") -> str:
    city = (city or "北京").strip()
    data = _WEATHER_TABLE.get(city)
    if not data:
        return f"{city}:暂无天气数据(演示数据源仅覆盖: {'、'.join(_WEATHER_TABLE)})"
    return f"{city} 当前天气:{data}(数据源:演示用模拟数据,可替换为和风天气 API)"


# ----------------------------------------------------------------------
def build_tools(pipeline) -> list[Tool]:
    """依赖注入 pipeline(知识库工具需要检索能力),返回工具列表。"""
    return [
        Tool(
            name="kb_search",
            description=(
                "在企业知识库中检索相关片段,返回带编号与 doc_id 的资料。"
                "当用户询问公司制度、报销、考勤、流程、产品、部署等内部信息时先调用它。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "检索问题或关键词"},
                },
                "required": ["query"],
            },
            func=lambda query: _kb_search(pipeline, query),
            max_output_chars=2400,
        ),
        Tool(
            name="kb_read",
            description=(
                "读取知识库中某篇文档的完整内容。当 kb_search 返回的片段不足以回答时,"
                "用片段里给出的 doc_id 调本工具展开全文。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "doc_id": {"type": "string", "description": "形如 doc-人事制度-费用报销流程"},
                },
                "required": ["doc_id"],
            },
            func=lambda doc_id: _kb_read(pipeline.store, doc_id),
            max_output_chars=3200,
        ),
        Tool(
            name="calculator",
            description="计算数学表达式,支持加减乘除、取模与括号,例如 (1200+300)*0.8。",
            parameters={
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "数学表达式"},
                },
                "required": ["expression"],
            },
            func=calculator,
            timeout_s=2.0,
        ),
        Tool(
            name="get_datetime",
            description="获取当前日期、时间与星期。",
            parameters={"type": "object", "properties": {}},
            func=get_datetime,
            timeout_s=2.0,
        ),
        Tool(
            name="get_weather",
            description="查询指定城市的当前天气(演示数据源,仅覆盖主要城市)。",
            parameters={
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名,如 上海"},
                },
                "required": ["city"],
            },
            func=get_weather,
            timeout_s=3.0,
        ),
    ]


__all__: list[Any] = ["Tool", "build_tools", "calculator", "get_datetime", "get_weather"]

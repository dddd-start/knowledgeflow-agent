"""工具注册表 —— 参数校验 + 超时 + 输出截断 + 异常隔离。

Agent 循环里最怕两件事:模型传错参数导致异常冒泡,或者工具卡死拖垮整个请求。
所以 `execute()` 承诺:永远返回 (text, is_error),永不抛异常,并且有硬超时。
这层"防呆"是简历里"失败自动降级"的具体实现。
"""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass

from app.agent.tools import Tool

logger = logging.getLogger(__name__)

_MAX_RESULT_CHARS = 4000


@dataclass
class ToolResult:
    name: str
    arguments: dict
    text: str
    is_error: bool
    ms: int

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "arguments": self.arguments,
            "output": self.text[:600],
            "is_error": self.is_error,
            "ms": self.ms,
        }


class ToolRegistry:
    def __init__(self, tools: list[Tool], max_workers: int = 4) -> None:
        self._tools = {tool.name: tool for tool in tools}
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="tool")

    # ------------------------------------------------------------------
    def names(self) -> list[str]:
        return list(self._tools)

    def specs(self) -> list[dict]:
        return [tool.spec() for tool in self._tools.values()]

    def has(self, name: str) -> bool:
        return name in self._tools

    # ------------------------------------------------------------------
    def execute(self, name: str, arguments: dict | None) -> ToolResult:
        started = time.perf_counter()
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(name, arguments or {}, f"未知工具: {name}", True, 0)

        args = arguments or {}
        if not isinstance(args, dict):
            return ToolResult(name, {}, f"参数必须是对象,收到: {type(args).__name__}", True, 0)

        error = _validate(tool.parameters, args)
        if error:
            return ToolResult(name, args, f"参数校验失败: {error}", True, 0)

        try:
            future = self._executor.submit(tool.func, **args)
            output = future.result(timeout=tool.timeout_s)
            text = str(output)
            is_error = False
        except FutureTimeout:
            return ToolResult(
                name,
                args,
                f"工具执行超时(>{tool.timeout_s}s),请简化输入或改用其他工具",
                True,
                int((time.perf_counter() - started) * 1000),
            )
        except TypeError as exc:
            text, is_error = f"参数错误: {exc}", True
        except Exception as exc:  # noqa: BLE001
            logger.warning("工具 %s 执行失败: %s", name, exc)
            text, is_error = f"工具执行失败: {type(exc).__name__}: {exc}", True

        if len(text) > min(tool.max_output_chars, _MAX_RESULT_CHARS):
            text = text[: min(tool.max_output_chars, _MAX_RESULT_CHARS)] + "\n…(输出已截断)"
        return ToolResult(name, args, text, is_error, int((time.perf_counter() - started) * 1000))

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)


def _validate(schema: dict, args: dict) -> str | None:
    """轻量 JSON Schema 校验:只校验 required / type / enum,不引 jsonschema 依赖。"""
    required = schema.get("required", [])
    for key in required:
        if key not in args:
            return f"缺少必填参数 {key}"
    properties = schema.get("properties", {})
    for key, value in args.items():
        spec = properties.get(key)
        if not spec:
            continue
        expected = spec.get("type")
        if expected == "string" and not isinstance(value, str):
            return f"参数 {key} 应为字符串"
        if expected == "number" and not isinstance(value, (int, float)):
            return f"参数 {key} 应为数字"
        if expected == "integer" and not isinstance(value, int):
            return f"参数 {key} 应为整数"
        if expected == "boolean" and not isinstance(value, bool):
            return f"参数 {key} 应为布尔值"
        if expected == "object" and not isinstance(value, dict):
            return f"参数 {key} 应为对象"
        if spec.get("enum") and value not in spec["enum"]:
            return f"参数 {key} 取值必须是 {spec['enum']} 之一"
    return None


def tool_calls_json(result: list[ToolResult]) -> str:
    return json.dumps([r.to_dict() for r in result], ensure_ascii=False)

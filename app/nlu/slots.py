"""槽位抽取 —— 意图分类只回答「做什么」,槽位回答「参数是什么」。

经典 NLU = intent + slot 两步。这里用规则抽取城市与算式:
规则槽位在"意图明确、槽位格式固定"的场景比模型更可靠、更便宜;
抽不到就返回 None,由路由层回退到 Agent(复杂问题永远有兜底)。
"""

from __future__ import annotations

import re

_CITIES = [
    "北京",
    "上海",
    "广州",
    "深圳",
    "成都",
    "杭州",
    "重庆",
    "武汉",
    "西安",
    "南京",
    "天津",
    "苏州",
    "长沙",
    "青岛",
    "大连",
    "厦门",
    "昆明",
    "哈尔滨",
    "三亚",
    "郑州",
    "济南",
    "合肥",
    "福州",
    "南昌",
    "贵阳",
    "南宁",
    "兰州",
    "沈阳",
    "长春",
    "海口",
]
_CITY_RE = re.compile("|".join(sorted(_CITIES, key=len, reverse=True)))

_CN_OPS = {
    "乘以": "*",
    "除以": "/",
    "加上": "+",
    "减去": "-",
    "加": "+",
    "减": "-",
    "乘": "*",
    "除": "/",
    "×": "*",
    "÷": "/",
}
_EXPR_RE = re.compile(r"[0-9.()\s+\-*/]+")


def extract_city(text: str) -> str | None:
    match = _CITY_RE.search(text or "")
    return match.group(0) if match else None


def extract_expression(text: str) -> str | None:
    """从「帮我算一下 (1200+300)*0.8 等于多少」里抽出算式。"""
    normalized = text or ""
    for cn, symbol in sorted(_CN_OPS.items(), key=lambda kv: len(kv[0]), reverse=True):
        normalized = normalized.replace(cn, symbol)
    match = _EXPR_RE.search(normalized)
    if not match:
        return None
    expr = match.group(0).strip()
    return expr if re.search(r"\d", expr) else None

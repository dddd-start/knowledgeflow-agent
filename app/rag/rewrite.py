"""查询改写 + HyDE —— 多级检索优化的前两级。

- 查询改写(rewrite):解决「多轮对话里的指代」和「口语化表达」问题,
  例:"那它要多久" → "差旅报销审批要多久";
- HyDE:让 LLM 先写一段"假设答案文档",用它的向量去检索,
  拉近 query 与真实文档的向量距离,对中文短查询尤其有效。

两个函数都做了失败兜底:LLM 不可用/输出离谱时退回原查询,
绝不因为"优化步骤"把主链路搞挂。
"""

from __future__ import annotations

import logging
import re

from app.prompts.manager import get_prompt_manager

logger = logging.getLogger(__name__)

_QUOTES = "\"'“”‘’「」"


def _clean_query(text: str, fallback: str) -> str:
    line = (text or "").strip().splitlines()
    if not line:
        return fallback
    cleaned = line[0].strip().strip(_QUOTES).strip()
    cleaned = re.sub(r"^(改写后[:：]?|查询[:：])\s*", "", cleaned)
    # 防御:模型跑偏输出长段落时,认为改写失败
    if not cleaned or len(cleaned) > 120:
        return fallback
    return cleaned


def rewrite_query(llm, query: str, history: list[dict] | None = None) -> str:
    if llm is None:
        return query
    try:
        prompt = get_prompt_manager().render("rewrite", query=query, history=history or [])
        # temperature=0:检索优化要可复现(同输入同输出),也便于后续做语义缓存
        text = llm.generate_text(prompt, max_tokens=128, temperature=0.0, purpose="rewrite")
        return _clean_query(text, query)
    except Exception as exc:  # noqa: BLE001
        logger.warning("查询改写失败,使用原查询: %s", exc)
        return query


def hyde(llm, query: str) -> str:
    if llm is None:
        return ""
    try:
        prompt = get_prompt_manager().render("hyde", query=query)
        text = llm.generate_text(prompt, max_tokens=256, temperature=0.0, purpose="hyde")
        text = (text or "").strip()
        return text if 10 <= len(text) <= 800 else ""
    except Exception as exc:  # noqa: BLE001
        logger.warning("HyDE 生成失败,跳过: %s", exc)
        return ""

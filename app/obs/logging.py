"""日志 —— 统一格式(时间 / 级别 / 模块 / 消息),支持 LOG_LEVEL 环境变量。"""

from __future__ import annotations

import logging
import sys

from app.config import settings

_CONFIGURED = False


def setup_logging(level: str | None = None) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-7s | %(name)-28s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(getattr(logging, (level or settings.log_level).upper(), logging.INFO))
    # 降噪:三方库的 INFO 太多
    for noisy in (
        "httpx",
        "httpx2",
        "httpcore",
        "sentence_transformers",
        "urllib3",
        "jieba",
        "huggingface_hub",
        "filelock",
        "transformers",
        "asyncio",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _CONFIGURED = True

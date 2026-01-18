"""长期记忆 —— 用户偏好事实存 SQLite,跨会话/跨进程复用。

与短期记忆的分工:
- 短期:原始对话轮次 + 摘要,只服务当前会话;
- 长期:LLM 抽取出的**稳定偏好事实**(部门、常驻城市、回答风格),
  key-value 存储,永远很小,重启不丢。

为什么抽"偏好"而不是把历史全存进向量库?
    企业知识库助手的跨会话价值在于"记住你是谁/你关心什么",
    而不是记住三周前问过的某个具体问题;前者是偏好,后者是噪音。
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from app.config import settings

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS preferences (
    user_id    TEXT NOT NULL,
    key        TEXT NOT NULL,
    value      TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 1.0,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (user_id, key)
);
"""


class LongTermMemory:
    def __init__(self, db_path: Path | str | None = None) -> None:
        base = Path(db_path) if db_path else settings.data_dir / "memory"
        base.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = base
        self._conn = sqlite3.connect(str(base), check_same_thread=False)
        self._conn.execute(_SCHEMA)
        self._conn.commit()

    # ------------------------------------------------------------------
    def add(self, user_id: str, key: str, value: str) -> None:
        now = datetime.now(tz=UTC).astimezone().isoformat(timespec="seconds")
        self._conn.execute(
            """INSERT INTO preferences(user_id, key, value, updated_at)
               VALUES(?,?,?,?)
               ON CONFLICT(user_id, key) DO UPDATE SET
                 value=excluded.value, updated_at=excluded.updated_at""",
            (user_id, key, value, now),
        )
        self._conn.commit()

    def load(self, user_id: str) -> list[str]:
        rows = self._conn.execute(
            "SELECT key, value FROM preferences WHERE user_id=? ORDER BY key",
            (user_id,),
        ).fetchall()
        return [f"{k}: {v}" for k, v in rows]

    def delete_user(self, user_id: str) -> None:
        self._conn.execute("DELETE FROM preferences WHERE user_id=?", (user_id,))
        self._conn.commit()

    # ------------------------------------------------------------------
    def extract_and_store(self, llm, user_id: str, conversation: str) -> list[str]:
        if not conversation.strip():
            return []
        try:
            from app.prompts.manager import get_prompt_manager

            prompt = get_prompt_manager().render("preferences", conversation=conversation)
            raw = llm.generate_text(prompt, max_tokens=256, purpose="preferences")
            facts = _parse_json_array(raw)
        except Exception as exc:  # noqa: BLE001
            logger.warning("偏好抽取失败: %s", exc)
            return []
        added = []
        for fact in facts:
            key = str(fact.get("key", "")).strip()
            value = str(fact.get("value", "")).strip()
            if not key or not value or len(key) > 20 or len(value) > 80:
                continue
            self.add(user_id, key, value)
            added.append(f"{key}: {value}")
        return added

    def close(self) -> None:
        self._conn.close()


def _parse_json_array(raw: str) -> list[dict]:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", (raw or "").strip(), flags=re.DOTALL)
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    return [item for item in data if isinstance(item, dict)] if isinstance(data, list) else []

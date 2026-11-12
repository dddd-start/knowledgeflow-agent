"""请求级链路追踪 —— 把"这次问答慢在哪"变成可观测数据。

每个请求生成一个 trace,记录:模式、各阶段耗时、LLM 调用次数与 token、命中的证据、
自检结论、总耗时。它同时被三个地方消费:
- Web 前端:渲染"检索链路"时间轴(面试演示的亮点);
- /metrics:汇总成 p50/p95、token、成本;
- 评测脚本:离线跑批时导出 JSON,供 docs/EVALUATION.md 复现。
"""

from __future__ import annotations

import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field

from app.llm.base import Usage


@dataclass
class StageRecord:
    name: str
    ms: int
    info: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"name": self.name, "ms": self.ms, "info": self.info}


@dataclass
class RequestTrace:
    mode: str
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    started: float = field(default_factory=time.perf_counter)
    stages: list[StageRecord] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    llm_calls: int = 0
    hit_count: int = 0
    self_check: str | None = None
    path: str = ""

    def stage(self, name: str, ms: int, **info) -> None:
        self.stages.append(StageRecord(name=name, ms=int(ms), info=info))

    @contextmanager
    def timed(self, name: str, **info):
        started = time.perf_counter()
        box: dict = {}
        try:
            yield box
        finally:
            payload = {**info, **box}
            self.stage(name, int((time.perf_counter() - started) * 1000), **payload)

    @property
    def elapsed_ms(self) -> int:
        return int((time.perf_counter() - self.started) * 1000)

    def to_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "mode": self.mode,
            "path": self.path,
            "elapsed_ms": self.elapsed_ms,
            "llm_calls": self.llm_calls,
            "usage": self.usage.to_dict(),
            "hit_count": self.hit_count,
            "self_check": self.self_check,
            "stages": [s.to_dict() for s in self.stages],
        }

"""RAG 数据结构 —— 文档 / 分块 / 命中 / 检索链路轨迹。

单独放一个模块,是因为这些结构同时被检索、Agent 工具、API、评测脚本消费,
必须稳定且可序列化(前端要能看到完整的检索过程)。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Document:
    """一篇原始文档(入库前的形态)。"""

    doc_id: str
    title: str
    category: str
    text: str
    source: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def chars(self) -> int:
        return len(self.text)


@dataclass
class Chunk:
    """一个可检索的文本块。metadata 只放标量,便于过滤与持久化。"""

    chunk_id: str
    doc_id: str
    text: str
    metadata: dict = field(default_factory=dict)

    @property
    def title(self) -> str:
        return str(self.metadata.get("title", ""))

    @property
    def category(self) -> str:
        return str(self.metadata.get("category", ""))


@dataclass
class Hit:
    """一次检索命中的片段(带通道与分数,便于解释"为什么它排第一")。"""

    chunk_id: str
    doc_id: str
    title: str
    text: str
    score: float = 0.0
    channel: str = ""  # vector | bm25 | fused | rerank
    rank: int = 0
    detail: dict = field(default_factory=dict)

    def to_dict(self, *, text_limit: int = 400) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "title": self.title,
            "text": self.text[:text_limit],
            "score": round(self.score, 4),
            "channel": self.channel,
            "rank": self.rank,
            "detail": self.detail,
        }


@dataclass
class Stage:
    """检索链路中的一个阶段(耗时 + 过程信息),前端按时间轴展示。"""

    name: str
    ms: int = 0
    info: dict = field(default_factory=dict)
    hits: list[Hit] = field(default_factory=list)

    def to_dict(self, *, text_limit: int = 240) -> dict:
        return {
            "name": self.name,
            "ms": self.ms,
            "info": self.info,
            "hits": [h.to_dict(text_limit=text_limit) for h in self.hits[:5]],
        }


@dataclass
class RetrievalTrace:
    """一次检索的完整轨迹:查询改写 → 双路召回 → 融合 → 重排。"""

    query: str
    mode: str = "fast"
    stages: list[Stage] = field(default_factory=list)
    hits: list[Hit] = field(default_factory=list)
    total_ms: int = 0

    def add(self, stage: Stage) -> None:
        self.stages.append(stage)

    def to_dict(self, *, text_limit: int = 240) -> dict:
        return {
            "query": self.query,
            "mode": self.mode,
            "total_ms": self.total_ms,
            "stages": [s.to_dict(text_limit=text_limit) for s in self.stages],
            "candidates": sum(len(s.hits) for s in self.stages),
        }

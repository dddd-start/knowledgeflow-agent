"""服务编排层 —— 把「意图路由 / 检索 / Agent / Agentic / 自检 / 记忆」串成一条流。

这一层是 HTTP API 与 CLI 的共同后端,也是面试时讲"一次问答到底发生了什么"的地方:

    message
      → 记忆维护(压缩 / 话题检测)
      → 意图路由(0 次 or 1 次 LLM?)
        ├─ fast  : 直接执行确定性工具,0 次 LLM
        ├─ chat  : 闲聊,1 次 LLM
        ├─ rag   : 检索(fast 2 路召回 / quality 改写+HyDE+重排)→ 生成 → 自检
        ├─ agent : Function Calling 循环(≤3 轮)
        └─ agentic: 迭代检索,直到证据充分
      → 记忆写回 + 偏好抽取
      → 事件流(meta / retrieval / delta / citations / usage / done)

对"事件流"而不是"一个字符串":Web 端可以边生成边渲染,
非流式调用方把事件收集起来就是完整 JSON,一套逻辑两种用法。
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Iterator
from dataclasses import dataclass, field

from app.agent.loop import AgentLoop
from app.agent.registry import ToolRegistry
from app.agent.tools import build_tools
from app.agentic.agentic_rag import AgenticRAG
from app.config import Settings, settings
from app.llm.base import ChatResult, StreamEvent, Usage
from app.llm.factory import get_llm
from app.memory.memory_manager import MemoryManager
from app.nlu.router import IntentRouter, get_router
from app.obs.metrics import metrics
from app.obs.trace import RequestTrace
from app.prompts.manager import get_prompt_manager
from app.quality.self_correction import SelfChecker
from app.rag.pipeline import RetrievalPipeline
from app.rag.schemas import Hit

logger = logging.getLogger(__name__)

MODE_TO_PATH = {
    "auto": "auto",
    "fast": "rag",
    "quality": "rag",
    "chat": "chat",
    "agent": "agent",
    "agentic": "agentic",
}


@dataclass
class Event:
    """流式事件。type 决定前端如何渲染。"""

    type: str
    data: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"type": self.type, **self.data}


class MeteredLLM:
    """给 LLM 加计量:每个请求统计调用次数、token 与用途分布。

    以代理方式注入,pipeline / agent / checker 都不需要知道计量的存在。
    只读共享连接,不负责 close。
    """

    def __init__(self, inner) -> None:
        self.inner = inner
        self.usage = Usage()
        self.calls = 0
        self.by_purpose: dict[str, int] = {}

    @property
    def backend(self) -> str:
        return getattr(self.inner, "backend", "unknown")

    @property
    def model(self) -> str:
        return getattr(self.inner, "model", "unknown")

    def _record(self, purpose: str, usage: Usage) -> None:
        self.calls += 1
        self.by_purpose[purpose] = self.by_purpose.get(purpose, 0) + 1
        self.usage = self.usage + usage

    def chat(self, *args, **kwargs) -> ChatResult:
        purpose = kwargs.get("purpose", "answer")
        result = self.inner.chat(*args, **kwargs)
        self._record(purpose, result.usage)
        return result

    def generate_text(self, prompt: str, **kwargs) -> str:
        purpose = kwargs.get("purpose", "answer")
        result = self.inner.chat(
            [{"role": "user", "content": prompt}],
            system=kwargs.get("system"),
            max_tokens=kwargs.get("max_tokens", 512),
            temperature=kwargs.get("temperature", 0.2),
            thinking=kwargs.get("thinking", False),
            purpose=purpose,
        )
        self._record(purpose, result.usage)
        return result.text.strip()

    def stream(self, *args, **kwargs) -> Iterator[StreamEvent]:
        purpose = kwargs.get("purpose", "answer")
        for event in self.inner.stream(*args, **kwargs):
            if event.done and event.usage:
                self._record(purpose, event.usage)
            yield event

    def close(self) -> None:  # 共享实例,不在这里关闭
        return None


class ChatService:
    def __init__(
        self,
        cfg: Settings | None = None,
        llm=None,
        pipeline: RetrievalPipeline | None = None,
        router: IntentRouter | None = None,
    ) -> None:
        self.cfg = cfg or settings
        self.llm = llm or get_llm(self.cfg)
        self.pipeline = pipeline or RetrievalPipeline.build(self.llm, self.cfg)
        self.registry = ToolRegistry(build_tools(self.pipeline))
        self.checker = SelfChecker(self.llm)
        self.router = router or get_router()
        self._memories: dict[str, MemoryManager] = {}

    # ------------------------------------------------------------------ 元信息
    def memory(self, session_id: str, user_id: str) -> MemoryManager:
        key = f"{user_id}::{session_id}"
        if key not in self._memories:
            self._memories[key] = MemoryManager(session_id=session_id, user_id=user_id)
            if len(self._memories) > 200:  # 防止字典无限增长
                self._memories.pop(next(iter(self._memories)))
        return self._memories[key]

    def ensure_index_ready(self) -> None:
        """索引与当前 embedding 模型不匹配时自动重建(换模型/无 ML 依赖场景)。

        典型场景:面试官机器没装 ML 依赖 -> 哈希向量;而你本地用 bge 建过索引。
        维度不一致会让检索直接报错,这里自动检测并用 data/raw 重建,保证开箱可用。
        """
        store = self.pipeline.store
        vectors = getattr(store, "_vectors", None)
        if store.count() == 0 or vectors is None or vectors.size == 0:
            return
        try:
            current_dim = self.pipeline.embedder.dim
        except Exception:  # noqa: BLE001
            return
        if vectors.shape[1] == current_dim:
            return
        logger.warning(
            "索引维度 %d 与当前 embedding 维度 %d 不一致,自动重建索引",
            vectors.shape[1],
            current_dim,
        )
        from app.rag.ingest import load_documents

        docs = load_documents(self.cfg.raw_dir)
        if docs:
            self.pipeline.ingest(docs, strategy="recursive", reset=True)

    def warmup(self) -> dict:
        """预热:embedding 模型、jieba 分词、重排模型。

        不预热的话第一个请求要额外承担 2~10s 的模型加载,演示时很尴尬。
        """
        from app.rag.schemas import Hit

        timings: dict[str, int] = {}
        t0 = time.perf_counter()
        try:
            self.pipeline.embedder.encode(["预热"])
            timings["embedding"] = int((time.perf_counter() - t0) * 1000)
        except Exception as exc:  # noqa: BLE001
            logger.warning("embedding 预热失败: %s", exc)
        if self.pipeline.bm25 is not None:
            t0 = time.perf_counter()
            self.pipeline.bm25.search("预热")
            timings["bm25"] = int((time.perf_counter() - t0) * 1000)
        if self.pipeline.reranker.enabled:
            t0 = time.perf_counter()
            try:
                dummy = Hit(chunk_id="warmup", doc_id="warmup", title="预热", text="预热文本")
                self.pipeline.reranker.rerank("预热", [dummy])
                timings["rerank"] = int((time.perf_counter() - t0) * 1000)
            except Exception as exc:  # noqa: BLE001
                logger.warning("重排预热失败: %s", exc)
        logger.info("预热完成: %s", timings)
        return timings

    def meta(self) -> dict:
        describe = self.cfg.describe()
        return {
            "app": describe["app"],
            "llm": describe["llm"],
            "embedding": describe["embedding"],
            "vector_store": describe["vector_store"],
            "rerank": describe["rerank"],
            "intent": {
                **describe["intent"],
                "backend": getattr(self.router.classifier, "backend", "rule"),
            },
            "prompts": {
                "answer": get_prompt_manager().get_version("answer"),
                "rewrite": get_prompt_manager().get_version("rewrite"),
                "self_check": get_prompt_manager().get_version("self_check"),
            },
            "kb": self.pipeline.stats(),
            "offline_demo": self.cfg.offline_demo,
            "modes": ["auto", "fast", "quality", "agent", "agentic"],
        }

    # ------------------------------------------------------------------ 主入口
    def answer(
        self,
        message: str,
        *,
        session_id: str = "default",
        user_id: str = "default",
        mode: str = "auto",
        stream: bool = True,
    ) -> Iterator[Event]:
        message = (message or "").strip()
        if not message:
            yield Event("error", {"message": "问题不能为空"})
            return

        started = time.perf_counter()
        mode = mode if mode in MODE_TO_PATH else "auto"
        trace = RequestTrace(mode)
        meter = MeteredLLM(self.llm)
        state: dict = {"answer": "", "citations": [], "hits": [], "tools": [], "self_check": None}
        error = False

        mem = self.memory(session_id, user_id)
        memory_events: dict = {}
        try:
            memory_events = mem.before_user_turn(message, meter)
        except Exception as exc:  # noqa: BLE001
            logger.debug("记忆维护失败: %s", exc)

        plan = self.router.plan(message)
        path = plan.path if mode == "auto" else MODE_TO_PATH[mode]
        retrieval_mode = (
            "quality" if mode == "quality" or (mode == "auto" and path == "rag") else "fast"
        )
        trace.path = path

        yield Event(
            "meta",
            {
                "request_id": trace.request_id,
                "mode": mode,
                "path": path,
                "plan": plan.to_dict(),
                "llm": {
                    "backend": meter.backend,
                    "model": meter.model,
                    "degraded": bool(getattr(self.llm, "degraded", False)),
                },
                "kb_chunks": self.pipeline.store.count(),
                "memory": {
                    "turns": len(mem.short.turns) // 2,
                    "summary": mem.short.summary,
                    "estimated_tokens": mem.short.estimated_tokens(),
                },
            },
        )
        if memory_events:
            yield Event("memory", memory_events)

        try:
            if path == "fast":
                yield from self._run_fast(message, state, trace)
            elif path == "chat":
                yield from self._run_chat(message, mem, meter, trace, state, stream)
            elif path == "agent":
                yield from self._run_agent(message, mem, meter, trace, state)
            elif path == "agentic":
                yield from self._run_agentic(message, mem, meter, trace, state)
            else:
                yield from self._run_rag(message, mem, meter, trace, state, retrieval_mode, stream)
        except Exception as exc:  # noqa: BLE001
            error = True
            logger.exception("问答失败")
            yield Event("error", {"message": f"服务异常: {type(exc).__name__}: {exc}"})

        # ---------------- 收尾:记忆、指标、用量 ----------------
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        trace.usage = meter.usage
        trace.llm_calls = meter.calls
        answer_text = state["answer"] or ""

        if answer_text and not error:
            try:
                mem.record_turn(message, answer_text)
                turns = len(mem.short.turns) // 2
                if turns and turns % 3 == 0:
                    conversation = "\n".join(
                        f"[{m['role']}] {m['content']}" for m in mem.history()[-6:]
                    )
                    mem.extract_preferences(meter, conversation)
            except Exception as exc:  # noqa: BLE001
                logger.debug("记忆写回失败: %s", exc)

        cost = self.cfg.cost_usd(meter.usage.prompt_tokens, meter.usage.completion_tokens)
        metrics.record_request(
            mode=mode,
            path=path,
            elapsed_ms=elapsed_ms,
            prompt_tokens=meter.usage.prompt_tokens,
            completion_tokens=meter.usage.completion_tokens,
            cost_usd=cost,
            llm_calls=meter.calls,
            error=error,
        )
        metrics.observe("stage_llm_calls", meter.calls * 1000)

        yield Event(
            "usage",
            {
                "prompt_tokens": meter.usage.prompt_tokens,
                "completion_tokens": meter.usage.completion_tokens,
                "total_tokens": meter.usage.total_tokens,
                "cost_usd": round(cost, 6),
                "llm_calls": meter.calls,
                "by_purpose": meter.by_purpose,
                "elapsed_ms": elapsed_ms,
            },
        )
        yield Event(
            "done",
            {
                "answer": answer_text,
                "request_id": trace.request_id,
                "path": path,
                "mode": mode,
                "self_check": state["self_check"],
                "elapsed_ms": elapsed_ms,
                "trace": trace.to_dict(),
            },
        )

    # ------------------------------------------------------------------ 各路径
    def _run_fast(self, message: str, state: dict, trace: RequestTrace) -> Iterator[Event]:
        """确定性工具直达:0 次 LLM。"""
        with trace.timed("fast_tool"):
            fast = self.router.fast_answer(message, self.registry)
        if fast is None:  # 槽位抽取失败:降级为闲聊
            state["answer"] = "我没能识别出具体的查询参数,可以再补充一下城市、算式或时间点。"
            yield Event("delta", {"text": state["answer"]})
            yield Event("tools", {"items": []})
            return
        state["answer"] = fast.answer
        state["tools"] = [tc.to_dict() for tc in fast.tool_calls]
        yield Event("delta", {"text": fast.answer})
        yield Event("tools", {"items": state["tools"]})

    def _run_chat(
        self,
        message: str,
        mem: MemoryManager,
        meter: MeteredLLM,
        trace: RequestTrace,
        state: dict,
        stream: bool,
    ) -> Iterator[Event]:
        system = get_prompt_manager().render("chat")
        messages = mem.history() + [{"role": "user", "content": message}]
        if stream:
            with trace.timed("generate"):
                for event in meter.stream(
                    messages, system=system, max_tokens=800, purpose="answer"
                ):
                    if event.delta:
                        state["answer"] += event.delta
                        yield Event("delta", {"text": event.delta})
        else:
            with trace.timed("generate"):
                result = meter.chat(
                    messages, system=system, max_tokens=800, temperature=0.5, purpose="answer"
                )
            state["answer"] = result.text.strip()
            yield Event("delta", {"text": state["answer"]})

    def _run_rag(
        self,
        message: str,
        mem: MemoryManager,
        meter: MeteredLLM,
        trace: RequestTrace,
        state: dict,
        retrieval_mode: str,
        stream: bool,
    ) -> Iterator[Event]:
        started = time.perf_counter()
        hits, rtrace = self.pipeline.retrieve(
            message, mode=retrieval_mode, history=mem.history(), llm=meter
        )
        trace.stage("retrieve", int((time.perf_counter() - started) * 1000), mode=retrieval_mode)
        state["hits"] = hits
        trace.hit_count = len(hits)
        yield Event("retrieval", {"trace": rtrace.to_dict(text_limit=500)})

        if not hits:
            state["answer"] = (
                "知识库中暂时没有检索到相关内容。"
                "可能是文档还没入库,或这个说法在资料里没有对应表述,可以换一种问法。"
            )
            yield Event("delta", {"text": state["answer"]})
            yield Event("citations", {"items": []})
            return

        context = self.pipeline.format_context(hits)
        system = get_prompt_manager().render("answer", context=context, **mem.system_extra())
        messages = mem.history() + [{"role": "user", "content": message}]

        if stream:
            with trace.timed("generate"):
                for event in meter.stream(
                    messages, system=system, max_tokens=1200, purpose="answer"
                ):
                    if event.delta:
                        state["answer"] += event.delta
                        yield Event("delta", {"text": event.delta})
        else:
            with trace.timed("generate"):
                result = meter.chat(messages, system=system, max_tokens=1200, purpose="answer")
            state["answer"] = result.text.strip()
            yield Event("delta", {"text": state["answer"]})

        # ---- 幻觉自检(仅高质量模式,避免给快路径增加一次 LLM 调用)----
        if self.cfg.self_check_enabled and retrieval_mode == "quality" and state["answer"]:
            with trace.timed("self_check"):
                # 用请求级 meter 包一层,自检的 token 也要算进成本
                final, check = SelfChecker(meter).apply(message, state["answer"], hits)
            state["self_check"] = check.to_dict()
            trace.self_check = check.verdict
            if final != state["answer"]:
                state["answer"] = final
                yield Event("replace", {"text": final, "reason": check.reason})
            yield Event("self_check", check.to_dict())

        state["citations"] = _citations(hits)
        yield Event("citations", {"items": state["citations"]})

    def _run_agent(
        self, message: str, mem: MemoryManager, meter: MeteredLLM, trace: RequestTrace, state: dict
    ) -> Iterator[Event]:
        system = get_prompt_manager().render(
            "agent_system",
            tools=self.registry.specs(),
            max_rounds=self.cfg.max_agent_rounds,
            **mem.system_extra(),
        )
        loop = AgentLoop(
            meter, self.registry, system_prompt=system, max_rounds=self.cfg.max_agent_rounds
        )
        with trace.timed("agent_loop"):
            run = loop.run(message, history=mem.history())
        state["answer"] = run.answer
        state["tools"] = [tc.to_dict() for tc in run.tool_calls]
        yield Event("tools", {"items": state["tools"]})
        yield Event("delta", {"text": run.answer})
        state["citations"] = _citations_from_tools(run.tool_calls)
        yield Event("citations", {"items": state["citations"]})

    def _run_agentic(
        self, message: str, mem: MemoryManager, meter: MeteredLLM, trace: RequestTrace, state: dict
    ) -> Iterator[Event]:
        rag = AgenticRAG(meter, self.pipeline, max_rounds=self.cfg.agentic_max_rounds)
        with trace.timed("agentic_rag"):
            result = rag.run(message)
        state["answer"] = result.answer
        state["hits"] = result.hits
        yield Event(
            "retrieval",
            {
                "trace": {
                    "query": message,
                    "mode": "agentic",
                    "total_ms": sum(r.ms for r in result.rounds),
                    "rounds": [r.to_dict() for r in result.rounds],
                    "stages": [
                        {
                            "name": f"第{i}轮检索",
                            "ms": r.ms,
                            "info": {
                                "query": r.query,
                                "sufficient": r.sufficient,
                                "missing": r.missing,
                            },
                            "hits": [h.to_dict(text_limit=200) for h in r.hits[:5]],
                        }
                        for i, r in enumerate(result.rounds, start=1)
                    ],
                    "candidates": sum(len(r.hits) for r in result.rounds),
                }
            },
        )
        yield Event("delta", {"text": result.answer})
        state["citations"] = _citations(result.hits)
        yield Event("citations", {"items": state["citations"]})


# ----------------------------------------------------------------------
_CITATION_RE = re.compile(r"\[(\d+)\]\s*《([^》]+)》\s*\(doc_id=([^,]+),\s*相关度=([\d.]+)")


def _citations(hits: list[Hit]) -> list[dict]:
    return [
        {
            "index": i,
            "doc_id": hit.doc_id,
            "title": hit.title,
            "score": round(hit.score, 4),
            "channel": hit.channel,
            "text": hit.text[:500],
        }
        for i, hit in enumerate(hits, start=1)
    ]


def _citations_from_tools(tool_calls) -> list[dict]:
    """从 Agent 的工具输出里解析引用(知识库工具的输出格式是稳定的)。"""
    items: dict[str, dict] = {}
    for call in tool_calls:
        if call.name not in {"kb_search", "kb_read"}:
            continue
        for match in _CITATION_RE.finditer(call.text):
            index, title, doc_id, score = match.groups()
            items.setdefault(
                doc_id.strip(),
                {
                    "index": int(index),
                    "doc_id": doc_id.strip(),
                    "title": title,
                    "score": float(score),
                    "channel": "agent_tool",
                    "text": "",
                },
            )
    return list(items.values())


def collect_events(events: Iterator[Event]) -> dict:
    """把事件流收敛成一次性响应(非流式 API / CLI / 测试用)。"""
    result: dict = {
        "answer": "",
        "citations": [],
        "tools": [],
        "retrieval": None,
        "usage": {},
        "meta": {},
        "error": None,
        "self_check": None,
    }
    for event in events:
        if event.type == "delta":
            result["answer"] += event.data.get("text", "")
        elif event.type == "replace":
            result["answer"] = event.data.get("text", "")
        elif event.type == "citations":
            result["citations"] = event.data.get("items", [])
        elif event.type == "tools":
            result["tools"] = event.data.get("items", [])
        elif event.type == "retrieval":
            result["retrieval"] = event.data.get("trace")
        elif event.type == "usage":
            result["usage"] = event.data
        elif event.type == "meta":
            result["meta"] = event.data
        elif event.type == "self_check":
            result["self_check"] = event.data
        elif event.type == "error":
            result["error"] = event.data
        elif event.type == "done":
            result["answer"] = event.data.get("answer", result["answer"])
    return result

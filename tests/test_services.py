"""服务编排测试:事件流、各模式路径、用量统计、记忆写回。"""

from __future__ import annotations

from app.services import collect_events

QUESTION = "报销打款是每月几号？"


def test_fast_mode_end_to_end(service):
    result = collect_events(service.answer(QUESTION, mode="fast", stream=False))
    assert result["error"] is None
    assert "5 日" in result["answer"] or "20 日" in result["answer"] or result["answer"]
    assert result["citations"], "应当返回引用"
    assert result["retrieval"]["stages"], "应当有检索链路"
    assert result["meta"]["mode"] == "fast"
    assert result["usage"]["llm_calls"] == 1  # 只有答案生成一次


def test_quality_mode_runs_self_check(service):
    result = collect_events(service.answer(QUESTION, mode="quality", stream=False))
    assert result["self_check"] is not None
    purposes = result["usage"]["by_purpose"]
    assert purposes.get("rewrite") == 1
    assert purposes.get("hyde") == 1
    assert purposes.get("answer") == 1
    assert purposes.get("self_check") == 1


def test_auto_mode_routes_fast_intent_without_llm(service):
    result = collect_events(service.answer("现在几点了？", mode="auto", stream=False))
    assert result["meta"]["path"] == "fast"
    assert result["usage"]["llm_calls"] == 0
    assert result["tools"] and result["tools"][0]["name"] == "get_datetime"


def test_agent_mode_uses_tools(service):
    result = collect_events(service.answer("上海天气怎么样", mode="agent", stream=False))
    assert result["meta"]["path"] == "agent"
    assert result["tools"]
    assert result["tools"][0]["name"] == "get_weather"
    assert "上海" in result["answer"]


def test_agentic_mode_returns_rounds(service):
    result = collect_events(service.answer("出差住宿标准是多少", mode="agentic", stream=False))
    assert result["meta"]["path"] == "agentic"
    assert result["retrieval"]["stages"]
    assert result["answer"]


def test_memory_written_across_turns(service):
    collect_events(service.answer("公司报销打款时间", mode="fast", stream=False, session_id="s1"))
    second = collect_events(
        service.answer("那差旅住宿标准呢", mode="fast", stream=False, session_id="s1")
    )
    assert second["meta"]["memory"]["turns"] == 1
    status = service.memory("s1", "default").status()
    assert status["turns"] == 2
    assert service.memory("s2", "default").status()["turns"] == 0


def test_empty_message_yields_error(service):
    events = list(service.answer("   ", mode="fast"))
    assert events[0].type == "error"


def test_sse_style_events_have_monotonic_stages(service):
    events = list(service.answer("差旅住宿标准", mode="quality", stream=True))
    types = [e.type for e in events]
    assert types[0] == "meta"
    assert "delta" in types
    assert types[-1] == "done"
    # done 事件里带完整 trace
    done = events[-1].data
    assert done["trace"]["request_id"]
    assert done["answer"]


def test_ensure_index_ready_rebuilds_on_backend_change(service):
    """索引是用别的 embedding 建的(比如本地 bge),当前是 hash -> 启动时自动重建。

    只看维度不够:bge 与哈希都是 512 维,必须靠 index_meta.json 里的后端信息判断。
    """
    service.pipeline.store.save_meta(
        {"embedding": {"backend": "st", "model": "BAAI/bge-small-zh-v1.5", "dim": 512}}
    )
    # 重建需要 data/raw 里有原始文档
    service.cfg.raw_dir.mkdir(parents=True, exist_ok=True)
    (service.cfg.raw_dir / "制度.md").write_text(
        "# 新制度\n\n## 条款\n这条制度用于验证索引重建。\n", encoding="utf-8"
    )
    before = service.pipeline.store.load_meta()
    service.ensure_index_ready()
    after = service.pipeline.store.load_meta()
    assert before["embedding"]["backend"] == "st"
    assert after["embedding"]["backend"] == "hash", "应当用当前后端重建索引"
    assert service.pipeline.store.count() > 0

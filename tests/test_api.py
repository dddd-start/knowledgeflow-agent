"""HTTP API 测试(TestClient + 注入离线服务,不触网)。"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.api.main import create_app


@pytest.fixture()
def client(service):
    app = create_app(service_factory=lambda: service)
    with TestClient(app) as test_client:
        yield test_client


def test_health_and_meta(client):
    health = client.get("/api/health").json()
    assert health["status"] == "ok"
    assert health["kb_chunks"] > 0
    meta = client.get("/api/meta").json()
    assert meta["llm"]["backend"] == "mock"
    assert meta["embedding"]["backend"] == "hash"
    assert "auto" in meta["modes"]
    assert meta["kb"]["documents"] == 4


def test_kb_documents_and_detail(client):
    data = client.get("/api/kb/documents").json()
    assert data["total"] == 4
    assert data["chunks"] > 0
    doc_id = data["documents"][0]["doc_id"]
    detail = client.get(f"/api/kb/documents/{doc_id}").json()
    assert detail["chunks"]
    assert client.get("/api/kb/documents/doc-not-exist").status_code == 404


def test_chat_json_mode(client):
    resp = client.post(
        "/api/chat",
        json={"message": "报销打款是每月几号？", "mode": "fast", "stream": False},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"]
    assert data["citations"]
    assert data["usage"]["llm_calls"] >= 1


def test_chat_sse_mode(client):
    resp = client.post(
        "/api/chat",
        json={"message": "差旅住宿标准", "mode": "fast", "stream": True},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    body = resp.text
    assert "event: meta" in body
    assert "event: delta" in body
    assert "event: done" in body
    # 每条事件都是 data: {json}
    for block in body.strip().split("\n\n"):
        lines = block.splitlines()
        assert lines[0].startswith("event: ")
        assert lines[1].startswith("data: ")


def test_ingest_and_reindex(client):
    payload = {
        "title": "测试制度",
        "category": "测试",
        "text": "# 测试制度\n\n## 条款\n新制度内容，用于验证入库接口。" * 3,
        "strategy": "fixed",
    }
    before = client.get("/api/kb/documents").json()["total"]
    resp = client.post("/api/kb/ingest", json=payload)
    assert resp.status_code == 200
    stats = resp.json()
    assert stats["total_chunks"] >= stats["chunks"]
    after = client.get("/api/kb/documents").json()["total"]
    assert after == before + 1


def test_ingest_validation(client):
    resp = client.post("/api/kb/ingest", json={"title": "", "text": "x"})
    assert resp.status_code == 422


def test_memory_endpoint(client):
    client.post(
        "/api/chat",
        json={
            "message": "报销流程",
            "mode": "fast",
            "stream": False,
            "user_id": "u9",
            "session_id": "s9",
        },
    )
    data = client.get("/api/memory/u9/s9").json()
    assert data["user_id"] == "u9"
    assert data["turns"] == 1
    assert data["window_turns"] >= 1
    assert client.delete("/api/memory/u9").status_code == 200


def test_metrics_and_prompts(client):
    text = client.get("/metrics").text
    assert "kf_kb_chunks" in text
    assert "kf_requests_total" in text
    prompts = client.get("/api/prompts").json()
    assert "answer" in prompts["versions"]


def test_demo_questions(client, eval_dir):
    data = client.get("/api/demo/questions").json()
    assert data["questions"]
    assert {"text", "type", "note"} <= set(data["questions"][0])


def test_index_page(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "KnowledgeFlow" in resp.text

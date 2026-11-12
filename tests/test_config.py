"""配置与后端解析测试。"""

from __future__ import annotations

from app.config import Settings


def test_auto_llm_without_key_falls_back_to_mock(cfg):
    assert cfg.llm_api_key == ""
    assert cfg.resolved_llm_backend() == "mock"
    assert cfg.offline_demo is True


def test_auto_llm_with_key_uses_openai(cfg):
    cfg.llm_backend = "auto"
    cfg.llm_api_key = "sk-test"
    assert cfg.resolved_llm_backend() == "openai"
    assert cfg.offline_demo is False


def test_explicit_backends_win(cfg):
    cfg.llm_backend = "mock"
    cfg.llm_api_key = "sk-test"
    assert cfg.resolved_llm_backend() == "mock"
    cfg.embedding_backend = "hash"
    assert cfg.resolved_embedding_backend() == "hash"
    cfg.vector_backend = "numpy"
    assert cfg.resolved_vector_backend() == "numpy"


def test_rerank_disabled_resolves_none(cfg):
    cfg.rerank_enabled = False
    assert cfg.resolved_rerank_backend() == "none"


def test_describe_shape(cfg):
    info = cfg.describe()
    assert set(info) >= {"app", "llm", "embedding", "vector_store", "rerank", "intent"}
    assert info["llm"]["backend"] in {"mock", "openai"}
    assert info["embedding"]["backend"] == "hash"


def test_cost_estimation(cfg):
    cfg.cost_per_1k_prompt = 0.001
    cfg.cost_per_1k_completion = 0.002
    cost = cfg.cost_usd(1000, 500)
    assert abs(cost - (0.001 + 0.001)) < 1e-9
    assert cfg.cost_usd(0, 0) == 0


def test_directories_created(tmp_path):
    cfg = Settings(
        data_dir=tmp_path / "d",
        index_dir=tmp_path / "i",
        raw_dir=tmp_path / "r",
        llm_backend="mock",
        embedding_backend="hash",
    )
    assert cfg.data_dir.exists() and cfg.index_dir.exists() and cfg.raw_dir.exists()


def test_default_retrieval_params(cfg):
    assert cfg.top_k == 3
    assert cfg.candidate_k == 10
    assert cfg.rrf_k == 60

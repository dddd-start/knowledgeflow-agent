"""Embedding 后端测试(哈希后端,零依赖、确定性)。"""

from __future__ import annotations

import numpy as np

from app.rag.embeddings import Embedder


def test_hash_embedding_deterministic_and_normalized():
    embedder = Embedder(backend="hash", hash_dim=64)
    v1 = embedder.encode(["费用报销流程"])[0]
    v2 = Embedder(backend="hash", hash_dim=64).encode(["费用报销流程"])[0]
    assert v1.shape == (64,)
    assert np.allclose(v1, v2)  # 跨实例稳定(CI 可复现)
    assert abs(float(np.linalg.norm(v1)) - 1.0) < 1e-6


def test_similar_text_has_higher_similarity():
    embedder = Embedder(backend="hash", hash_dim=128)
    query = embedder.encode_query("报销打款时间")
    near = embedder.encode_documents(["报销打款是每月 5 日和 20 日"])[0]
    far = embedder.encode_documents(["VPN 申请需要主管审批"])[0]
    assert float(np.dot(query, near)) > float(np.dot(query, far))


def test_empty_input_and_dim():
    embedder = Embedder(backend="hash", hash_dim=16)
    assert embedder.encode([]).shape == (0, 16)
    assert embedder.dim == 16
    assert embedder.describe()["backend"] == "hash"


def test_unrelated_text_low_similarity():
    embedder = Embedder(backend="hash", hash_dim=256)
    a = embedder.encode(["年假天数"])[0]
    b = embedder.encode(["服务器部署架构"])[0]
    assert float(np.dot(a, b)) < 0.5

"""分块策略测试:边界、重叠、元数据、章节前缀。"""

from __future__ import annotations

from app.rag.chunking import (
    chunk_document,
    chunk_fixed,
    chunk_recursive,
    chunk_semantic,
    strip_markdown,
)
from app.rag.embeddings import Embedder
from app.rag.schemas import Document


def test_chunk_fixed_respects_size_and_overlap():
    text = "甲" * 1200
    chunks = chunk_fixed(text, size=500, overlap=50)
    assert len(chunks) == 3
    assert all(len(c) <= 500 for c in chunks)
    # 相邻块有重叠(下一块开头 = 上一块结尾)
    assert chunks[1][:50] == chunks[0][-50:]


def test_chunk_fixed_short_text_returns_single_chunk():
    assert chunk_fixed("短文本", size=500) == ["短文本"]


def test_chunk_recursive_keeps_paragraph_boundaries():
    text = "第一段内容。\n\n第二段内容。\n\n第三段内容。"
    chunks = chunk_recursive(text, size=12, overlap=0)
    assert chunks
    # 段落没有被从中间切断
    assert any("第一段内容" in c for c in chunks)
    assert any("第三段内容" in c for c in chunks)


def test_chunk_recursive_splits_long_paragraph_by_sentence():
    sentence = "这是一个很长的句子，用来测试按句切分。" * 12
    chunks = chunk_recursive(sentence, size=100, overlap=0)
    assert len(chunks) > 1
    assert all(len(c) <= 120 for c in chunks)


def test_chunk_semantic_covers_content():
    text = "公司提供免费午餐。\n报销每月打款两次。\nVPN 申请需要主管审批。\n差旅住宿有城市分级。"
    embedder = Embedder(backend="hash", hash_dim=32)
    chunks = chunk_semantic(text, embedder.encode, size=60, overlap=0)
    joined = "".join(chunks)
    for piece in ("免费午餐", "打款两次", "VPN", "城市分级"):
        assert piece in joined


def test_chunk_document_metadata():
    doc = Document(
        doc_id="doc-财务制度-费用报销流程",
        title="费用报销流程",
        category="财务制度",
        text="# 费用报销流程\n\n## 报销时限\n30 天内提交。\n\n## 打款\n每月 5 日、20 日打款。",
    )
    chunks = chunk_document(doc, "recursive", size=40, overlap=0)
    assert chunks
    for i, chunk in enumerate(chunks):
        assert chunk.doc_id == doc.doc_id
        assert chunk.metadata["title"] == "费用报销流程"
        assert chunk.metadata["category"] == "财务制度"
        assert chunk.metadata["chunk_index"] == i
        assert chunk.metadata["n_chunks"] == len(chunks)
        assert chunk.chunk_id == f"{doc.doc_id}#{i:03d}"
    # 带小节标题的块会被加上【小节】前缀,便于解释片段归属
    assert any("【" in c.text for c in chunks)


def test_chunk_document_empty_text():
    doc = Document(doc_id="d", title="t", category="c", text="   ")
    assert chunk_document(doc, "recursive") == []


def test_strip_markdown_removes_tags():
    html = "<p>你好</p><script>x</script>  <b>世界</b>"
    text = strip_markdown(html)
    assert "<" not in text and "你好" in text and "世界" in text

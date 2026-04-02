"""文档接入测试:front matter 剥离、doc_id 规则、批量加载。"""

from __future__ import annotations

import pytest

from app.rag.ingest import build_document, load_documents, read_document, strip_front_matter


def test_strip_front_matter():
    text = "---\ncategory: IT服务\ntitle: 账号\n---\n\n# 账号与权限\n正文"
    stripped = strip_front_matter(text)
    assert stripped.startswith("# 账号与权限")
    assert "category" not in stripped


def test_read_document_parses_metadata(tmp_path):
    path = tmp_path / "IT服务-账号与权限管理.md"
    path.write_text(
        "---\ncategory: IT服务\ntitle: 账号\n---\n# 账号与权限管理\n\n## 开通\n入职当天开通。",
        encoding="utf-8",
    )
    doc = read_document(path)
    assert doc.doc_id == "doc-IT服务-账号与权限管理"
    assert doc.category == "IT服务"
    assert doc.title == "账号与权限管理"  # H1 优先于 front matter
    assert doc.text.startswith("# 账号与权限管理")


def test_read_document_html_and_unsupported(tmp_path):
    html = tmp_path / "技术文档-部署.htm"
    html.write_text("<h1>部署</h1><p>使用 docker compose。</p>", encoding="utf-8")
    doc = read_document(html)
    assert "<h1>" not in doc.text
    bad = tmp_path / "a.docx"
    bad.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError):
        read_document(bad)


def test_build_document_slug():
    doc = build_document("接口 / API 接入", "技术文档", "内容" * 20)
    assert doc.doc_id.startswith("doc-")
    assert "/" not in doc.doc_id
    assert doc.category == "技术文档"


def test_load_documents_skips_unsupported(tmp_path):
    (tmp_path / "财务制度-报销.md").write_text("# 报销\n内容", encoding="utf-8")
    (tmp_path / "notes.docx").write_text("x", encoding="utf-8")
    (tmp_path / ".hidden.md").write_text("# hidden", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "a.md").write_text("# a", encoding="utf-8")
    docs = load_documents(tmp_path)
    assert [d.doc_id for d in docs] == ["doc-财务制度-报销"]

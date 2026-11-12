"""文档接入 —— 从 data/raw 读取 → 解析 → 分块 → 向量化 → 建索引。

支持的格式:Markdown / 纯文本 / HTML / PDF(装了 pypdf 才支持,否则给出明确提示)。
文档 ID 规则:`doc-{文件名去扩展名}`,与引用展示、bm25、向量库 payload 全链路一致。

真实企业里这一步通常是异步任务 + 增量更新;本项目为了面试演示把它做成一键同步,
但在 API 层保留了单篇上传接口(/api/kb/ingest),体现"可增量"。
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from app.config import settings
from app.rag.chunking import strip_markdown
from app.rag.schemas import Document

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".md", ".markdown", ".txt", ".html", ".htm", ".pdf"}


def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - 可选依赖
        raise RuntimeError("解析 PDF 需要 pypdf:pip install pypdf") from exc
    reader = PdfReader(str(path))
    return "\n\n".join(page.extract_text() or "" for page in reader.pages)


_FRONT_MATTER = re.compile(r"\A---\s*\n.*?\n---\s*\n", re.DOTALL)


def strip_front_matter(text: str) -> str:
    """剥离 Markdown 的 YAML front matter,避免元信息被切进正文产生检索噪音。"""
    return _FRONT_MATTER.sub("", text, count=1).lstrip("\n")


def read_document(path: Path) -> Document:
    suffix = path.suffix.lower()
    if suffix in {".md", ".markdown"} or suffix in {".txt"}:
        text = path.read_text(encoding="utf-8", errors="ignore")
    elif suffix in {".html", ".htm"}:
        text = strip_markdown(path.read_text(encoding="utf-8", errors="ignore"))
    elif suffix == ".pdf":
        text = _read_pdf(path)
    else:
        raise ValueError(f"不支持的文档格式: {suffix}")

    text = strip_front_matter(text)
    stem = path.stem
    # 文件名规范:类别-标题.md -> category=类别, title=标题
    if "-" in stem:
        category, _, title = stem.partition("-")
        category, title = category.strip(), title.strip()
    else:
        category, title = "未分类", stem
    # 文档内 H1 优先作为标题(更准确)
    h1 = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
    if h1:
        title = h1.group(1).strip()

    return Document(
        doc_id=f"doc-{stem}",
        title=title,
        category=category,
        text=text.strip(),
        source=path.name,
        meta={"suffix": suffix},
    )


def load_documents(raw_dir: Path | str | None = None) -> list[Document]:
    raw = Path(raw_dir or settings.raw_dir)
    if not raw.exists():
        return []
    docs: list[Document] = []
    for path in sorted(raw.iterdir()):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        if path.name.startswith("."):
            continue
        try:
            docs.append(read_document(path))
        except Exception as exc:  # noqa: BLE001  单篇失败不影响整批
            logger.warning("跳过 %s: %s", path.name, exc)
    return docs


def build_document(title: str, category: str, text: str, source: str = "") -> Document:
    """API 上传用:没有文件名时按标题构造稳定 doc_id。"""
    safe = re.sub(r"[^\w\u4e00-\u9fff-]+", "-", title).strip("-") or "upload"
    return Document(
        doc_id=f"doc-{safe}",
        title=title,
        category=category or "未分类",
        text=text.strip(),
        source=source or f"{title}.md",
    )

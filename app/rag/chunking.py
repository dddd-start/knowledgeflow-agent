"""文档分块 —— 三种策略,全部手写实现(不依赖 LangChain 的 splitter)。

分块是 RAG 里最容易被低估的一环:块太大,检索粒度粗、上下文里噪音多;
块太小,一句话拆散、语义不完整。三种策略代表三种取舍:

1. `fixed`      固定窗口 + 重叠:最简单、块大小可控,但容易在句子中间切断;
2. `recursive`  按「标题 → 段落 → 句子」优先级递归切分:块边界尽量落在语义边界上,
                并保留「所属小节标题」作为上下文前缀(面试可讲的细节);
3. `semantic`   用句向量相邻相似度找语义断点:最贴合语义,但要多一次编码,慢。

元数据里带上 doc_id / title / category / chunk_index / 章节,既供过滤,也供引用溯源。
"""

from __future__ import annotations

import re

import numpy as np

from app.rag.schemas import Chunk, Document

_SENTENCE_END = re.compile(r"(?<=[。!?;.!?])\s*")
_HEADING = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)


# ----------------------------------------------------------------------
# 1) 固定窗口
# ----------------------------------------------------------------------
def chunk_fixed(text: str, size: int = 500, overlap: int = 60) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
    return chunks


# ----------------------------------------------------------------------
# 2) 递归切分(标题 -> 段落 -> 句子)
# ----------------------------------------------------------------------
def _split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_END.split(text) if s.strip()]


def _pack(pieces: list[str], size: int, overlap: int) -> list[str]:
    """把语义片段贪心装进 ~size 字符的块,块间保留 overlap 字符重叠。"""
    chunks: list[str] = []
    buf = ""
    for piece in pieces:
        if not buf:
            buf = piece
        elif len(buf) + len(piece) + 1 <= size:
            buf = f"{buf}\n{piece}"
        else:
            chunks.append(buf)
            tail = buf[-overlap:] if overlap else ""
            buf = f"{tail}\n{piece}" if tail else piece
    if buf:
        chunks.append(buf)
    return chunks


def chunk_recursive(text: str, size: int = 500, overlap: int = 60) -> list[str]:
    """按语义完整度从高到低递归:段落装不下就拆句子,句子还超长才硬切。"""
    text = text.strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]

    pieces: list[str] = []
    for para in _split_paragraphs(text):
        if len(para) <= size:
            pieces.append(para)
            continue
        for sent in _split_sentences(para):
            if len(sent) <= size:
                pieces.append(sent)
            else:
                pieces.extend(chunk_fixed(sent, size=size, overlap=0))
    return _pack(pieces, size=size, overlap=overlap)


# ----------------------------------------------------------------------
# 3) 语义切分(句向量相邻相似度)
# ----------------------------------------------------------------------
def chunk_semantic(
    text: str,
    encode,
    size: int = 500,
    overlap: int = 60,
    threshold: float = 0.55,
) -> list[str]:
    """相邻句子相似度低于阈值时断开。

    encode: 形如 embedder.encode(list[str]) -> np.ndarray 的函数。
    """
    sentences = [s for s in _split_sentences(text.strip()) if s]
    if len(sentences) <= 2:
        return [text.strip()] if text.strip() else []

    vectors = np.asarray(encode(sentences))
    groups: list[list[str]] = [[sentences[0]]]
    for i in range(1, len(sentences)):
        sim = float(np.dot(vectors[i - 1], vectors[i]))
        current_len = sum(len(s) for s in groups[-1])
        if sim < threshold and current_len >= size * 0.4:
            groups.append([sentences[i]])
        else:
            groups[-1].append(sentences[i])

    # 合并过短的组,再按窗口兜底切分
    merged: list[str] = []
    for group in groups:
        merged.append("\n".join(group))
    return _pack(merged, size=size, overlap=overlap)


# ----------------------------------------------------------------------
# 文档 -> chunks(带章节上下文前缀)
# ----------------------------------------------------------------------
def _section_prefix(chunk: str) -> str:
    """取块内第一个二级标题作为上下文前缀,提升「该片段属于哪一节」的可读性。"""
    m = re.search(r"^##\s+(.+)$", chunk, re.MULTILINE)
    return f"【{m.group(1).strip()}】" if m else ""


def chunk_document(
    doc: Document,
    strategy: str = "recursive",
    encode=None,
    size: int = 500,
    overlap: int = 60,
) -> list[Chunk]:
    if strategy == "fixed":
        pieces = chunk_fixed(doc.text, size, overlap)
    elif strategy == "semantic" and encode is not None:
        pieces = chunk_semantic(doc.text, encode, size, overlap)
    else:
        pieces = chunk_recursive(doc.text, size, overlap)

    total = len(pieces)
    chunks: list[Chunk] = []
    for i, piece in enumerate(pieces):
        prefix = _section_prefix(piece)
        body = piece if prefix else piece
        chunks.append(
            Chunk(
                chunk_id=f"{doc.doc_id}#{i:03d}",
                doc_id=doc.doc_id,
                text=f"{prefix}{body}" if prefix and not body.startswith(prefix) else body,
                metadata={
                    "doc_id": doc.doc_id,
                    "title": doc.title,
                    "category": doc.category,
                    "chunk_index": i,
                    "n_chunks": total,
                    "source": doc.source,
                    "strategy": strategy,
                },
            )
        )
    return chunks


def strip_markdown(text: str) -> str:
    """粗略转纯文本(HTML/PDF 抽取后清理用)。"""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()

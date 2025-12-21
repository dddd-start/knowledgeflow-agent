"""RAG 子包:解析 → 分块 → 向量化 → 混合检索 → 重排 → 上下文拼装。"""

from app.rag.bm25 import BM25Index
from app.rag.chunking import chunk_document, chunk_fixed, chunk_recursive, chunk_semantic
from app.rag.embeddings import Embedder, get_embedder
from app.rag.hybrid import rrf_fuse
from app.rag.ingest import build_document, load_documents, read_document
from app.rag.pipeline import RetrievalPipeline
from app.rag.rerank import Reranker, get_reranker
from app.rag.schemas import Chunk, Document, Hit, RetrievalTrace, Stage
from app.rag.vectorstore import (
    BaseVectorStore,
    ChromaVectorStore,
    NumpyVectorStore,
    create_vector_store,
    get_vector_store,
)

__all__ = [
    "BM25Index",
    "BaseVectorStore",
    "Chunk",
    "ChromaVectorStore",
    "Document",
    "Embedder",
    "Hit",
    "NumpyVectorStore",
    "Reranker",
    "RetrievalPipeline",
    "RetrievalTrace",
    "Stage",
    "build_document",
    "chunk_document",
    "chunk_fixed",
    "chunk_recursive",
    "chunk_semantic",
    "create_vector_store",
    "get_embedder",
    "get_reranker",
    "get_vector_store",
    "load_documents",
    "read_document",
    "rrf_fuse",
]

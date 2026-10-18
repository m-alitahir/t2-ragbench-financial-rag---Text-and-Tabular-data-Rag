"""Retrieval components for lexical, dense, fused, and reranked search.

The package keeps each ranking stage independent.  This makes ablation tests
simple: callers can measure BM25 alone, dense retrieval alone, their fusion,
and optional cloud reranking without changing the indexed evidence.
"""

from .embeddings import (
    EmbeddingBackend,
    OllamaEmbeddingBackend,
    SentenceTransformerEmbeddingBackend,
    create_embedding_backend,
)
from .faiss_store import FaissIndexManifest, FaissVectorStore, fingerprint_chunks
from .fusion import reciprocal_rank_fusion
from .lexical import SQLiteBM25Index
from .reranker import DeepInfraReranker
from .router import QueryRoute, route_query
from .service import HybridSearchService

__all__ = [
    "DeepInfraReranker",
    "EmbeddingBackend",
    "FaissIndexManifest",
    "FaissVectorStore",
    "HybridSearchService",
    "OllamaEmbeddingBackend",
    "QueryRoute",
    "SQLiteBM25Index",
    "SentenceTransformerEmbeddingBackend",
    "create_embedding_backend",
    "fingerprint_chunks",
    "reciprocal_rank_fusion",
    "route_query",
]

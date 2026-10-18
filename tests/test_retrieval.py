"""Unit tests for retrieval stages without model downloads or live services."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import numpy as np
import pytest

from t2rag.config import AppConfig, PathSettings, RetrievalSettings
from t2rag.errors import ExternalServiceError
from t2rag.models import ChunkRecord, SearchHit
from t2rag.retrieval.embeddings import OllamaEmbeddingBackend
from t2rag.retrieval.faiss_store import FaissVectorStore
from t2rag.retrieval.fusion import reciprocal_rank_fusion
from t2rag.retrieval.lexical import SQLiteBM25Index
from t2rag.retrieval.query_correction import FinancialQueryCorrector
from t2rag.retrieval.reranker import DeepInfraReranker
from t2rag.retrieval.service import HybridSearchService


def _chunks() -> list[ChunkRecord]:
    return [
        ChunkRecord(
            chunk_id=1,
            context_uid="finqa:one",
            document_id="report-one",
            dataset="finqa",
            split="train",
            kind="table",
            source_text="Revenue increased to 100 million dollars in 2024.",
            page_number=4,
            metadata={"alias_path": "reports/one.pdf"},
        ),
        ChunkRecord(
            chunk_id=2,
            context_uid="finqa:two",
            document_id="report-two",
            dataset="finqa",
            split="train",
            kind="text",
            source_text="Cash decreased to 20 million dollars in 2024.",
            description_text="A generated note mentions the secret revenue category.",
            metadata={"alias_path": "reports/two.pdf"},
        ),
        ChunkRecord(
            chunk_id=3,
            context_uid="tatdqa:three",
            document_id="report-three",
            dataset="tat-dqa",
            split="dev",
            kind="text",
            source_text="Operating expenses were 75 million dollars.",
        ),
    ]


def _hit(chunk_id: int, score: float, rank: int, source: str) -> SearchHit:
    return SearchHit(
        chunk_id=chunk_id,
        context_uid=f"context-{chunk_id}",
        document_id=f"document-{chunk_id}",
        source_text=f"text-{chunk_id}",
        score=score,
        rank=rank,
        retrieval_source=source,
    )


def _config(tmp_path: Path) -> AppConfig:
    return AppConfig(
        paths=PathSettings(
            dataset_root=tmp_path / "dataset",
            work_root=tmp_path / "work",
            registry_db=tmp_path / "work" / "registry.sqlite3",
            index_dir=tmp_path / "work" / "indexes",
            output_dir=tmp_path / "work" / "outputs",
        ),
        retrieval=RetrievalSettings(
            lexical_candidates=3,
            dense_candidates=3,
            fused_candidates=3,
            rerank_candidates=3,
            answer_contexts=2,
        ),
    )


def test_sqlite_bm25_excludes_descriptions_and_hydrates_dense_ids(tmp_path: Path) -> None:
    index = SQLiteBM25Index(tmp_path / "lexical.sqlite3")
    assert index.build(_chunks(), include_descriptions=False, source_fingerprint="snapshot") == 3

    hits = index.search("revenue 2024", top_k=3)
    assert hits[0].chunk_id == 1
    assert all(hit.retrieval_source == "bm25" for hit in hits)
    assert index.search("secret", top_k=3) == []
    assert index.source_fingerprint() == "snapshot"

    dense = index.hydrate([(2, 0.9), (1, 0.7)], retrieval_source="dense")
    assert [hit.chunk_id for hit in dense] == [2, 1]
    assert dense[0].source_text.startswith("Cash decreased")
    assert dense[0].alias_path == "reports/two.pdf"


def test_sqlite_vocabulary_reads_source_terms_without_a_rebuild(tmp_path: Path) -> None:
    index = SQLiteBM25Index(tmp_path / "lexical.sqlite3")
    index.build(_chunks())

    vocabulary = index.vocabulary()
    frequencies = index.vocabulary_stats()

    assert "revenue" in vocabulary
    assert "operating" in vocabulary
    assert frequencies["revenue"] >= 1


def test_sqlite_bm25_can_index_description_without_replacing_evidence(tmp_path: Path) -> None:
    index = SQLiteBM25Index(tmp_path / "lexical.sqlite3")
    index.build(_chunks(), include_descriptions=True)

    hit = index.search("secret", top_k=1)[0]
    assert hit.chunk_id == 2
    assert "generated note" not in hit.source_text.casefold()


def test_sqlite_bm25_filters_before_ranking(tmp_path: Path) -> None:
    index = SQLiteBM25Index(tmp_path / "lexical.sqlite3")
    index.build(_chunks())

    hits = index.search(
        "million",
        top_k=10,
        dataset="tat-dqa",
        split="dev",
    )

    assert [hit.chunk_id for hit in hits] == [3]
    assert all(hit.dataset == "tat-dqa" and hit.split == "dev" for hit in hits)


def test_rrf_uses_rank_not_incompatible_raw_scores() -> None:
    bm25 = [_hit(1, 1000.0, 1, "bm25"), _hit(2, 900.0, 2, "bm25")]
    dense = [_hit(2, 0.8, 1, "dense"), _hit(3, 0.7, 2, "dense")]

    fused = reciprocal_rank_fusion({"bm25": bm25, "dense": dense}, k=60)

    assert [hit.chunk_id for hit in fused] == [2, 1, 3]
    assert fused[0].component_scores["bm25"] == 900.0
    assert fused[0].component_scores["dense"] == 0.8
    assert fused[0].retrieval_source == "rrf"


def test_ollama_embeddings_are_batched_and_normalized() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        payload = json.loads(request.content)
        vectors = [[3.0, 4.0] if "one" in text else [0.0, 2.0] for text in payload["input"]]
        return httpx.Response(200, json={"embeddings": vectors})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    backend = OllamaEmbeddingBackend(
        base_url="http://127.0.0.1:11434",
        model="qwen3-embedding:4b",
        batch_size=1,
        max_retries=0,
        client=client,
    )
    vectors = backend.embed_documents(["one", "two"])

    assert len(requests) == 2
    assert requests[0].url.path == "/api/embed"
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), [1.0, 1.0])
    np.testing.assert_allclose(vectors[0], [0.6, 0.8])
    client.close()


def test_ollama_http_error_reports_size_without_source_text() -> None:
    secret_text = "CONFIDENTIAL FINANCIAL SOURCE"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": "input exceeds context length"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    backend = OllamaEmbeddingBackend(
        base_url="http://127.0.0.1:11434",
        model="qwen3-embedding:4b",
        max_retries=0,
        client=client,
    )

    with pytest.raises(ExternalServiceError) as captured:
        backend.embed_documents([secret_text])

    message = str(captured.value)
    assert "HTTP 400" in message
    assert "input exceeds context length" in message
    assert f"longest input {len(secret_text)} characters" in message
    assert secret_text not in message
    client.close()


def test_deepinfra_reranker_uses_inference_endpoint_and_preserves_metadata() -> None:
    captured: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(
            {
                "url": str(request.url),
                "authorization": request.headers.get("authorization"),
                "body": json.loads(request.content),
            }
        )
        return httpx.Response(200, json={"scores": [0.1, 0.9]})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    reranker = DeepInfraReranker(
        api_key="test-key",
        model="Qwen/Qwen3-Reranker-4B",
        batch_size=2,
        max_retries=0,
        client=client,
        sleeper=lambda _: None,
    )
    reranked = reranker.rerank(
        "What was revenue?",
        [_hit(1, 0.02, 1, "rrf"), _hit(2, 0.01, 2, "rrf")],
    )

    assert captured[0]["url"].endswith("/v1/inference/Qwen/Qwen3-Reranker-4B")
    assert captured[0]["authorization"] == "Bearer test-key"
    assert captured[0]["body"]["queries"] == [
        "What was revenue?",
        "What was revenue?",
    ]
    assert [hit.chunk_id for hit in reranked] == [2, 1]
    assert reranked[0].component_scores["qwen3_reranker"] == 0.9
    client.close()


def test_faiss_store_normalizes_and_validates_manifest(tmp_path: Path) -> None:
    pytest.importorskip("faiss")
    store = FaissVectorStore(tmp_path / "dense.faiss", tmp_path / "dense.json")
    manifest = store.build(
        np.asarray([[3.0, 4.0], [1.0, 0.0]], dtype=np.float32),
        [11, 22],
        embedding_model="example-model",
        source_fingerprint="source-hash",
    )
    assert manifest.dimension == 2
    assert store.search([[3.0, 4.0]], top_k=2)[0][0] == 11

    reloaded = FaissVectorStore(tmp_path / "dense.faiss", tmp_path / "dense.json")
    reloaded.load(
        expected_embedding_model="example-model",
        expected_source_fingerprint="source-hash",
    )
    with pytest.raises(ValueError, match="Embedding model mismatch"):
        reloaded.load(expected_embedding_model="different-model")


class _FakeEmbedder:
    model_id = "fake-embedding-model"

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.ones((len(texts), 2), dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return np.asarray([[1.0, 0.0]], dtype=np.float32)

    def close(self) -> None:
        return None


class _FakeVectorStore:
    def __init__(self) -> None:
        self.is_loaded = False
        self.manifest: Any = None
        self.loaded_with: dict[str, Any] = {}
        self.built_ids: list[int] = []
        self.built_fingerprint = ""

    def build(
        self,
        vectors: Any,
        chunk_ids: list[int],
        *,
        embedding_model: str,
        source_fingerprint: str,
    ) -> Any:
        self.built_ids = list(chunk_ids)
        self.built_fingerprint = source_fingerprint
        return SimpleNamespace(vector_count=len(chunk_ids), embedding_model=embedding_model)

    def load(self, **kwargs: Any) -> Any:
        self.loaded_with = kwargs
        self.manifest = SimpleNamespace(
            source_fingerprint=kwargs["expected_source_fingerprint"],
            vector_count=3,
        )
        self.is_loaded = True
        return self.manifest

    def search(self, query_vector: Any, *, top_k: int) -> list[tuple[int, float]]:
        return [(2, 0.95), (1, 0.7), (3, 0.6)][:top_k]


def test_hybrid_service_fuses_lexical_and_dense_results(tmp_path: Path) -> None:
    lexical = SQLiteBM25Index(tmp_path / "lexical.sqlite3")
    lexical.build(
        _chunks(),
        source_fingerprint="all-snapshot",
        dense_source_fingerprint="same-snapshot",
    )
    vector_store = _FakeVectorStore()
    service = HybridSearchService(
        config=_config(tmp_path),
        lexical_index=lexical,
        embedding_backend=_FakeEmbedder(),
        vector_store=vector_store,  # type: ignore[arg-type]
    )

    hits = service.search("revenue 2024", use_reranker=False)

    assert hits
    assert vector_store.loaded_with["expected_embedding_model"] == "fake-embedding-model"
    assert vector_store.loaded_with["expected_source_fingerprint"] == "same-snapshot"
    assert any("dense" in hit.component_scores for hit in hits)


def test_typo_expansion_adds_corrected_bm25_candidates_only_when_enabled(
    tmp_path: Path,
) -> None:
    lexical = SQLiteBM25Index(tmp_path / "lexical.sqlite3")
    lexical.build(_chunks())
    corrector = FinancialQueryCorrector(lambda: {"revenue", "million"})
    service = HybridSearchService(
        config=_config(tmp_path),
        lexical_index=lexical,
        query_corrector=corrector,
    )

    unchanged = service.search_lexical("revanue", typo_tolerant=False)
    corrected = service.search_lexical("revanue", typo_tolerant=True)

    assert unchanged == []
    assert corrected[0].chunk_id == 1
    assert corrected[0].metadata["query_correction_corrected"] == "revenue"
    assert "bm25_corrected" in corrected[0].component_scores


def test_typo_expansion_calls_reranker_once_with_original_question(
    tmp_path: Path,
) -> None:
    class CapturingReranker:
        def __init__(self) -> None:
            self.queries: list[str] = []

        def rerank(
            self,
            query: str,
            hits: list[SearchHit],
            *,
            top_k: int | None,
        ) -> list[SearchHit]:
            self.queries.append(query)
            return hits if top_k is None else hits[:top_k]

        def close(self) -> None:
            return None

    lexical = SQLiteBM25Index(tmp_path / "lexical.sqlite3")
    lexical.build(_chunks())
    reranker = CapturingReranker()
    service = HybridSearchService(
        config=_config(tmp_path),
        lexical_index=lexical,
        reranker=reranker,  # type: ignore[arg-type]
        query_corrector=FinancialQueryCorrector(lambda: {"revenue"}),
    )

    hits = service.search(
        "revanue",
        use_dense=False,
        use_reranker=True,
        typo_tolerant=True,
    )

    assert hits
    assert reranker.queries == ["revanue"]


def test_hybrid_service_filters_dense_and_lexical_candidates(tmp_path: Path) -> None:
    lexical = SQLiteBM25Index(tmp_path / "lexical.sqlite3")
    lexical.build(
        _chunks(),
        source_fingerprint="all-snapshot",
        dense_source_fingerprint="same-snapshot",
    )
    service = HybridSearchService(
        config=_config(tmp_path),
        lexical_index=lexical,
        embedding_backend=_FakeEmbedder(),
        vector_store=_FakeVectorStore(),  # type: ignore[arg-type]
    )

    hits = service.search(
        "operating expenses million",
        use_reranker=False,
        dataset="tat-dqa",
        split="dev",
    )

    assert [hit.chunk_id for hit in hits] == [3]
    assert all(hit.dataset == "tat-dqa" and hit.split == "dev" for hit in hits)


def test_dense_build_excludes_unbounded_parent_contexts(tmp_path: Path) -> None:
    chunks = _chunks()
    chunks.append(
        ChunkRecord(
            chunk_id=4,
            context_uid="finqa:parent",
            document_id="report-parent",
            dataset="finqa",
            split="train",
            kind="whole_context",
            source_text="very long parent context " * 1000,
        )
    )
    lexical = SQLiteBM25Index(tmp_path / "lexical.sqlite3")
    vector_store = _FakeVectorStore()
    service = HybridSearchService(
        config=_config(tmp_path),
        lexical_index=lexical,
        embedding_backend=_FakeEmbedder(),
        vector_store=vector_store,  # type: ignore[arg-type]
    )

    result = service.build_indexes(chunks)

    assert lexical.count() == 4
    assert vector_store.built_ids == [1, 2, 3]
    assert result["dense_chunk_count"] == 3
    assert result["skipped_parent_contexts"] == 1
    assert lexical.dense_source_fingerprint() == vector_store.built_fingerprint

"""Orchestrate lexical, dense, fused, and reranked retrieval stages."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path
from typing import Any

from t2rag.config import AppConfig
from t2rag.models import ChunkRecord, SearchHit

from .embeddings import EmbeddingBackend, create_embedding_backend
from .faiss_store import FaissVectorStore, fingerprint_chunks
from .fusion import reciprocal_rank_fusion
from .lexical import SQLiteBM25Index
from .query_correction import FinancialQueryCorrector, QueryExpansion
from .reranker import DeepInfraReranker
from .router import QueryRoute, route_query


class HybridSearchService:
    """Coordinate independent retrieval stages through one small interface.

    The service supports controlled ablations.  Lexical retrieval always runs.
    Dense retrieval and cloud reranking run only when their components are
    configured and the caller enables them.
    """

    def __init__(
        self,
        *,
        config: AppConfig,
        lexical_index: SQLiteBM25Index,
        embedding_backend: EmbeddingBackend | None = None,
        vector_store: FaissVectorStore | None = None,
        reranker: DeepInfraReranker | None = None,
        query_corrector: FinancialQueryCorrector | None = None,
    ) -> None:
        if (embedding_backend is None) != (vector_store is None):
            raise ValueError(
                "embedding_backend and vector_store must both be configured or both be absent."
            )
        self.config = config
        self.lexical_index = lexical_index
        self.embedding_backend = embedding_backend
        self.vector_store = vector_store
        self.reranker = reranker
        self.query_corrector = query_corrector or FinancialQueryCorrector(
            lexical_index.vocabulary_stats,
            minimum_length=config.retrieval.typo_minimum_length,
            similarity_threshold=config.retrieval.typo_similarity_threshold,
            ambiguity_margin=config.retrieval.typo_ambiguity_margin,
            max_edits=config.retrieval.typo_max_edits,
        )

    @classmethod
    def from_config(
        cls,
        config: AppConfig,
        *,
        enable_dense: bool = True,
        enable_reranker: bool = False,
    ) -> "HybridSearchService":
        """Create components without building indexes or contacting services.

        Cloud reranking defaults to disabled.  Enabling it is the explicit
        action that reads ``DEEPINFRA_API_KEY`` from configuration.
        """

        index_dir = Path(config.paths.index_dir)
        lexical = SQLiteBM25Index(index_dir / "lexical.sqlite3")
        embedder: EmbeddingBackend | None = None
        vector_store: FaissVectorStore | None = None
        if enable_dense:
            embedder = create_embedding_backend(config)
            vector_store = FaissVectorStore(
                index_dir / "dense.faiss",
                index_dir / "dense.manifest.json",
            )

        reranker: DeepInfraReranker | None = None
        if enable_reranker:
            reranker = DeepInfraReranker(
                api_key=config.deepinfra_api_key,
                base_url=config.models.deepinfra_inference_url,
                model=config.models.reranker_model,
                timeout_seconds=config.runtime.request_timeout_seconds,
                max_retries=config.runtime.max_retries,
                batch_size=config.runtime.embedding_batch_size,
            )
        return cls(
            config=config,
            lexical_index=lexical,
            embedding_backend=embedder,
            vector_store=vector_store,
            reranker=reranker,
        )

    def build_indexes(self, chunks: Iterable[ChunkRecord]) -> dict[str, Any]:
        """Build lexical search and an optional child-only dense index.

        BM25 keeps complete contexts for the lexical baseline. FAISS embeds
        only bounded child chunks. The answer stage expands a retrieved child
        back to its complete parent context. Separate fingerprints preserve
        strict stale-index checks for both snapshots.
        """

        materialized = list(chunks)
        if not materialized:
            raise ValueError("Cannot build retrieval indexes from an empty chunk list.")
        source_fingerprint = fingerprint_chunks(
            materialized,
            include_descriptions=self.config.retrieval.include_descriptions,
        )
        dense_chunks = [
            chunk for chunk in materialized if chunk.kind != "whole_context"
        ]
        dense_source_fingerprint = ""
        if self.embedding_backend is not None:
            if not dense_chunks:
                raise ValueError("Dense indexing requires at least one child chunk.")
            dense_source_fingerprint = fingerprint_chunks(
                dense_chunks,
                include_descriptions=self.config.retrieval.include_descriptions,
            )
        lexical_count = self.lexical_index.build(
            materialized,
            include_descriptions=self.config.retrieval.include_descriptions,
            source_fingerprint=source_fingerprint,
            dense_source_fingerprint=dense_source_fingerprint,
        )
        result: dict[str, Any] = {
            "source_fingerprint": source_fingerprint,
            "lexical_count": lexical_count,
        }

        if self.embedding_backend is not None and self.vector_store is not None:
            texts = [
                chunk.text_for_index(self.config.retrieval.include_descriptions)
                for chunk in dense_chunks
            ]
            vectors = self.embedding_backend.embed_documents(texts)
            ids = [
                chunk.chunk_id
                for chunk in dense_chunks
                if chunk.chunk_id is not None
            ]
            # fingerprint_chunks already rejects missing IDs.  This explicit
            # check keeps static type narrowing and a clear future invariant.
            if len(ids) != len(dense_chunks):
                raise ValueError("Every dense-index chunk must have chunk_id.")
            manifest = self.vector_store.build(
                vectors,
                ids,
                embedding_model=self.embedding_backend.model_id,
                source_fingerprint=dense_source_fingerprint,
            )
            result["dense_manifest"] = manifest
            result["dense_chunk_count"] = len(dense_chunks)
            result["skipped_parent_contexts"] = len(materialized) - len(dense_chunks)
        return result

    @staticmethod
    def _filter_scope(
        hits: Iterable[SearchHit],
        *,
        dataset: str | None,
        split: str | None,
    ) -> list[SearchHit]:
        """Keep one dataset/split scope and rebuild consecutive ranks."""

        clean_dataset = dataset.strip().casefold() if dataset else None
        clean_split = split.strip().casefold() if split else None
        selected = [
            hit
            for hit in hits
            if (clean_dataset is None or hit.dataset.casefold() == clean_dataset)
            and (clean_split is None or hit.split.casefold() == clean_split)
        ]
        return [replace(hit, rank=rank) for rank, hit in enumerate(selected, start=1)]

    def _dense_hits(
        self,
        query: str,
        *,
        dataset: str | None = None,
        split: str | None = None,
    ) -> list[SearchHit]:
        """Run dense retrieval and hydrate IDs from lexical source metadata."""

        if self.embedding_backend is None or self.vector_store is None:
            return []
        source_fingerprint = self.lexical_index.dense_source_fingerprint()
        if not source_fingerprint:
            raise ValueError(
                "The lexical metadata is not paired with a dense child index. "
                "Run build-dense."
            )
        if not self.vector_store.is_loaded:
            self.vector_store.load(
                expected_embedding_model=self.embedding_backend.model_id,
                expected_source_fingerprint=source_fingerprint,
            )
        elif (
            self.vector_store.manifest is None
            or self.vector_store.manifest.source_fingerprint != source_fingerprint
        ):
            raise ValueError("Loaded dense and lexical indexes use different chunk snapshots.")

        query_vector = self.embedding_backend.embed_query(query)
        wanted = self.config.retrieval.dense_candidates
        if dataset is None and split is None:
            scores = self.vector_store.search(query_vector, top_k=wanted)
            return self.lexical_index.hydrate(scores, retrieval_source="dense")

        manifest = self.vector_store.manifest
        if manifest is None:
            raise RuntimeError("The loaded dense index has no manifest.")
        search_limit = min(max(wanted * 4, 64), manifest.vector_count)
        while True:
            scores = self.vector_store.search(query_vector, top_k=search_limit)
            hydrated = self.lexical_index.hydrate(scores, retrieval_source="dense")
            scoped = self._filter_scope(
                hydrated,
                dataset=dataset,
                split=split,
            )
            if len(scoped) >= wanted or search_limit >= manifest.vector_count:
                return scoped[:wanted]
            search_limit = min(search_limit * 2, manifest.vector_count)

    def search_lexical(
        self,
        query: str,
        *,
        top_k: int | None = None,
        dataset: str | None = None,
        split: str | None = None,
        typo_tolerant: bool = False,
    ) -> list[SearchHit]:
        """Run BM25 alone for a controlled retrieval baseline."""

        limit = self.config.retrieval.answer_contexts if top_k is None else top_k
        candidates, expansion = self._lexical_candidates(
            query,
            typo_tolerant=typo_tolerant,
            dataset=dataset,
            split=split,
        )
        parents = self._parent_context_hits(candidates, top_k=limit)
        return self._attach_query_expansion(parents, expansion)

    def _lexical_candidates(
        self,
        query: str,
        *,
        typo_tolerant: bool,
        dataset: str | None,
        split: str | None,
    ) -> tuple[list[SearchHit], QueryExpansion]:
        """Search original and corrected BM25 text, then fuse both lists."""

        original = self.lexical_index.search(
            query,
            top_k=self.config.retrieval.lexical_candidates,
            dataset=dataset,
            split=split,
        )
        expansion = QueryExpansion(original=query, corrected=query)
        if typo_tolerant:
            expansion = self.query_corrector.expand(query)
        if not expansion.applied:
            return original, expansion
        corrected = self.lexical_index.search(
            expansion.corrected,
            top_k=self.config.retrieval.lexical_candidates,
            dataset=dataset,
            split=split,
        )
        fused = reciprocal_rank_fusion(
            {
                "bm25_original": original,
                "bm25_corrected": corrected,
            },
            k=self.config.retrieval.rrf_k,
            limit=self.config.retrieval.lexical_candidates,
        )
        return fused, expansion

    @staticmethod
    def _attach_query_expansion(
        hits: Iterable[SearchHit],
        expansion: QueryExpansion,
    ) -> list[SearchHit]:
        """Attach transparent correction metadata without changing evidence."""

        materialized = list(hits)
        if not expansion.applied:
            return materialized
        edits = [
            {
                "original": edit.original,
                "corrected": edit.corrected,
                "similarity": edit.similarity,
            }
            for edit in expansion.edits
        ]
        return [
            replace(
                hit,
                metadata={
                    **hit.metadata,
                    "query_correction_original": expansion.original,
                    "query_correction_corrected": expansion.corrected,
                    "query_correction_edits": edits,
                },
            )
            for hit in materialized
        ]

    def search_dense(
        self,
        query: str,
        *,
        top_k: int | None = None,
        dataset: str | None = None,
        split: str | None = None,
    ) -> list[SearchHit]:
        """Run FAISS alone for a controlled dense-retrieval baseline."""

        if self.embedding_backend is None or self.vector_store is None:
            raise ValueError("Dense retrieval is not configured for this service.")
        limit = self.config.retrieval.answer_contexts if top_k is None else top_k
        if limit <= 0 or not query.strip():
            return []
        dense_hits = self._dense_hits(query, dataset=dataset, split=split)
        return self._parent_context_hits(dense_hits, top_k=limit)

    def _parent_context_hits(
        self,
        hits: Iterable[SearchHit],
        *,
        top_k: int,
    ) -> list[SearchHit]:
        """Collapse chunk hits by physical PDF and load full parent contexts.

        Retrieval scores belong to child chunks. Answer evidence belongs to the
        complete source context. Keeping the triggering ``chunk_id`` preserves
        traceability while replacing only ``source_text`` with the parent text.
        Exact duplicate FinQA, ConvFinQA, and TAT aliases collapse through the
        canonical SHA-256 ``document_id``.
        """

        if top_k <= 0:
            return []
        ordered = sorted(hits, key=lambda hit: hit.rank)
        selected: list[SearchHit] = []
        seen_documents: set[str] = set()
        for hit in ordered:
            if hit.document_id in seen_documents:
                continue
            seen_documents.add(hit.document_id)
            selected.append(hit)
            if len(selected) >= top_k:
                break
        if not selected:
            return []

        # Tests and standalone index inspection can use retrieval without the
        # registry. Production commands build the registry first and therefore
        # receive complete parent contexts here.
        registry_path = Path(self.config.paths.registry_db)
        parents: dict[str, tuple[str, str, int | None]] = {}
        if registry_path.is_file():
            context_uids = [hit.context_uid for hit in selected]
            placeholders = ",".join("?" for _ in context_uids)
            with sqlite3.connect(registry_path) as connection:
                rows = connection.execute(
                    f"""
                    SELECT c.context_uid, c.source_text, c.page_number, a.relative_path
                    FROM context_records AS c
                    JOIN document_aliases AS a ON a.alias_id = c.alias_id
                    WHERE c.context_uid IN ({placeholders})
                    """,
                    context_uids,
                ).fetchall()
            parents = {
                str(row[0]): (str(row[1]), str(row[3]), row[2]) for row in rows
            }

        expanded: list[SearchHit] = []
        for rank, hit in enumerate(selected, start=1):
            parent = parents.get(hit.context_uid)
            expanded.append(
                replace(
                    hit,
                    source_text=parent[0] if parent else hit.source_text,
                    alias_path=parent[1] if parent else hit.alias_path,
                    page_number=parent[2] if parent else hit.page_number,
                    rank=rank,
                    metadata={
                        **hit.metadata,
                        "trigger_chunk_id": hit.chunk_id,
                        "evidence_scope": "parent_context" if parent else "chunk",
                    },
                )
            )
        return expanded

    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        use_dense: bool = True,
        use_reranker: bool = True,
        dataset: str | None = None,
        split: str | None = None,
        typo_tolerant: bool = False,
    ) -> list[SearchHit]:
        """Run the configured hybrid pipeline for one question.

        ``top_k`` defaults to the number of contexts given to answer
        generation.  Candidate counts for earlier stages remain independent so
        development evaluation can tune them without changing this API.
        """

        if not query.strip():
            return []
        final_k = self.config.retrieval.answer_contexts if top_k is None else top_k
        if final_k <= 0:
            return []

        lexical_hits, expansion = self._lexical_candidates(
            query,
            typo_tolerant=typo_tolerant,
            dataset=dataset,
            split=split,
        )
        rankings: dict[str, list[SearchHit]] = {"bm25": lexical_hits}
        if use_dense and self.embedding_backend is not None:
            rankings["dense"] = self._dense_hits(
                query,
                dataset=dataset,
                split=split,
            )

        fused = reciprocal_rank_fusion(
            rankings,
            k=self.config.retrieval.rrf_k,
            limit=self.config.retrieval.fused_candidates,
        )
        if use_reranker and self.reranker is not None:
            candidates = fused[: self.config.retrieval.rerank_candidates]
            reranked = self.reranker.rerank(query, candidates, top_k=None)
            parents = self._parent_context_hits(reranked, top_k=final_k)
            return self._attach_query_expansion(parents, expansion)
        parents = self._parent_context_hits(fused, top_k=final_k)
        return self._attach_query_expansion(parents, expansion)

    def search_routed(
        self,
        query: str,
        *,
        top_k: int | None = None,
        use_reranker: bool = True,
        dataset: str | None = None,
        split: str | None = None,
        typo_tolerant: bool = False,
    ) -> tuple[QueryRoute, list[SearchHit]]:
        """Route identifiers to BM25 and other questions to hybrid search.

        The route is returned with the hits, so evaluation can inspect every
        decision. Visual questions still use text retrieval. Vision runs only
        during ingestion and does not add query-time model cost.
        """

        route = route_query(query)
        hits = self.search(
            query,
            top_k=top_k,
            use_dense=route.use_dense,
            use_reranker=use_reranker,
            dataset=dataset,
            split=split,
            typo_tolerant=typo_tolerant,
        )
        labeled = [
            replace(
                hit,
                retrieval_source=f"routed:{route.query_type}:{hit.retrieval_source}",
                metadata={
                    **hit.metadata,
                    "query_route": route.query_type,
                    "query_route_reason": route.reason,
                },
            )
            for hit in hits
        ]
        return route, labeled

    def close(self) -> None:
        """Release local HTTP/model resources owned by retrieval components."""

        if self.embedding_backend is not None:
            self.embedding_backend.close()
        if self.reranker is not None:
            self.reranker.close()

    def __enter__(self) -> "HybridSearchService":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

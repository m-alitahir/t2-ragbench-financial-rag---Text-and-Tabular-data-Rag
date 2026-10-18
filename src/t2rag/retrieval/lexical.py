"""SQLite FTS5 lexical retrieval with reproducible BM25 ranking."""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from t2rag.models import ChunkRecord, SearchHit


_SCHEMA_VERSION = "1"
_QUERY_TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)


class SQLiteBM25Index:
    """Store chunks in SQLite and rank them with FTS5's BM25 function.

    FTS5 considers a smaller BM25 value better.  ``search`` negates that
    value so every retrieval component in this project follows one rule:
    larger scores are better.

    The source text and provenance live in a normal SQLite table.  The FTS
    table contains only the text used for retrieval.  This separation lets us
    index an optional generated description without presenting that generated
    text to the answer model as evidence.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._vocabulary_cache: dict[str, int] | None = None

    def _connect(self) -> sqlite3.Connection:
        """Open one short-lived connection with predictable row handling."""

        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    @staticmethod
    def _create_schema(connection: sqlite3.Connection) -> None:
        """Create the metadata and FTS tables.

        FTS5 is included with standard CPython SQLite builds.  The explicit
        error gives a useful diagnosis if a custom Python build omits it.
        """

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS lexical_metadata (
                chunk_id INTEGER PRIMARY KEY,
                context_uid TEXT NOT NULL,
                document_id TEXT NOT NULL,
                dataset TEXT NOT NULL,
                split TEXT NOT NULL,
                page_number INTEGER,
                source_text TEXT NOT NULL,
                alias_path TEXT NOT NULL,
                metadata_json TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS lexical_settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        try:
            connection.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS lexical_fts USING fts5(
                    chunk_id UNINDEXED,
                    index_text,
                    tokenize = 'unicode61 remove_diacritics 2'
                )
                """
            )
        except sqlite3.OperationalError as exc:
            if "fts5" in str(exc).casefold():
                raise RuntimeError(
                    "This SQLite build does not provide FTS5, which is required for BM25."
                ) from exc
            raise

    def build(
        self,
        chunks: Iterable[ChunkRecord],
        *,
        include_descriptions: bool = False,
        source_fingerprint: str = "",
        dense_source_fingerprint: str = "",
    ) -> int:
        """Replace the lexical index with ``chunks`` and return its row count.

        The operation is transactional.  A duplicate or missing chunk ID
        rolls the complete replacement back instead of leaving a partial
        index.  Questions and answers are never accepted by this API; callers
        can supply only the shared ``ChunkRecord`` evidence type.
        """

        materialized = list(chunks)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        seen_ids: set[int] = set()
        rows: list[tuple[Any, ...]] = []
        fts_rows: list[tuple[int, str]] = []
        for chunk in materialized:
            if chunk.chunk_id is None:
                raise ValueError("Every indexed chunk must have a persistent chunk_id.")
            if chunk.chunk_id in seen_ids:
                raise ValueError(f"Duplicate chunk_id in lexical build: {chunk.chunk_id}")
            if not chunk.source_text.strip():
                raise ValueError(f"Chunk {chunk.chunk_id} has empty source_text.")
            seen_ids.add(chunk.chunk_id)

            # alias_path is a presentation field, not part of the shared
            # dataclass.  Ingestion stores it in metadata when it is known.
            alias_path = str(chunk.metadata.get("alias_path", ""))
            metadata_json = json.dumps(
                chunk.metadata,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            rows.append(
                (
                    chunk.chunk_id,
                    chunk.context_uid,
                    chunk.document_id,
                    chunk.dataset,
                    chunk.split,
                    chunk.page_number,
                    chunk.source_text,
                    alias_path,
                    metadata_json,
                )
            )
            fts_rows.append(
                (chunk.chunk_id, chunk.text_for_index(include_descriptions))
            )

        with self._connect() as connection:
            self._create_schema(connection)
            connection.execute("DELETE FROM lexical_fts")
            connection.execute("DELETE FROM lexical_metadata")
            connection.executemany(
                """
                INSERT INTO lexical_metadata (
                    chunk_id, context_uid, document_id, dataset, split,
                    page_number, source_text, alias_path, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            connection.executemany(
                "INSERT INTO lexical_fts (chunk_id, index_text) VALUES (?, ?)",
                fts_rows,
            )
            settings = {
                "schema_version": _SCHEMA_VERSION,
                "include_descriptions": json.dumps(include_descriptions),
                "chunk_count": str(len(rows)),
                "source_fingerprint": source_fingerprint,
                # Empty for a BM25-only build. A later dense search must not
                # reuse a FAISS file created from a different chunk snapshot.
                "dense_source_fingerprint": dense_source_fingerprint,
            }
            connection.executemany(
                """
                INSERT INTO lexical_settings (key, value) VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                settings.items(),
            )
        self._vocabulary_cache = None
        return len(rows)

    def vocabulary_stats(self) -> dict[str, int]:
        """Return source-only FTS terms and their document frequencies.

        The temporary vocabulary table reads the existing BM25 index. It does
        not change the index and never includes benchmark questions or answers.
        """

        if self._vocabulary_cache is not None:
            return dict(self._vocabulary_cache)
        if not self.path.exists():
            return {}
        with self._connect() as connection:
            self._create_schema(connection)
            connection.execute(
                "CREATE VIRTUAL TABLE temp.query_vocabulary "
                "USING fts5vocab('main', 'lexical_fts', 'row')"
            )
            rows = connection.execute(
                "SELECT term, doc FROM temp.query_vocabulary ORDER BY term"
            ).fetchall()
        self._vocabulary_cache = {str(row[0]): int(row[1]) for row in rows}
        return dict(self._vocabulary_cache)

    def vocabulary(self) -> frozenset[str]:
        """Return source-only FTS terms without frequency metadata."""

        return frozenset(self.vocabulary_stats())

    def source_fingerprint(self) -> str:
        """Return the chunk snapshot hash recorded when the index was built."""

        if not self.path.exists():
            return ""
        with self._connect() as connection:
            self._create_schema(connection)
            row = connection.execute(
                "SELECT value FROM lexical_settings WHERE key = 'source_fingerprint'"
            ).fetchone()
        return "" if row is None else str(row[0])

    def dense_source_fingerprint(self) -> str:
        """Return the child-chunk fingerprint paired with the FAISS index."""

        if not self.path.exists():
            return ""
        with self._connect() as connection:
            self._create_schema(connection)
            row = connection.execute(
                "SELECT value FROM lexical_settings "
                "WHERE key = 'dense_source_fingerprint'"
            ).fetchone()
        return "" if row is None else str(row[0])

    @staticmethod
    def _safe_match_query(text: str, *, max_terms: int = 128) -> str:
        """Convert user text to a literal-token FTS query.

        Passing a question directly to ``MATCH`` would interpret punctuation
        such as quotes, colons, or minus signs as FTS operators.  Literal
        quoted tokens avoid syntax errors and prevent query-language injection.
        Terms use OR semantics because financial questions often contain a few
        very selective company names, years, and line-item labels.
        """

        tokens = _QUERY_TOKEN_RE.findall(text.casefold())[:max_terms]
        return " OR ".join(f'"{token.replace(chr(34), chr(34) * 2)}"' for token in tokens)

    @staticmethod
    def _row_to_hit(
        row: sqlite3.Row,
        *,
        score: float,
        rank: int,
        retrieval_source: str,
        component_scores: dict[str, float] | None = None,
    ) -> SearchHit:
        """Hydrate a shared ``SearchHit`` from stored source metadata."""

        metadata = json.loads(row["metadata_json"])
        if not isinstance(metadata, dict):
            raise ValueError(f"Stored metadata for chunk {row['chunk_id']} is not an object.")
        return SearchHit(
            chunk_id=int(row["chunk_id"]),
            context_uid=str(row["context_uid"]),
            document_id=str(row["document_id"]),
            source_text=str(row["source_text"]),
            score=float(score),
            rank=rank,
            retrieval_source=retrieval_source,
            dataset=str(row["dataset"]),
            split=str(row["split"]),
            page_number=row["page_number"],
            alias_path=str(row["alias_path"]),
            component_scores=dict(component_scores or {}),
            metadata=metadata,
        )

    def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        dataset: str | None = None,
        split: str | None = None,
    ) -> list[SearchHit]:
        """Return top lexical hits, optionally restricted by dataset and split."""

        if top_k <= 0 or not query.strip():
            return []
        match_query = self._safe_match_query(query)
        if not match_query:
            return []

        conditions = ["lexical_fts MATCH ?"]
        parameters: list[object] = [match_query]
        if dataset is not None:
            conditions.append("metadata.dataset = ? COLLATE NOCASE")
            parameters.append(dataset)
        if split is not None:
            conditions.append("metadata.split = ? COLLATE NOCASE")
            parameters.append(split)
        parameters.append(top_k)

        with self._connect() as connection:
            self._create_schema(connection)
            rows = connection.execute(
                f"""
                SELECT
                    metadata.*,
                    bm25(lexical_fts) AS raw_bm25
                FROM lexical_fts
                JOIN lexical_metadata AS metadata
                  ON metadata.chunk_id = lexical_fts.chunk_id
                WHERE {' AND '.join(conditions)}
                ORDER BY raw_bm25 ASC, metadata.chunk_id ASC
                LIMIT ?
                """,
                parameters,
            ).fetchall()

        hits: list[SearchHit] = []
        for rank, row in enumerate(rows, start=1):
            raw_bm25 = float(row["raw_bm25"])
            score = -raw_bm25
            hits.append(
                self._row_to_hit(
                    row,
                    score=score,
                    rank=rank,
                    retrieval_source="bm25",
                    component_scores={"bm25": score, "fts5_raw_bm25": raw_bm25},
                )
            )
        return hits

    def hydrate(
        self,
        chunk_scores: Sequence[tuple[int, float]],
        *,
        retrieval_source: str,
    ) -> list[SearchHit]:
        """Attach source metadata to ranked IDs returned by a vector index.

        The order of ``chunk_scores`` is preserved.  Missing IDs indicate that
        the lexical and dense indexes were built from different chunk sets, so
        the method fails instead of silently degrading evaluation results.
        """

        if not chunk_scores:
            return []
        ids = [int(chunk_id) for chunk_id, _ in chunk_scores]
        placeholders = ",".join("?" for _ in ids)
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM lexical_metadata WHERE chunk_id IN ({placeholders})",
                ids,
            ).fetchall()
        by_id = {int(row["chunk_id"]): row for row in rows}
        missing = [chunk_id for chunk_id in ids if chunk_id not in by_id]
        if missing:
            raise ValueError(
                "Dense index contains chunk IDs missing from the lexical metadata: "
                f"{missing[:10]}"
            )

        return [
            self._row_to_hit(
                by_id[chunk_id],
                score=float(score),
                rank=rank,
                retrieval_source=retrieval_source,
                component_scores={retrieval_source: float(score)},
            )
            for rank, (chunk_id, score) in enumerate(chunk_scores, start=1)
        ]

    def count(self) -> int:
        """Return the number of indexed chunks."""

        if not self.path.exists():
            return 0
        with self._connect() as connection:
            self._create_schema(connection)
            row = connection.execute("SELECT COUNT(*) FROM lexical_metadata").fetchone()
        return int(row[0])

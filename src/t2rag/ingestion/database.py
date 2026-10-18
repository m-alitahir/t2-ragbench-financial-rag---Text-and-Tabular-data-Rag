"""SQLite schema and writes for the canonical document registry.

All writes use explicit columns and JSON serialization.  Source evidence,
generated descriptions, and aliases are kept in separate tables so downstream
retrieval and citation code can enforce provenance.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from t2rag.models import ChunkRecord

from .metadata import NormalizedContext, assert_no_forbidden_fields


SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS physical_documents (
    document_id TEXT PRIMARY KEY,
    byte_size INTEGER NOT NULL CHECK (byte_size >= 0),
    page_count INTEGER NOT NULL CHECK (page_count > 0),
    media_type TEXT NOT NULL DEFAULT 'application/pdf',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CHECK (length(document_id) = 64)
);

CREATE TABLE IF NOT EXISTS document_aliases (
    alias_id INTEGER PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES physical_documents(document_id),
    dataset TEXT NOT NULL,
    split TEXT NOT NULL,
    relative_path TEXT NOT NULL UNIQUE,
    alias_kind TEXT NOT NULL CHECK (alias_kind IN ('pdf', 'raw_json', 'png')),
    page_number INTEGER,
    content_sha256 TEXT,
    is_distractor INTEGER NOT NULL DEFAULT 0 CHECK (is_distractor IN (0, 1)),
    metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS aliases_document_idx
    ON document_aliases(document_id);
CREATE INDEX IF NOT EXISTS aliases_dataset_split_idx
    ON document_aliases(dataset, split, alias_kind);

CREATE TABLE IF NOT EXISTS context_records (
    context_uid TEXT PRIMARY KEY,
    context_id TEXT NOT NULL,
    document_id TEXT NOT NULL REFERENCES physical_documents(document_id),
    alias_id INTEGER NOT NULL REFERENCES document_aliases(alias_id),
    dataset TEXT NOT NULL,
    split TEXT NOT NULL,
    page_number INTEGER,
    source_text TEXT NOT NULL CHECK (length(trim(source_text)) > 0),
    table_text TEXT NOT NULL DEFAULT '',
    pre_text TEXT NOT NULL DEFAULT '',
    post_text TEXT NOT NULL DEFAULT '',
    metadata_json TEXT NOT NULL DEFAULT '{}',
    UNIQUE(dataset, context_id)
);

CREATE INDEX IF NOT EXISTS contexts_document_idx
    ON context_records(document_id);
CREATE INDEX IF NOT EXISTS contexts_alias_idx
    ON context_records(alias_id);
CREATE INDEX IF NOT EXISTS contexts_dataset_split_idx
    ON context_records(dataset, split);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id INTEGER PRIMARY KEY,
    context_uid TEXT NOT NULL REFERENCES context_records(context_uid) ON DELETE CASCADE,
    document_id TEXT NOT NULL REFERENCES physical_documents(document_id),
    alias_id INTEGER NOT NULL REFERENCES document_aliases(alias_id),
    dataset TEXT NOT NULL,
    split TEXT NOT NULL,
    kind TEXT NOT NULL,
    page_number INTEGER,
    position INTEGER NOT NULL CHECK (position >= 0),
    source_text TEXT NOT NULL CHECK (length(trim(source_text)) > 0),
    description_text TEXT NOT NULL DEFAULT '',
    metadata_json TEXT NOT NULL DEFAULT '{}',
    UNIQUE(context_uid, kind, position)
);

CREATE INDEX IF NOT EXISTS chunks_document_idx ON chunks(document_id);
CREATE INDEX IF NOT EXISTS chunks_context_idx ON chunks(context_uid);
CREATE INDEX IF NOT EXISTS chunks_dataset_split_idx ON chunks(dataset, split);

-- Only source evidence enters the lexical index.  Generated descriptions stay
-- outside FTS unless a later, explicit experiment builds a separate index.
CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(
    source_text,
    content='chunks',
    content_rowid='chunk_id',
    tokenize='unicode61 remove_diacritics 2'
);

CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunk_fts(rowid, source_text) VALUES (new.chunk_id, new.source_text);
END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunk_fts(chunk_fts, rowid, source_text)
    VALUES ('delete', old.chunk_id, old.source_text);
END;
CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE OF source_text ON chunks BEGIN
    INSERT INTO chunk_fts(chunk_fts, rowid, source_text)
    VALUES ('delete', old.chunk_id, old.source_text);
    INSERT INTO chunk_fts(rowid, source_text) VALUES (new.chunk_id, new.source_text);
END;

CREATE TABLE IF NOT EXISTS tat_page_blocks (
    block_id INTEGER PRIMARY KEY,
    alias_id INTEGER NOT NULL REFERENCES document_aliases(alias_id) ON DELETE CASCADE,
    page_number INTEGER NOT NULL CHECK (page_number > 0),
    block_order INTEGER NOT NULL CHECK (block_order >= 0),
    block_uuid TEXT,
    bbox_json TEXT NOT NULL,
    text TEXT NOT NULL,
    words_json TEXT NOT NULL DEFAULT '{}',
    UNIQUE(alias_id, page_number, block_order)
);

CREATE INDEX IF NOT EXISTS tat_blocks_alias_page_idx
    ON tat_page_blocks(alias_id, page_number, block_order);

CREATE TABLE IF NOT EXISTS document_descriptions (
    description_id INTEGER PRIMARY KEY,
    subject_key TEXT NOT NULL,
    document_id TEXT NOT NULL REFERENCES physical_documents(document_id),
    context_uid TEXT REFERENCES context_records(context_uid) ON DELETE CASCADE,
    alias_id INTEGER REFERENCES document_aliases(alias_id) ON DELETE CASCADE,
    page_number INTEGER,
    modality TEXT NOT NULL CHECK (modality IN ('text', 'vision')),
    model TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    input_sha256 TEXT NOT NULL,
    description_text TEXT NOT NULL CHECK (length(trim(description_text)) > 0),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(subject_key, modality, model, prompt_version)
);

-- Raw-PDF extraction is an optional comparison corpus. It never overwrites
-- the benchmark-provided contexts used by the baseline.
CREATE TABLE IF NOT EXISTS raw_pdf_pages (
    document_id TEXT NOT NULL REFERENCES physical_documents(document_id),
    internal_page_number INTEGER NOT NULL CHECK (internal_page_number > 0),
    extraction_method TEXT NOT NULL,
    source_text TEXT NOT NULL DEFAULT '',
    rendered_image_path TEXT,
    width_points REAL,
    height_points REAL,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY(document_id, internal_page_number)
);

CREATE TABLE IF NOT EXISTS raw_pdf_tables (
    document_id TEXT NOT NULL,
    internal_page_number INTEGER NOT NULL,
    table_index INTEGER NOT NULL CHECK (table_index >= 0),
    markdown_text TEXT NOT NULL,
    extraction_method TEXT NOT NULL,
    PRIMARY KEY(document_id, internal_page_number, table_index),
    FOREIGN KEY(document_id, internal_page_number)
      REFERENCES raw_pdf_pages(document_id, internal_page_number)
      ON DELETE CASCADE
);
"""


def _json(value: Mapping[str, Any] | Sequence[Any]) -> str:
    """Serialize deterministic JSON after applying the leakage guard."""

    assert_no_forbidden_fields(value, location="database JSON")
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class RegistryDatabase:
    """Small repository wrapper around one SQLite registry connection."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.execute("PRAGMA journal_mode = WAL")
        self.connection.execute("PRAGMA synchronous = NORMAL")

    def close(self) -> None:
        """Close the underlying SQLite connection."""

        self.connection.close()

    def __enter__(self) -> RegistryDatabase:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Commit a logical ingestion unit or roll it back on failure."""

        try:
            yield self.connection
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise

    def initialize(self) -> None:
        """Create all registry tables, indexes, and FTS synchronization triggers."""

        with self.transaction():
            self.connection.executescript(SCHEMA_SQL)

    def upsert_document(self, document_id: str, byte_size: int, page_count: int) -> None:
        """Insert one canonical PDF, verifying immutable properties on reuse."""

        existing = self.connection.execute(
            "SELECT byte_size, page_count FROM physical_documents WHERE document_id = ?",
            (document_id,),
        ).fetchone()
        if existing is not None:
            if existing["byte_size"] != byte_size or existing["page_count"] != page_count:
                raise ValueError(f"Conflicting properties for canonical document {document_id}.")
            return
        self.connection.execute(
            """
            INSERT INTO physical_documents(document_id, byte_size, page_count)
            VALUES (?, ?, ?)
            """,
            (document_id, byte_size, page_count),
        )

    def upsert_alias(
        self,
        *,
        document_id: str,
        dataset: str,
        split: str,
        relative_path: str,
        alias_kind: str,
        page_number: int | None = None,
        content_sha256: str | None = None,
        is_distractor: bool = False,
        metadata: Mapping[str, Any] | None = None,
    ) -> int:
        """Insert an alias and return its stable integer identifier."""

        metadata_json = _json(dict(metadata or {}))
        self.connection.execute(
            """
            INSERT INTO document_aliases(
                document_id, dataset, split, relative_path, alias_kind,
                page_number, content_sha256, is_distractor, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(relative_path) DO UPDATE SET
                is_distractor = MIN(document_aliases.is_distractor, excluded.is_distractor)
            """,
            (
                document_id,
                dataset,
                split,
                relative_path,
                alias_kind,
                page_number,
                content_sha256,
                int(is_distractor),
                metadata_json,
            ),
        )
        row = self.connection.execute(
            """
            SELECT alias_id, document_id, alias_kind
            FROM document_aliases WHERE relative_path = ?
            """,
            (relative_path,),
        ).fetchone()
        if row is None or row["document_id"] != document_id or row["alias_kind"] != alias_kind:
            raise ValueError(f"Alias conflict for {relative_path}.")
        return int(row["alias_id"])

    def alias_by_path(self, relative_path: str) -> sqlite3.Row | None:
        """Look up an alias without exposing a mutable cursor."""

        return self.connection.execute(
            "SELECT * FROM document_aliases WHERE relative_path = ?", (relative_path,)
        ).fetchone()

    def alias_has_context(self, alias_id: int) -> bool:
        """Return whether at least one normalized context cites this alias."""

        row = self.connection.execute(
            "SELECT 1 FROM context_records WHERE alias_id = ? LIMIT 1", (alias_id,)
        ).fetchone()
        return row is not None

    def upsert_context(
        self,
        context: NormalizedContext,
        *,
        document_id: str,
        alias_id: int,
    ) -> bool:
        """Insert a unique source context; return ``True`` only on first insert."""

        assert_no_forbidden_fields(context.metadata, location="context metadata")
        existing = self.connection.execute(
            """
            SELECT document_id, alias_id, source_text, table_text, pre_text, post_text
            FROM context_records WHERE context_uid = ?
            """,
            (context.context_uid,),
        ).fetchone()
        values = (
            document_id,
            alias_id,
            context.source_text,
            context.table_text,
            context.pre_text,
            context.post_text,
        )
        if existing is not None:
            observed = tuple(existing[key] for key in existing.keys())
            if observed != values:
                raise ValueError(f"Conflicting duplicate context {context.context_uid}.")
            return False

        self.connection.execute(
            """
            INSERT INTO context_records(
                context_uid, context_id, document_id, alias_id, dataset, split,
                page_number, source_text, table_text, pre_text, post_text, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                context.context_uid,
                context.context_id,
                document_id,
                alias_id,
                context.dataset,
                context.split,
                context.page_number,
                context.source_text,
                context.table_text,
                context.pre_text,
                context.post_text,
                _json(context.metadata),
            ),
        )
        return True

    def insert_chunk(self, chunk: ChunkRecord, *, position: int) -> int:
        """Insert one source chunk and return its SQLite identifier."""

        if chunk.alias_id is None:
            raise ValueError("A persisted chunk requires alias_id.")
        assert_no_forbidden_fields(chunk.metadata, location="chunk metadata")
        self.connection.execute(
            """
            INSERT INTO chunks(
                context_uid, document_id, alias_id, dataset, split, kind,
                page_number, position, source_text, description_text, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(context_uid, kind, position) DO UPDATE SET
                source_text = excluded.source_text,
                metadata_json = excluded.metadata_json
            """,
            (
                chunk.context_uid,
                chunk.document_id,
                chunk.alias_id,
                chunk.dataset,
                chunk.split,
                chunk.kind,
                chunk.page_number,
                position,
                chunk.source_text,
                "",  # Generated text never enters the source chunk during ingestion.
                _json(chunk.metadata),
            ),
        )
        row = self.connection.execute(
            "SELECT chunk_id FROM chunks WHERE context_uid = ? AND kind = ? AND position = ?",
            (chunk.context_uid, chunk.kind, position),
        ).fetchone()
        if row is None:
            raise RuntimeError("Chunk insert did not return a row.")
        return int(row["chunk_id"])

    def insert_tat_block(
        self,
        *,
        alias_id: int,
        page_number: int,
        block_order: int,
        block_uuid: str | None,
        bbox: Sequence[Any],
        text: str,
        words: Mapping[str, Any],
    ) -> None:
        """Persist one ordered TAT-DQA layout block."""

        self.connection.execute(
            """
            INSERT INTO tat_page_blocks(
                alias_id, page_number, block_order, block_uuid,
                bbox_json, text, words_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(alias_id, page_number, block_order) DO UPDATE SET
                block_uuid = excluded.block_uuid,
                bbox_json = excluded.bbox_json,
                text = excluded.text,
                words_json = excluded.words_json
            """,
            (
                alias_id,
                page_number,
                block_order,
                block_uuid,
                _json(list(bbox)),
                text,
                _json(dict(words)),
            ),
        )

    def iter_chunks(self, *, include_descriptions: bool = False) -> Iterator[ChunkRecord]:
        """Yield persisted chunks in stable ID order for retrieval index builds.

        Generated descriptions are attached only when explicitly requested.
        They remain distinguishable through ``ChunkRecord.description_text`` and
        never replace the source evidence.
        """

        context_descriptions: dict[str, list[str]] = {}
        page_descriptions: dict[tuple[str, int], list[str]] = {}
        if include_descriptions:
            for row in self.connection.execute(
                """
                SELECT document_id, context_uid, alias_id, page_number, description_text
                FROM document_descriptions ORDER BY description_id
                """
            ):
                if row["context_uid"] is not None:
                    context_descriptions.setdefault(row["context_uid"], []).append(
                        row["description_text"]
                    )
                elif row["alias_id"] is not None and row["page_number"] is not None:
                    key = (str(row["document_id"]), int(row["page_number"]))
                    page_descriptions.setdefault(key, []).append(row["description_text"])

        rows = self.connection.execute("SELECT * FROM chunks ORDER BY chunk_id")
        for row in rows:
            metadata = json.loads(row["metadata_json"])
            if not isinstance(metadata, dict):
                raise ValueError(f"Chunk {row['chunk_id']} metadata is not an object.")
            descriptions = context_descriptions.get(row["context_uid"], [])
            if row["page_number"] is not None:
                descriptions = [
                    *descriptions,
                    *page_descriptions.get(
                        (str(row["document_id"]), int(row["page_number"])), []
                    ),
                ]
            yield ChunkRecord(
                chunk_id=int(row["chunk_id"]),
                context_uid=row["context_uid"],
                document_id=row["document_id"],
                alias_id=int(row["alias_id"]),
                dataset=row["dataset"],
                split=row["split"],
                kind=row["kind"],
                page_number=row["page_number"],
                source_text=row["source_text"],
                description_text="\n\n".join(descriptions),
                metadata=metadata,
            )

    def store_description(
        self,
        *,
        document_id: str,
        context_uid: str | None,
        alias_id: int | None,
        page_number: int | None,
        modality: str,
        model: str,
        prompt_version: str,
        input_sha256: str,
        description_text: str,
    ) -> int:
        """Store generated text separately from source chunks and FTS."""

        if not description_text.strip():
            raise ValueError("A generated description cannot be empty.")
        if context_uid is None and (alias_id is None or page_number is None):
            raise ValueError("A page description requires alias_id and page_number.")
        # SQLite considers NULL values distinct in UNIQUE constraints.  A
        # non-null subject key gives context and page descriptions idempotent
        # identities even though each uses different nullable foreign keys.
        subject_key = (
            f"context:{context_uid}"
            if context_uid is not None
            else f"page:{alias_id}:{page_number}"
        )
        self.connection.execute(
            """
            INSERT INTO document_descriptions(
                subject_key, document_id, context_uid, alias_id, page_number, modality,
                model, prompt_version, input_sha256, description_text
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(subject_key, modality, model, prompt_version) DO UPDATE SET
                input_sha256 = excluded.input_sha256,
                description_text = excluded.description_text,
                created_at = CURRENT_TIMESTAMP
            """,
            (
                subject_key,
                document_id,
                context_uid,
                alias_id,
                page_number,
                modality,
                model,
                prompt_version,
                input_sha256,
                description_text.strip(),
            ),
        )
        row = self.connection.execute(
            """
            SELECT description_id FROM document_descriptions
            WHERE subject_key = ? AND modality = ? AND model = ? AND prompt_version = ?
            """,
            (
                subject_key,
                modality,
                model,
                prompt_version,
            ),
        ).fetchone()
        if row is None:
            raise RuntimeError("Description insert did not return a row.")
        return int(row["description_id"])

    def iter_text_description_inputs(
        self,
        *,
        model: str,
        prompt_version: str,
        limit: int | None = None,
    ) -> Iterator[sqlite3.Row]:
        """Yield contexts that do not yet have the requested text description."""

        sql = """
            SELECT c.context_uid, c.document_id, c.alias_id, c.source_text
            FROM context_records AS c
            LEFT JOIN document_descriptions AS d
              ON d.subject_key = 'context:' || c.context_uid
             AND d.modality = 'text'
             AND d.model = ?
             AND d.prompt_version = ?
            WHERE d.description_id IS NULL
            ORDER BY c.context_uid
        """
        parameters: list[Any] = [model, prompt_version]
        if limit is not None:
            if limit <= 0:
                return
            sql += " LIMIT ?"
            parameters.append(limit)
        yield from self.connection.execute(sql, parameters)

    def iter_vision_description_inputs(
        self,
        *,
        model: str,
        prompt_version: str,
        limit: int | None = None,
    ) -> Iterator[sqlite3.Row]:
        """Yield TAT page PNG aliases without the requested visual description."""

        sql = """
            SELECT a.document_id, a.alias_id, a.page_number, a.relative_path
            FROM document_aliases AS a
            LEFT JOIN document_descriptions AS d
              ON d.subject_key = 'page:' || a.alias_id || ':' || a.page_number
             AND d.modality = 'vision'
             AND d.model = ?
             AND d.prompt_version = ?
            WHERE a.alias_kind = 'png'
              AND a.page_number IS NOT NULL
              AND d.description_id IS NULL
            ORDER BY a.dataset, a.split, a.relative_path
        """
        parameters: list[Any] = [model, prompt_version]
        if limit is not None:
            if limit <= 0:
                return
            sql += " LIMIT ?"
            parameters.append(limit)
        yield from self.connection.execute(sql, parameters)

    def iter_rendered_vision_inputs(
        self,
        *,
        model: str,
        prompt_version: str,
        limit: int | None = None,
    ) -> Iterator[sqlite3.Row]:
        """Yield rendered raw pages when the dataset has no supplied page PNG."""

        sql = """
            SELECT
                p.document_id,
                MIN(pdf_alias.alias_id) AS alias_id,
                COALESCE(pdf_alias.page_number, p.internal_page_number) AS page_number,
                p.rendered_image_path AS image_path
            FROM raw_pdf_pages AS p
            JOIN document_aliases AS pdf_alias
              ON pdf_alias.document_id = p.document_id
             AND pdf_alias.alias_kind = 'pdf'
            LEFT JOIN document_descriptions AS d
              ON d.subject_key = (
                    'page:' || pdf_alias.alias_id || ':' ||
                    COALESCE(pdf_alias.page_number, p.internal_page_number)
                 )
             AND d.modality = 'vision'
             AND d.model = ?
             AND d.prompt_version = ?
            WHERE p.rendered_image_path IS NOT NULL
              AND d.description_id IS NULL
              AND NOT EXISTS (
                    SELECT 1 FROM document_aliases AS png_alias
                    WHERE png_alias.document_id = p.document_id
                      AND png_alias.alias_kind = 'png'
                      AND png_alias.page_number = p.internal_page_number
              )
            GROUP BY
                p.document_id,
                p.internal_page_number,
                COALESCE(pdf_alias.page_number, p.internal_page_number),
                p.rendered_image_path
            ORDER BY p.document_id, p.internal_page_number
        """
        parameters: list[Any] = [model, prompt_version]
        if limit is not None:
            if limit <= 0:
                return
            sql += " LIMIT ?"
            parameters.append(limit)
        yield from self.connection.execute(sql, parameters)

    def summary(self) -> dict[str, int]:
        """Return small registry counts for status output and manual checks."""

        tables = {
            "physical_documents": "physical_documents",
            "aliases": "document_aliases",
            "contexts": "context_records",
            "chunks": "chunks",
            "tat_blocks": "tat_page_blocks",
            "descriptions": "document_descriptions",
            "raw_pdf_pages": "raw_pdf_pages",
            "raw_pdf_tables": "raw_pdf_tables",
        }
        return {
            name: int(self.connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            for name, table in tables.items()
        }

    def iter_pdf_documents(self, *, limit: int | None = None) -> Iterator[sqlite3.Row]:
        """Yield one stable source PDF alias for each canonical document."""

        sql = """
            SELECT d.document_id, MIN(a.relative_path) AS relative_path
            FROM physical_documents AS d
            JOIN document_aliases AS a ON a.document_id = d.document_id
            WHERE a.alias_kind = 'pdf'
            GROUP BY d.document_id
            ORDER BY d.document_id
        """
        parameters: list[Any] = []
        if limit is not None:
            if limit <= 0:
                return
            sql += " LIMIT ?"
            parameters.append(limit)
        yield from self.connection.execute(sql, parameters)

    def tat_layout_pages(self, document_id: str) -> dict[int, str]:
        """Return ordered TAT block text when structured layout data exists."""

        rows = self.connection.execute(
            """
            SELECT b.page_number, b.text
            FROM tat_page_blocks AS b
            JOIN document_aliases AS a ON a.alias_id = b.alias_id
            WHERE a.document_id = ?
            ORDER BY b.page_number, b.block_order
            """,
            (document_id,),
        )
        pages: dict[int, list[str]] = {}
        for row in rows:
            text = str(row["text"]).strip()
            if text:
                pages.setdefault(int(row["page_number"]), []).append(text)
        return {page: "\n".join(blocks) for page, blocks in pages.items()}

    def upsert_raw_page(
        self,
        *,
        document_id: str,
        internal_page_number: int,
        extraction_method: str,
        source_text: str,
        rendered_image_path: str | None,
        width_points: float | None,
        height_points: float | None,
    ) -> None:
        """Store optional raw extraction without modifying benchmark contexts."""

        self.connection.execute(
            """
            INSERT INTO raw_pdf_pages(
                document_id, internal_page_number, extraction_method, source_text,
                rendered_image_path, width_points, height_points
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(document_id, internal_page_number) DO UPDATE SET
                extraction_method = excluded.extraction_method,
                source_text = excluded.source_text,
                rendered_image_path = excluded.rendered_image_path,
                width_points = excluded.width_points,
                height_points = excluded.height_points,
                updated_at = CURRENT_TIMESTAMP
            """,
            (
                document_id,
                internal_page_number,
                extraction_method,
                source_text,
                rendered_image_path,
                width_points,
                height_points,
            ),
        )

    def replace_raw_tables(
        self,
        *,
        document_id: str,
        internal_page_number: int,
        tables: Sequence[str],
        extraction_method: str,
    ) -> None:
        """Replace extracted tables for one page as one transaction unit."""

        self.connection.execute(
            """
            DELETE FROM raw_pdf_tables
            WHERE document_id = ? AND internal_page_number = ?
            """,
            (document_id, internal_page_number),
        )
        self.connection.executemany(
            """
            INSERT INTO raw_pdf_tables(
                document_id, internal_page_number, table_index,
                markdown_text, extraction_method
            ) VALUES (?, ?, ?, ?, ?)
            """,
            [
                (
                    document_id,
                    internal_page_number,
                    table_index,
                    markdown,
                    extraction_method,
                )
                for table_index, markdown in enumerate(tables)
            ],
        )

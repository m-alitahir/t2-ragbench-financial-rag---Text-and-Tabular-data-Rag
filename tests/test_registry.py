"""Contract tests for canonical, leakage-safe registry ingestion."""

from __future__ import annotations

import json
from pathlib import Path

from pypdf import PdfWriter

from t2rag.ingestion.database import RegistryDatabase
from t2rag.ingestion.metadata import MetadataFile
from t2rag.ingestion.registry import RegistryBuilder


def _pdf(path: Path) -> None:
    """Create a small valid PDF using an installed runtime dependency."""

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    with path.open("wb") as handle:
        writer.write(handle)


def _jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row) + "\n" for row in rows)
    path.write_text(payload, encoding="utf-8")


def _row(
    *,
    context_id: str,
    split: str,
    source_file: str,
    context: str = "Revenue increased from $10 million to $12 million.",
) -> dict[str, object]:
    return {
        "id": f"qa-{context_id}",
        "context_id": context_id,
        "split": split,
        "question": "SENTINEL QUESTION MUST NOT BE STORED",
        "program_answer": "SENTINEL PROGRAM ANSWER",
        "original_answer": "SENTINEL ORIGINAL ANSWER",
        "context": context,
        "table": "| Year | Revenue |\n| --- | --- |\n| 2020 | $12 million |",
        "pre_text": "Revenue increased.",
        "post_text": "The company reported growth.",
        "file_name": source_file,
        "company_name": "Example Corp",
        "report_year": 2020,
        "page_number": 7,
    }


def test_finqa_convfinqa_pdf_dedup_and_label_exclusion(tmp_path: Path) -> None:
    """Byte-identical FinQA aliases share one physical document and no labels."""

    repository = tmp_path / "dataset"
    fin_pdf = repository / "data/FinQA/train/pdf/ABC/2020/page_7.pdf"
    conv_pdf = repository / "data/ConvFinQA/pdf/ABC/2020/page_7.pdf"
    _pdf(fin_pdf)
    conv_pdf.parent.mkdir(parents=True, exist_ok=True)
    conv_pdf.write_bytes(fin_pdf.read_bytes())

    fin_metadata = repository / "data/FinQA/train/metadata.jsonl"
    conv_metadata = repository / "data/ConvFinQA/turn_0.jsonl"
    _jsonl(
        fin_metadata,
        [
            _row(
                context_id="finqa_ctx_1",
                split="train",
                source_file="pdf/ABC/2020/page_7.pdf",
            )
        ],
    )
    _jsonl(
        conv_metadata,
        [
            _row(
                context_id="convfinqa_ctx_1",
                split="all",
                source_file="pdf/ABC/2020/page_7.pdf",
            )
        ],
    )

    with RegistryDatabase(tmp_path / "registry.sqlite3") as database:
        database.initialize()
        builder = RegistryBuilder(dataset_root=repository, database=database)
        builder.ingest_metadata_file(MetadataFile(fin_metadata, "FinQA", "train"))
        builder.ingest_metadata_file(MetadataFile(conv_metadata, "ConvFinQA", "all"))

        assert database.connection.execute(
            "SELECT count(*) FROM physical_documents"
        ).fetchone()[0] == 1
        assert database.connection.execute(
            "SELECT count(*) FROM document_aliases WHERE alias_kind = 'pdf'"
        ).fetchone()[0] == 2
        assert database.connection.execute(
            "SELECT count(*) FROM context_records"
        ).fetchone()[0] == 2
        loaded_chunks = list(database.iter_chunks())
        assert loaded_chunks
        assert all(chunk.chunk_id is not None for chunk in loaded_chunks)
        assert {chunk.metadata["alias_path"] for chunk in loaded_chunks} == {
            "pdf/ABC/2020/page_7.pdf"
        }

        # Search every persisted text/JSON column for unique sealed sentinels.
        persisted = "\n".join(
            str(value)
            for table, columns in (
                ("context_records", "source_text, table_text, pre_text, post_text, metadata_json"),
                ("chunks", "source_text, description_text, metadata_json"),
                ("document_aliases", "metadata_json"),
            )
            for row in database.connection.execute(f"SELECT {columns} FROM {table}")
            for value in row
        )
        assert "SENTINEL QUESTION" not in persisted
        assert "SENTINEL PROGRAM ANSWER" not in persisted
        assert "SENTINEL ORIGINAL ANSWER" not in persisted


def test_tat_assets_ordered_blocks_png_aliases_and_distractor(tmp_path: Path) -> None:
    """TAT raw assets remain aliases and unreferenced train PDFs are marked."""

    repository = tmp_path / "dataset"
    raw_dir = repository / "data/TAT-DQA/train/raw"
    referenced_pdf = raw_dir / "referenced.pdf"
    distractor_pdf = raw_dir / "distractor.pdf"
    _pdf(referenced_pdf)
    _pdf(distractor_pdf)

    layout = {
        "pages": [
            {
                "bbox": [0, 0, 100, 100],
                "blocks": [
                    {
                        "bbox": [0, 20, 20, 30],
                        "uuid": "second",
                        "text": "Second block",
                        "words": {"word_list": ["Second", "block"]},
                        "order": 2,
                    },
                    {
                        "bbox": [0, 0, 20, 10],
                        "uuid": "first",
                        "text": "First block",
                        "words": {"word_list": ["First", "block"]},
                        "order": 1,
                    },
                ],
            }
        ]
    }
    for stem in ("referenced", "distractor"):
        (raw_dir / f"{stem}.json").write_text(json.dumps(layout), encoding="utf-8")
        (raw_dir / f"{stem}_1.png").write_bytes(b"not-decoded-during-ingestion")

    metadata = repository / "data/TAT-DQA/train/metadata.jsonl"
    _jsonl(
        metadata,
        [
            _row(
                context_id="referenced",
                split="train",
                source_file="raw/referenced.pdf",
            )
        ],
    )

    with RegistryDatabase(tmp_path / "registry.sqlite3") as database:
        database.initialize()
        builder = RegistryBuilder(dataset_root=repository, database=database)
        builder.ingest_metadata_file(MetadataFile(metadata, "TAT-DQA", "train"))
        builder.register_tat_assets()

        distractors = database.connection.execute(
            """
            SELECT relative_path FROM document_aliases
            WHERE alias_kind = 'pdf' AND is_distractor = 1
            """
        ).fetchall()
        assert [Path(row[0]).name for row in distractors] == ["distractor.pdf"]
        assert database.connection.execute(
            "SELECT count(*) FROM document_aliases WHERE alias_kind = 'raw_json'"
        ).fetchone()[0] == 2
        assert database.connection.execute(
            "SELECT count(*) FROM document_aliases WHERE alias_kind = 'png'"
        ).fetchone()[0] == 2

        first_blocks = database.connection.execute(
            """
            SELECT text FROM tat_page_blocks
            WHERE alias_id = (
                SELECT alias_id FROM document_aliases
                WHERE relative_path LIKE '%/referenced.pdf'
            )
            ORDER BY page_number, block_order
            """
        ).fetchall()
        assert [row[0] for row in first_blocks] == ["First block", "Second block"]


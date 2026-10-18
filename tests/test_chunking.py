"""Tests for source-only, table-aware chunk construction."""

from __future__ import annotations

import pytest

from t2rag.errors import LeakageError
from t2rag.ingestion.chunking import build_context_chunks
from t2rag.ingestion.metadata import NormalizedContext, assert_no_forbidden_fields


def _context(**changes: object) -> NormalizedContext:
    values: dict[str, object] = {
        "context_uid": "finqa:ctx-1",
        "context_id": "ctx-1",
        "dataset": "FinQA",
        "split": "train",
        "source_file": "pdf/ABC/2020/page_1.pdf",
        "source_text": "Narrative before.\n\n| Year | Revenue |\n| --- | --- |\n| 2020 | 12 |",
        "table_text": "| Year | Revenue |\n| --- | --- |\n| 2020 | 12 |",
        "pre_text": "Narrative before.",
        "post_text": "Narrative after.",
        "page_number": 1,
        "metadata": {"company_name": "Example Corp"},
    }
    values.update(changes)
    return NormalizedContext(**values)  # type: ignore[arg-type]


def test_builds_whole_context_and_preserves_table_rows() -> None:
    chunks = build_context_chunks(
        _context(),
        document_id="a" * 64,
        alias_id=3,
        max_words=20,
        overlap_words=4,
    )

    assert chunks[0].kind == "whole_context"
    assert chunks[0].source_text.startswith("Narrative before")
    tables = [chunk for chunk in chunks if chunk.kind == "table"]
    narratives = [chunk for chunk in chunks if chunk.kind == "narrative"]
    assert len(tables) == 1
    assert "| --- | --- |" in tables[0].source_text
    assert "| 2020 | 12 |" in tables[0].source_text
    assert {chunk.metadata["source_component"] for chunk in narratives} == {
        "pre_text",
        "post_text",
    }
    assert all(chunk.metadata["alias_path"] == "pdf/ABC/2020/page_1.pdf" for chunk in chunks)
    assert all(chunk.description_text == "" for chunk in chunks)


def test_large_table_splits_between_rows_and_repeats_header() -> None:
    table = "\n".join(
        ["| Year | Revenue |", "| --- | --- |"]
        + [f"| {year} | {year * 10} |" for year in range(2010, 2021)]
    )
    chunks = build_context_chunks(
        _context(source_text=table, table_text=table, pre_text="", post_text=""),
        document_id="b" * 64,
        alias_id=4,
        max_words=16,
        overlap_words=2,
    )

    tables = [chunk.source_text for chunk in chunks if chunk.kind == "table"]
    assert len(tables) > 1
    assert all(text.startswith("| Year | Revenue |\n| --- | --- |") for text in tables)
    assert sum(text.count("| 2010 | 20100 |") for text in tables) == 1
    assert sum(text.count("| 2020 | 20200 |") for text in tables) == 1


def test_narrative_windows_have_overlap() -> None:
    words = [f"word{index}" for index in range(14)]
    chunks = build_context_chunks(
        _context(
            source_text=" ".join(words),
            table_text="",
            pre_text="",
            post_text="",
        ),
        document_id="c" * 64,
        alias_id=5,
        max_words=6,
        overlap_words=2,
    )
    narratives = [chunk.source_text.split() for chunk in chunks if chunk.kind == "narrative"]
    assert narratives[0][-2:] == narratives[1][:2]


def test_forbidden_nested_field_is_rejected() -> None:
    with pytest.raises(LeakageError):
        assert_no_forbidden_fields(
            {"safe": {"gold-answers": ["42"]}},
            location="test payload",
        )

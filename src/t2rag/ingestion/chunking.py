"""Create source-only whole-context and table-aware retrieval chunks."""

from __future__ import annotations

import re
from collections.abc import Iterable

from t2rag.models import ChunkRecord

from .metadata import NormalizedContext, assert_no_forbidden_fields

_WORD = re.compile(r"\S+")
_TABLE_SEPARATOR = re.compile(r"^\s*\|?(?:\s*:?-{3,}:?\s*\|)+\s*$")


def _word_count(text: str) -> int:
    return len(_WORD.findall(text))


def _is_table_line(line: str) -> bool:
    """Recognize Markdown table rows without classifying ordinary prose pipes."""

    stripped = line.strip()
    return stripped.count("|") >= 2 or bool(_TABLE_SEPARATOR.match(stripped))


def _split_mixed_markdown(text: str) -> list[tuple[str, str]]:
    """Return ordered ``(kind, text)`` segments from mixed prose and tables."""

    segments: list[tuple[str, str]] = []
    current_kind: str | None = None
    current_lines: list[str] = []

    def flush() -> None:
        nonlocal current_kind, current_lines
        value = "\n".join(current_lines).strip()
        if value and current_kind is not None:
            segments.append((current_kind, value))
        current_kind = None
        current_lines = []

    for line in text.splitlines():
        kind = "table" if _is_table_line(line) else "narrative"
        # Blank lines stay with the current prose segment, but close a table.
        if not line.strip() and current_kind == "table":
            flush()
            continue
        if current_kind is not None and kind != current_kind and line.strip():
            flush()
        if current_kind is None:
            current_kind = kind
        current_lines.append(line)
    flush()
    return segments


def _window_words(text: str, max_words: int, overlap_words: int) -> list[str]:
    """Split prose into deterministic windows while retaining configured overlap."""

    words = _WORD.findall(text)
    if not words:
        return []
    if len(words) <= max_words:
        return [" ".join(words)]

    step = max_words - overlap_words
    windows: list[str] = []
    for start in range(0, len(words), step):
        window = words[start : start + max_words]
        if not window:
            break
        windows.append(" ".join(window))
        if start + max_words >= len(words):
            break
    return windows


def _table_header(lines: list[str]) -> tuple[list[str], list[str]]:
    """Separate a Markdown header from body rows when a separator is present."""

    for index, line in enumerate(lines[:3]):
        if _TABLE_SEPARATOR.match(line.strip()):
            return lines[: index + 1], lines[index + 1 :]
    return [], lines


def _split_table(text: str, max_words: int) -> list[str]:
    """Split a large table only between rows and repeat its column header."""

    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return []
    if _word_count(text) <= max_words:
        return ["\n".join(lines)]

    header, rows = _table_header(lines)
    # A single wide row is safer intact than split into meaningless columns.
    chunks: list[str] = []
    current = list(header)
    for row in rows:
        candidate = [*current, row]
        if current != header and _word_count("\n".join(candidate)) > max_words:
            chunks.append("\n".join(current))
            current = [*header, row]
        else:
            current = candidate
    if current and current != header:
        chunks.append("\n".join(current))
    return chunks or ["\n".join(lines)]


def _source_segments(context: NormalizedContext) -> Iterable[tuple[str, str, str]]:
    """Yield ordered kind, text, and component labels for table-aware chunking."""

    # FinQA variants provide lossless component fields.  Use them instead of
    # heuristics so table rows remain intact and narrative order is explicit.
    if context.table_text:
        if context.pre_text:
            yield "narrative", context.pre_text, "pre_text"
        yield "table", context.table_text, "table"
        if context.post_text:
            yield "narrative", context.post_text, "post_text"
        return

    # TAT-DQA supplies one combined Markdown string.  Detect its pipe tables.
    for kind, text in _split_mixed_markdown(context.source_text):
        yield kind, text, "context"


def build_context_chunks(
    context: NormalizedContext,
    *,
    document_id: str,
    alias_id: int,
    max_words: int = 500,
    overlap_words: int = 80,
) -> list[ChunkRecord]:
    """Build one complete context plus table-aware evidence chunks.

    The whole-context chunk supports an auditable baseline.  The smaller chunks
    preserve complete table rows and improve lexical matching.  None contains a
    question, answer, or generated description.
    """

    if max_words <= 0:
        raise ValueError("max_words must be positive.")
    if overlap_words < 0 or overlap_words >= max_words:
        raise ValueError("overlap_words must satisfy 0 <= overlap_words < max_words.")
    assert_no_forbidden_fields(context.metadata, location="chunk input metadata")

    chunks = [
        ChunkRecord(
            context_uid=context.context_uid,
            document_id=document_id,
            alias_id=alias_id,
            dataset=context.dataset,
            split=context.split,
            kind="whole_context",
            source_text=context.source_text,
            page_number=context.page_number,
            metadata={
                "position": 0,
                "source_component": "context",
                "alias_path": context.source_file,
            },
        )
    ]

    positions = {"narrative": 0, "table": 0}
    for kind, source_text, component in _source_segments(context):
        pieces = (
            _split_table(source_text, max_words)
            if kind == "table"
            else _window_words(source_text, max_words, overlap_words)
        )
        for piece in pieces:
            position = positions[kind]
            positions[kind] += 1
            chunks.append(
                ChunkRecord(
                    context_uid=context.context_uid,
                    document_id=document_id,
                    alias_id=alias_id,
                    dataset=context.dataset,
                    split=context.split,
                    kind=kind,
                    source_text=piece,
                    page_number=context.page_number,
                    metadata={
                        "position": position,
                        "source_component": component,
                        "alias_path": context.source_file,
                    },
                )
            )
    return chunks

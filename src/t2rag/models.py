"""Shared data models used across ingestion, retrieval, and generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ChunkRecord:
    """One searchable evidence unit.

    ``source_text`` always contains source-derived evidence. Model-generated
    text stays in ``description_text`` so callers can exclude it from answers.
    """

    context_uid: str
    document_id: str
    dataset: str
    split: str
    kind: str
    source_text: str
    chunk_id: int | None = None
    alias_id: int | None = None
    page_number: int | None = None
    description_text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def text_for_index(self, include_description: bool = False) -> str:
        """Return index input without changing the evidence text."""

        if include_description and self.description_text.strip():
            return f"{self.source_text}\n\n[Generated description]\n{self.description_text}"
        return self.source_text


@dataclass(slots=True)
class SearchHit:
    """A retrieved chunk with one or more retrieval scores."""

    chunk_id: int
    context_uid: str
    document_id: str
    source_text: str
    score: float
    rank: int
    retrieval_source: str
    dataset: str = ""
    split: str = ""
    page_number: int | None = None
    alias_path: str = ""
    component_scores: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Citation:
    """A source location cited by an answer."""

    document_id: str
    context_uid: str
    alias_path: str
    page_number: int | None
    chunk_id: int


@dataclass(slots=True)
class AnswerResult:
    """Validated answer returned by an answer model."""

    answer: str
    normalized_answer: str
    calculation: str
    citations: list[Citation]
    abstained: bool
    confidence: float | None = None
    result_unit: str = ""
    raw_response: str = ""
    usage: dict[str, Any] = field(default_factory=dict)

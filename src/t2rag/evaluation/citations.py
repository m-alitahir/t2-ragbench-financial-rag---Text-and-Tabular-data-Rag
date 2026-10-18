"""Citation validity and gold-source correctness metrics."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from statistics import fmean

from t2rag.models import Citation, SearchHit


@dataclass(frozen=True, slots=True)
class CitationMetricResult:
    """Per-answer citation metrics.

    ``validity`` checks that identifiers refer to retrieved evidence.
    ``precision`` and ``recall`` check references to canonical gold contexts.
    ``correct`` requires no invented citation and at least one gold citation.
    """

    citation_validity: float
    citation_precision: float
    citation_recall: float
    citation_correct: float
    invalid_citation_count: int

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _hit_key(hit: SearchHit) -> tuple[object, ...]:
    return (
        hit.document_id,
        hit.context_uid,
        hit.alias_path,
        hit.page_number,
        hit.chunk_id,
    )


def _citation_key(citation: Citation) -> tuple[object, ...]:
    return (
        citation.document_id,
        citation.context_uid,
        citation.alias_path,
        citation.page_number,
        citation.chunk_id,
    )


def score_citations(
    citations: Sequence[Citation],
    available_evidence: Sequence[SearchHit],
    relevant_context_uids: Iterable[str],
    relevant_document_ids: Iterable[str] | None = None,
) -> CitationMetricResult:
    """Score citations against retrieved evidence and canonical gold contexts."""

    relevant_contexts = set(relevant_context_uids)
    relevant_documents = set(relevant_document_ids or [])
    if not relevant_contexts and not relevant_documents:
        raise ValueError("At least one relevant context or document is required.")
    if not citations:
        return CitationMetricResult(0.0, 0.0, 0.0, 0.0, 0)

    allowed = {_hit_key(hit) for hit in available_evidence}
    valid = [citation for citation in citations if _citation_key(citation) in allowed]
    correct = [
        citation
        for citation in valid
        if citation.context_uid in relevant_contexts
        or citation.document_id in relevant_documents
    ]
    if relevant_documents:
        cited_relevant = {
            citation.document_id
            for citation in correct
            if citation.document_id in relevant_documents
        }
        relevant_count = len(relevant_documents)
    else:
        cited_relevant = {
            citation.context_uid
            for citation in correct
            if citation.context_uid in relevant_contexts
        }
        relevant_count = len(relevant_contexts)
    invalid_count = len(citations) - len(valid)

    return CitationMetricResult(
        citation_validity=len(valid) / len(citations),
        citation_precision=len(correct) / len(citations),
        citation_recall=len(cited_relevant) / relevant_count,
        citation_correct=float(invalid_count == 0 and bool(correct)),
        invalid_citation_count=invalid_count,
    )


def aggregate_citation_metrics(
    results: Iterable[CitationMetricResult],
) -> dict[str, float]:
    """Return macro citation rates and mean invalid-citation count."""

    rows = [result.as_dict() for result in results]
    if not rows:
        raise ValueError("At least one citation result is required.")
    return {name: fmean(float(row[name]) for row in rows) for name in rows[0]}

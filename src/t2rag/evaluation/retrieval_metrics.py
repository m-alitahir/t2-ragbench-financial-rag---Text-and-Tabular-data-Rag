"""Standard ranking metrics computed at canonical-context level."""

from __future__ import annotations

import math
from collections.abc import Hashable, Iterable, Sequence
from dataclasses import asdict, dataclass
from statistics import fmean

from t2rag.models import SearchHit


@dataclass(frozen=True, slots=True)
class RetrievalMetricResult:
    """Per-query Recall/Hit, reciprocal rank, and nDCG values."""

    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    hit_at_1: float
    hit_at_5: float
    hit_at_10: float
    mrr_at_10: float
    ndcg_at_10: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def _unique(items: Iterable[Hashable]) -> list[Hashable]:
    """Preserve first rank while collapsing multiple chunks from one context."""

    seen: set[Hashable] = set()
    result: list[Hashable] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _recall_at(ranking: Sequence[Hashable], relevant: set[Hashable], k: int) -> float:
    return len(relevant.intersection(ranking[:k])) / len(relevant)


def _hit_at(ranking: Sequence[Hashable], relevant: set[Hashable], k: int) -> float:
    return float(any(item in relevant for item in ranking[:k]))


def score_ranking(
    ranked_ids: Iterable[Hashable],
    relevant_ids: Iterable[Hashable],
) -> RetrievalMetricResult:
    """Score one ranked list using binary relevance.

    Identifiers are deduplicated before scoring.  This prevents several child
    chunks from the same canonical context from occupying several evaluation
    ranks.  The retrieval system may still keep those chunks for generation.
    """

    ranking = _unique(ranked_ids)
    relevant = set(relevant_ids)
    if not relevant:
        raise ValueError("At least one relevant identifier is required.")

    reciprocal_rank = 0.0
    for rank, item in enumerate(ranking[:10], start=1):
        if item in relevant:
            reciprocal_rank = 1.0 / rank
            break

    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, item in enumerate(ranking[:10], start=1)
        if item in relevant
    )
    ideal_hits = min(len(relevant), 10)
    ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))

    return RetrievalMetricResult(
        recall_at_1=_recall_at(ranking, relevant, 1),
        recall_at_5=_recall_at(ranking, relevant, 5),
        recall_at_10=_recall_at(ranking, relevant, 10),
        hit_at_1=_hit_at(ranking, relevant, 1),
        hit_at_5=_hit_at(ranking, relevant, 5),
        hit_at_10=_hit_at(ranking, relevant, 10),
        mrr_at_10=reciprocal_rank,
        ndcg_at_10=dcg / ideal_dcg,
    )


def score_search_hits(
    hits: Sequence[SearchHit],
    relevant_context_uids: Iterable[str],
) -> RetrievalMetricResult:
    """Convenience wrapper that evaluates ``SearchHit.context_uid`` values."""

    ordered = sorted(hits, key=lambda hit: hit.rank)
    return score_ranking(
        (hit.context_uid for hit in ordered),
        relevant_context_uids,
    )


def aggregate_retrieval_metrics(
    results: Iterable[RetrievalMetricResult],
) -> dict[str, float]:
    """Return macro means, giving every question equal weight."""

    rows = [result.as_dict() for result in results]
    if not rows:
        raise ValueError("At least one retrieval result is required.")
    return {name: fmean(row[name] for row in rows) for name in rows[0]}

"""Rank fusion utilities that do not assume comparable component scores."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace

from t2rag.models import SearchHit


def reciprocal_rank_fusion(
    rankings: Mapping[str, Sequence[SearchHit]],
    *,
    k: int = 60,
    limit: int | None = None,
) -> list[SearchHit]:
    """Fuse ranked lists with reciprocal-rank fusion (RRF).

    BM25 and cosine scores have unrelated scales.  RRF uses only position, so
    neither retriever dominates because of its score range.  A chunk receives
    ``1 / (k + rank)`` from every list that contains it.

    The mapping key is the stable component name stored in
    ``component_scores``.  Duplicate chunk IDs inside one component contribute
    only once, which prevents malformed input from inflating a result.
    """

    if k < 0:
        raise ValueError("RRF k cannot be negative.")
    if limit is not None and limit <= 0:
        return []

    totals: dict[int, float] = {}
    best_rank: dict[int, int] = {}
    representatives: dict[int, SearchHit] = {}
    components: dict[int, dict[str, float]] = {}

    # Sorting component names makes tie behavior independent of dictionary
    # insertion order and therefore reproducible across callers.
    for component_name in sorted(rankings):
        seen_in_component: set[int] = set()
        for position, hit in enumerate(rankings[component_name], start=1):
            if hit.chunk_id in seen_in_component:
                continue
            seen_in_component.add(hit.chunk_id)
            previous = representatives.get(hit.chunk_id)
            previous_best = best_rank.get(hit.chunk_id)
            if previous is None or previous_best is None or position < previous_best:
                representatives[hit.chunk_id] = hit
            totals[hit.chunk_id] = totals.get(hit.chunk_id, 0.0) + 1.0 / (k + position)
            best_rank[hit.chunk_id] = min(previous_best or position, position)
            components.setdefault(hit.chunk_id, {}).update(hit.component_scores)
            components[hit.chunk_id][component_name] = float(hit.score)

    ordered_ids = sorted(
        totals,
        key=lambda chunk_id: (-totals[chunk_id], best_rank[chunk_id], chunk_id),
    )
    if limit is not None:
        ordered_ids = ordered_ids[:limit]

    return [
        replace(
            representatives[chunk_id],
            score=totals[chunk_id],
            rank=rank,
            retrieval_source="rrf",
            component_scores={
                **components[chunk_id],
                "rrf": totals[chunk_id],
            },
        )
        for rank, chunk_id in enumerate(ordered_ids, start=1)
    ]

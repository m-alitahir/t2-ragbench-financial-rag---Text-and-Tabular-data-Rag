"""Small deterministic query router for optional retrieval experiments."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal


QueryType = Literal[
    "exact_lookup",
    "semantic_question",
    "table_lookup",
    "calculation",
    "visual_question",
    "multi_source_question",
]


@dataclass(frozen=True, slots=True)
class QueryRoute:
    """A transparent route that can be saved with evaluation output."""

    query_type: QueryType
    use_dense: bool
    reason: str


_EXACT_TOKEN = re.compile(r"\b(?=[A-Z0-9._/-]{4,}\b)(?=[A-Z0-9._/-]*\d)[A-Z0-9._/-]+\b")
_CALCULATION = re.compile(
    r"\b(?:sum|total|average|difference|change|growth|ratio|percentage|percent|"
    r"how much (?:more|less)|increase|decrease)\b",
    re.IGNORECASE,
)
_VISUAL = re.compile(r"\b(?:chart|figure|image|diagram|graph|visual)\b", re.IGNORECASE)
_TABLE = re.compile(r"\b(?:table|row|column|line item|statement)\b", re.IGNORECASE)
_MULTI_SOURCE = re.compile(
    r"\b(?:compare|across reports|across companies|both companies|respectively)\b",
    re.IGNORECASE,
)


def route_query(query: str) -> QueryRoute:
    """Classify one query with readable rules and no model call."""

    clean = query.strip()
    if not clean:
        raise ValueError("The query router requires a non-empty question.")
    if _VISUAL.search(clean):
        return QueryRoute("visual_question", True, "visual term detected")
    if _MULTI_SOURCE.search(clean):
        return QueryRoute("multi_source_question", True, "comparison term detected")
    if _CALCULATION.search(clean):
        return QueryRoute("calculation", True, "calculation term detected")
    if _EXACT_TOKEN.search(clean):
        return QueryRoute("exact_lookup", False, "identifier-like token detected")
    if _TABLE.search(clean):
        return QueryRoute("table_lookup", True, "table term detected")
    return QueryRoute("semantic_question", True, "default semantic route")

"""Tests for transparent query routing rules."""

from t2rag.retrieval.router import route_query


def test_identifier_query_prefers_lexical_retrieval() -> None:
    route = route_query("Find line item INV-2026-00471")

    assert route.query_type == "exact_lookup"
    assert route.use_dense is False


def test_calculation_query_keeps_dense_retrieval() -> None:
    route = route_query("What was the percentage change in revenue?")

    assert route.query_type == "calculation"
    assert route.use_dense is True

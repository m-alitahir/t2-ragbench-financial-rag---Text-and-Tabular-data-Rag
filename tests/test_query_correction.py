"""Offline tests for conservative financial-query spelling correction."""

from __future__ import annotations

from t2rag.retrieval.query_correction import FinancialQueryCorrector


def _corrector() -> FinancialQueryCorrector:
    vocabulary = {
        "apple",
        "gross",
        "margin",
        "million",
        "operating",
        "revenue",
        "transactions",
    }
    return FinancialQueryCorrector(lambda: vocabulary)


def test_corrects_clear_financial_typos_and_preserves_punctuation() -> None:
    expansion = _corrector().expand(
        "What was Aple's gross margn in 2004?"
    )

    assert expansion.corrected == "What was Apple's gross margin in 2004?"
    assert expansion.applied
    assert [(edit.original, edit.corrected) for edit in expansion.edits] == [
        ("Aple's", "Apple's"),
        ("margn", "margin"),
    ]


def test_preserves_numbers_tickers_currency_and_hyphenated_codes() -> None:
    question = "AAPL Q3 2024 revenue was $1,708.5 for INV-2026-00471."

    expansion = _corrector().expand(question)

    assert expansion.corrected == question
    assert not expansion.applied


def test_does_not_guess_a_low_confidence_unknown_entity() -> None:
    question = "What was Xylophone Holdings revenue?"

    expansion = _corrector().expand(question)

    assert expansion.corrected == question
    assert not expansion.applied


def test_exact_query_stays_unchanged() -> None:
    question = "What was Apple's gross margin in 2004?"

    expansion = _corrector().expand(question)

    assert expansion.corrected == question
    assert not expansion.applied


def test_common_term_can_override_one_document_ocr_spelling() -> None:
    corrector = FinancialQueryCorrector(
        lambda: {
            "aple": 1,
            "ample": 2,
            "apple": 100,
            "margin": 80,
        }
    )

    expansion = corrector.expand("What was Aple's margin?")

    assert expansion.corrected == "What was Apple's margin?"
    assert expansion.applied


def test_absent_name_uses_a_clear_frequency_lead_to_break_a_tie() -> None:
    corrector = FinancialQueryCorrector(
        lambda: {
            "ample": 3,
            "apple": 13,
            "margin": 20,
        }
    )

    expansion = corrector.expand("What was Aple's margin?")

    assert expansion.corrected == "What was Apple's margin?"
    assert expansion.applied


def test_frequent_exact_term_and_undominated_rare_term_stay_unchanged() -> None:
    corrector = FinancialQueryCorrector(
        lambda: {
            "apple": 100,
            "ample": 1,
            "apply": 3,
        }
    )

    assert not corrector.expand("Apple").applied
    assert not corrector.expand("Ample").applied

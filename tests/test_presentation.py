"""Offline tests for verified comparison extraction and arithmetic."""

from __future__ import annotations

import json
from decimal import Decimal

import pytest

from t2rag.generation.deepinfra import DeepInfraResponse
from t2rag.generation.presentation import (
    ComparisonPresenter,
    PresentationError,
    asks_for_average,
    compute_comparison,
    format_fact_value,
    is_comparison_question,
)
from t2rag.models import AnswerResult, SearchHit


def _hit(text: str, *, chunk_id: int = 7) -> SearchHit:
    return SearchHit(
        chunk_id=chunk_id,
        context_uid="finqa:apple-2004",
        document_id="apple-report",
        source_text=text,
        score=0.99,
        rank=1,
        retrieval_source="qwen3_reranker",
        dataset="FinQA",
        split="dev",
        page_number=36,
        alias_path="data/FinQA/dev/pdf/AAPL/2004/page_36.pdf",
    )


def _answer() -> AnswerResult:
    return AnswerResult(
        answer="2004 had the higher gross margin by $551 million",
        normalized_answer="551",
        result_unit="USD million",
        calculation="2259 - 1708",
        citations=[],
        abstained=False,
    )


def _payload(
    *,
    old_value: str = "1708",
    new_value: str = "2259",
    old_period: str = "2003",
    new_period: str = "2004",
    value_kind: str = "amount",
    unit: str = "USD",
    scale: str = "million",
) -> dict[str, object]:
    return {
        "supported": True,
        "title": "Apple Gross Margin Comparison",
        "facts": [
            {
                "id": "gross_margin_old",
                "label": "Gross margin",
                "period": old_period,
                "value": old_value,
                "value_kind": value_kind,
                "unit": unit,
                "scale": scale,
                "evidence_id": "E1",
            },
            {
                "id": "gross_margin_new",
                "label": "Gross margin",
                "period": new_period,
                "value": new_value,
                "value_kind": value_kind,
                "unit": unit,
                "scale": scale,
                "evidence_id": "E1",
            },
        ],
        "comparison": {
            "label": "Apple's gross margin",
            "baseline_fact_id": "gross_margin_old",
            "comparison_fact_id": "gross_margin_new",
        },
    }


class _FakeClient:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.calls = 0

    def complete(self, messages: object, **kwargs: object) -> DeepInfraResponse:
        del messages, kwargs
        self.calls += 1
        return DeepInfraResponse(content=json.dumps(self.payload))


def _present(payload: dict[str, object], source: str):
    client = _FakeClient(payload)
    presenter = ComparisonPresenter(object(), client=client)  # type: ignore[arg-type]
    plan = presenter.present("Compare gross margin.", _answer(), [_hit(source)])
    assert plan is not None
    assert client.calls == 1
    return plan


def test_apple_comparison_is_verified_and_computed_in_python() -> None:
    plan = _present(
        _payload(),
        "Gross margin ($ in millions): 2004 $2,259; 2003 $1,708.",
    )

    computed = compute_comparison(plan)

    assert computed.change == Decimal("551")
    assert computed.direction == "increase"
    assert computed.higher_fact_id == "gross_margin_new"
    assert computed.relative_change_percent is not None
    assert computed.relative_change_percent.quantize(Decimal("0.1")) == Decimal("32.3")
    assert format_fact_value(computed.comparison) == "$2,259 million"


def test_percent_values_use_percentage_points() -> None:
    plan = _present(
        _payload(
            old_value="20",
            new_value="22",
            value_kind="percent",
            unit="percent",
            scale="unit",
        ),
        "Gross margin was 20% in 2003 and 22% in 2004.",
    )

    computed = compute_comparison(plan)

    assert computed.percentage_point_change == Decimal("2")
    assert computed.relative_change_percent is None


@pytest.mark.parametrize(
    ("old_value", "source"),
    [
        ("0", "Gross margin ($ in millions): 2004 $10; 2003 $0."),
        ("-5", "Gross margin ($ in millions): 2004 $10; 2003 ($5)."),
    ],
)
def test_non_positive_baseline_omits_relative_percentage(
    old_value: str,
    source: str,
) -> None:
    plan = _present(_payload(old_value=old_value, new_value="10"), source)

    computed = compute_comparison(plan)

    assert computed.relative_change_percent is None
    assert computed.change == Decimal("10") - Decimal(old_value)


def test_invented_fact_value_is_rejected() -> None:
    payload = _payload(new_value="2258")
    client = _FakeClient(payload)
    presenter = ComparisonPresenter(object(), client=client)  # type: ignore[arg-type]

    with pytest.raises(PresentationError, match="not present"):
        presenter.present(
            "Compare gross margin.",
            _answer(),
            [_hit("Gross margin ($ in millions): 2004 $2,259; 2003 $1,708.")],
        )


def test_accounting_parentheses_are_read_as_negative_values() -> None:
    plan = _present(
        _payload(old_value="-448", new_value="-181"),
        "Pretax loss ($ in millions): 2004 (181); 2003 (448).",
    )

    computed = compute_comparison(plan)

    assert computed.change == Decimal("267")
    assert computed.relative_change_percent is None


def test_comparison_question_detection_is_narrow() -> None:
    assert is_comparison_question("Compare gross margin in 2003 and 2004.")
    assert is_comparison_question("Which year had the higher gross margin?")
    assert not is_comparison_question("What was revenue in 2004?")
    assert asks_for_average("What was the average across both years?")

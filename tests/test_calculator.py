"""Unit tests for the non-executing arithmetic evaluator."""

from decimal import Decimal

import pytest

from t2rag.generation.calculator import (
    CalculationError,
    evaluate_arithmetic,
    format_decimal,
)


def test_evaluates_financial_percentage_expression() -> None:
    result = evaluate_arithmetic("(532 - 470) / 470 * 100")

    assert result.quantize(Decimal("0.0001")) == Decimal("13.1915")


def test_supports_parentheses_unary_signs_and_small_powers() -> None:
    assert evaluate_arithmetic("-(2 + 3) * 4 + 2 ** 3") == Decimal("-12")


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('echo unsafe')",
        "sqrt(4)",
        "value + 1",
        "1 // 2",
        "5 % 2",
        "1 / 0",
        "2 ** 21",
        "2 ** 0.5",
    ],
)
def test_rejects_unsafe_or_unsupported_expressions(expression: str) -> None:
    with pytest.raises(CalculationError):
        evaluate_arithmetic(expression)


def test_rejects_excessive_input_size() -> None:
    with pytest.raises(CalculationError, match="too long"):
        evaluate_arithmetic("1+" * 200 + "1")


def test_formats_decimal_without_scientific_notation() -> None:
    assert format_decimal(Decimal("123.4500")) == "123.45"
    assert format_decimal(Decimal("1E+2")) == "100"

"""A small arithmetic evaluator with no access to Python execution features.

The answer model may propose an arithmetic expression such as
``(532 - 470) / 470 * 100``.  This module evaluates only numeric literals and
basic operators.  It deliberately rejects names, function calls, indexing,
attribute access, and all other Python syntax.
"""

from __future__ import annotations

import ast
from decimal import Decimal, DivisionByZero, InvalidOperation, localcontext


class CalculationError(ValueError):
    """Raised when an expression is unsafe, invalid, or unreasonably large."""


_MAX_EXPRESSION_LENGTH = 256
_MAX_AST_NODES = 64
_MAX_ABSOLUTE_EXPONENT = 20
_MAX_ABSOLUTE_RESULT = Decimal("1e100")


def _checked(value: Decimal) -> Decimal:
    """Reject non-finite or excessively large intermediate values."""

    if not value.is_finite():
        raise CalculationError("The calculation produced a non-finite number.")
    if abs(value) > _MAX_ABSOLUTE_RESULT:
        raise CalculationError("The calculation result is too large.")
    return value


def _evaluate_node(node: ast.AST) -> Decimal:
    """Evaluate one node from the already-validated expression tree."""

    if isinstance(node, ast.Expression):
        return _evaluate_node(node.body)

    if isinstance(node, ast.Constant):
        # ``bool`` is a subclass of ``int``.  Reject it explicitly so that the
        # calculator accepts numbers only.
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise CalculationError("Only numeric literals are allowed.")
        return _checked(Decimal(str(node.value)))

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        operand = _evaluate_node(node.operand)
        return operand if isinstance(node.op, ast.UAdd) else _checked(-operand)

    if not isinstance(node, ast.BinOp):
        raise CalculationError(f"Unsupported expression element: {type(node).__name__}.")

    left = _evaluate_node(node.left)
    right = _evaluate_node(node.right)

    try:
        if isinstance(node.op, ast.Add):
            result = left + right
        elif isinstance(node.op, ast.Sub):
            result = left - right
        elif isinstance(node.op, ast.Mult):
            result = left * right
        elif isinstance(node.op, ast.Div):
            if right == 0:
                raise CalculationError("Division by zero is not allowed.")
            result = left / right
        elif isinstance(node.op, ast.Pow):
            # Decimal powers with fractional exponents have surprising edge
            # cases.  Financial benchmark arithmetic does not need them.
            if right != right.to_integral_value():
                raise CalculationError("The exponent must be an integer.")
            exponent = int(right)
            if abs(exponent) > _MAX_ABSOLUTE_EXPONENT:
                raise CalculationError("The exponent is too large.")
            result = left**exponent
        else:
            raise CalculationError(
                "Only +, -, *, /, **, unary signs, and parentheses are allowed."
            )
    except (DivisionByZero, InvalidOperation, OverflowError) as exc:
        raise CalculationError("The arithmetic expression is invalid.") from exc

    return _checked(result)


def evaluate_arithmetic(expression: str) -> Decimal:
    """Safely evaluate one arithmetic expression and return a ``Decimal``.

    The parser never calls ``eval``.  Limits on text length, tree size,
    exponent size, and result magnitude prevent accidental resource abuse.
    """

    if not isinstance(expression, str) or not expression.strip():
        raise CalculationError("The calculation expression is empty.")
    if len(expression) > _MAX_EXPRESSION_LENGTH:
        raise CalculationError("The calculation expression is too long.")

    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except (SyntaxError, ValueError) as exc:
        raise CalculationError("The calculation expression is not valid arithmetic.") from exc

    if sum(1 for _ in ast.walk(tree)) > _MAX_AST_NODES:
        raise CalculationError("The calculation expression is too complex.")

    # Extra precision makes ratios stable while still imposing a finite work
    # budget.  The caller controls presentation and rounding of the result.
    with localcontext() as context:
        context.prec = 50
        return _evaluate_node(tree)


def format_decimal(value: Decimal) -> str:
    """Return a plain, compact decimal string without scientific notation."""

    if value == 0:
        return "0"
    text = format(value.normalize(), "f")
    return text.rstrip("0").rstrip(".") if "." in text else text

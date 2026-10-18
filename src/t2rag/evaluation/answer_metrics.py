"""Answer normalization and accuracy metrics for financial questions."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from statistics import fmean

from t2rag.models import AnswerResult


_NUMBER_PATTERN = re.compile(
    r"[-+]?(?:\d+(?:,\d{3})*(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
)
_CURRENCY = "$£€¥₹"


@dataclass(frozen=True, slots=True)
class AnswerMetricResult:
    """Per-question answer score, including abstention and parser failure."""

    exact_match: float
    numeric_match: float
    correct: float
    invalid_output: float
    abstained: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SemanticAnswerMetricResult:
    """Secondary deterministic score for human-readable financial answers.

    The official benchmark score remains :class:`AnswerMetricResult`, which
    compares ``normalized_answer`` with ``program_answer``.  This secondary
    score also considers the formatted ``original_answer``, explicit percent
    units, and the locally verified calculator result.  It never replaces or
    changes the official score.
    """

    display_exact_match: float
    semantic_numeric_match: float
    calculator_numeric_match: float
    calculator_consistent: float
    semantic_correct: float
    label_conflict: float
    invalid_output: float
    abstained: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def normalize_answer_text(value: object) -> str:
    """Normalize case, Unicode, separators, and surrounding punctuation."""

    text = unicodedata.normalize("NFKC", str(value)).strip().lower()
    text = text.replace("−", "-").replace("–", "-").replace("—", "-")
    text = text.replace(",", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip(" \t\r\n.;:")


def parse_numeric_answer(value: object) -> Decimal | None:
    """Extract one financial number without changing percentage magnitude.

    T2-RAGBench program answers represent ``13.2%`` as ``13.2``, not
    ``0.132``.  Therefore the percent sign is treated as a unit.  Likewise,
    words such as ``million`` are contextual units and do not scale the value.
    Accounting parentheses around the entire value indicate a negative number.
    """

    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, Decimal):
        return value if value.is_finite() else None
    if isinstance(value, (int, float)):
        try:
            number = Decimal(str(value))
        except InvalidOperation:
            return None
        return number if number.is_finite() else None

    text = unicodedata.normalize("NFKC", str(value)).strip()
    if not text:
        return None
    text = text.replace("−", "-").replace("–", "-").replace("—", "-")
    matches = _NUMBER_PATTERN.findall(text)
    if len(matches) != 1:
        return None

    token = matches[0].replace(",", "")
    try:
        number = Decimal(token)
    except InvalidOperation:
        return None
    if not number.is_finite():
        return None

    # Parentheses count as an accounting negative only when they wrap the
    # complete answer, apart from currency signs, percent, and unit words.
    compact = text.strip()
    accounting = re.fullmatch(
        rf"\(\s*[{re.escape(_CURRENCY)}]?\s*{re.escape(matches[0])}\s*"
        r"(?:%|percent|thousand|million|billion)?\s*\)",
        compact,
        flags=re.IGNORECASE,
    )
    if accounting and number > 0:
        number = -number
    return number


def numeric_answers_match(
    prediction: object,
    gold: object,
    *,
    absolute_tolerance: float = 1e-4,
    relative_tolerance: float = 1e-4,
) -> bool:
    """Compare two parsed numbers with fixed absolute and relative tolerances."""

    predicted = parse_numeric_answer(prediction)
    expected = parse_numeric_answer(gold)
    if predicted is None or expected is None:
        return False
    absolute = Decimal(str(absolute_tolerance))
    relative = Decimal(str(relative_tolerance))
    difference = abs(predicted - expected)
    limit = max(absolute, relative * max(abs(predicted), abs(expected)))
    return difference <= limit


def _contains_percent_unit(value: object) -> bool:
    if not isinstance(value, str):
        return False
    return bool(re.search(r"%|\bpercent(?:age)?\b", value, flags=re.IGNORECASE))


def _numeric_variants(value: object) -> list[Decimal]:
    """Return direct and explicit-percent numeric forms for one value.

    A written percentage has two common representations.  For example,
    ``93.5%`` can appear as ``93.5`` in a display answer or ``0.935`` in a
    normalized program result.  Both are retained only when the percent unit
    is explicit; ordinary unitless numbers are never rescaled.
    """

    parsed = parse_numeric_answer(value)
    if parsed is None:
        return []
    variants = [parsed]
    if _contains_percent_unit(value):
        variants.append(parsed / Decimal("100"))
    return variants


def semantic_numeric_answers_match(
    prediction: object,
    gold: object,
    *,
    absolute_tolerance: float = 1e-3,
    relative_tolerance: float = 1e-3,
) -> bool:
    """Compare numbers while respecting an explicitly written percent unit."""

    predicted_values = _numeric_variants(prediction)
    gold_values = _numeric_variants(gold)
    if not predicted_values or not gold_values:
        return False
    absolute = Decimal(str(absolute_tolerance))
    relative = Decimal(str(relative_tolerance))
    for predicted in predicted_values:
        for expected in gold_values:
            difference = abs(predicted - expected)
            limit = max(absolute, relative * max(abs(predicted), abs(expected)))
            if difference <= limit:
                return True
    return False


def _gold_values(gold: object | Sequence[object]) -> list[object]:
    if isinstance(gold, Sequence) and not isinstance(gold, (str, bytes, bytearray)):
        values = list(gold)
    else:
        values = [gold]
    if not values:
        raise ValueError("At least one gold answer is required.")
    return values


def score_answer(
    prediction: AnswerResult | str | None,
    gold: object | Sequence[object],
    *,
    absolute_tolerance: float = 1e-4,
    relative_tolerance: float = 1e-4,
) -> AnswerMetricResult:
    """Score one validated result; ``None`` represents invalid model output."""

    gold_values = _gold_values(gold)
    if prediction is None:
        return AnswerMetricResult(0.0, 0.0, 0.0, 1.0, 0.0)

    if isinstance(prediction, AnswerResult):
        if prediction.abstained:
            return AnswerMetricResult(0.0, 0.0, 0.0, 0.0, 1.0)
        predicted_value = prediction.normalized_answer or prediction.answer
    elif isinstance(prediction, str):
        predicted_value = prediction
    else:
        raise TypeError("Prediction must be AnswerResult, str, or None.")

    normalized_prediction = normalize_answer_text(predicted_value)
    exact = any(
        normalized_prediction == normalize_answer_text(candidate)
        for candidate in gold_values
    )
    numeric = any(
        numeric_answers_match(
            predicted_value,
            candidate,
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
        )
        for candidate in gold_values
    )
    return AnswerMetricResult(
        exact_match=float(exact),
        numeric_match=float(numeric),
        correct=float(exact or numeric),
        invalid_output=0.0,
        abstained=0.0,
    )


def _prediction_values(prediction: AnswerResult | str) -> list[object]:
    if isinstance(prediction, AnswerResult):
        return [prediction.normalized_answer, prediction.answer]
    return [prediction]


def labels_semantically_conflict(
    program_answer: object,
    original_answer: object | None,
    *,
    absolute_tolerance: float = 1e-4,
    relative_tolerance: float = 1e-3,
) -> bool:
    """Return true when normalized and display labels are not equivalent."""

    if original_answer is None or not str(original_answer).strip():
        return False
    if normalize_answer_text(program_answer) == normalize_answer_text(original_answer):
        return False
    return not semantic_numeric_answers_match(
        program_answer,
        original_answer,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
    )


def score_semantic_answer(
    prediction: AnswerResult | str | None,
    program_answer: object,
    original_answer: object | None,
    *,
    absolute_tolerance: float = 1e-4,
    relative_tolerance: float = 1e-3,
) -> SemanticAnswerMetricResult:
    """Score display equivalence without changing the official benchmark score."""

    conflict = float(
        labels_semantically_conflict(
            program_answer,
            original_answer,
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
        )
    )
    if prediction is None:
        return SemanticAnswerMetricResult(
            display_exact_match=0.0,
            semantic_numeric_match=0.0,
            calculator_numeric_match=0.0,
            calculator_consistent=0.0,
            semantic_correct=0.0,
            label_conflict=conflict,
            invalid_output=1.0,
            abstained=0.0,
        )
    if isinstance(prediction, AnswerResult) and prediction.abstained:
        return SemanticAnswerMetricResult(
            display_exact_match=0.0,
            semantic_numeric_match=0.0,
            calculator_numeric_match=0.0,
            calculator_consistent=0.0,
            semantic_correct=0.0,
            label_conflict=conflict,
            invalid_output=0.0,
            abstained=1.0,
        )
    if not isinstance(prediction, (AnswerResult, str)):
        raise TypeError("Prediction must be AnswerResult, str, or None.")

    gold_values = [program_answer]
    if original_answer is not None and str(original_answer).strip():
        gold_values.append(original_answer)
    prediction_values = _prediction_values(prediction)
    display_exact = any(
        normalize_answer_text(candidate) == normalize_answer_text(gold)
        for candidate in prediction_values
        for gold in gold_values
    )
    numeric = any(
        semantic_numeric_answers_match(
            candidate,
            gold,
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
        )
        for candidate in prediction_values
        for gold in gold_values
    )

    calculator_match = False
    calculator_consistent = False
    if isinstance(prediction, AnswerResult):
        calculator_value = prediction.usage.get("calculator_result")
        if isinstance(calculator_value, (str, int, float, Decimal)):
            calculator_consistent = any(
                semantic_numeric_answers_match(
                    calculator_value,
                    candidate,
                    absolute_tolerance=absolute_tolerance,
                    relative_tolerance=relative_tolerance,
                )
                for candidate in prediction_values
            )
            calculator_match = any(
                semantic_numeric_answers_match(
                    calculator_value,
                    gold,
                    absolute_tolerance=absolute_tolerance,
                    relative_tolerance=relative_tolerance,
                )
                for gold in gold_values
            )
            numeric = numeric or (calculator_match and calculator_consistent)

    return SemanticAnswerMetricResult(
        display_exact_match=float(display_exact),
        semantic_numeric_match=float(numeric),
        calculator_numeric_match=float(calculator_match),
        calculator_consistent=float(calculator_consistent),
        semantic_correct=float(display_exact or numeric),
        label_conflict=conflict,
        invalid_output=0.0,
        abstained=0.0,
    )


def aggregate_answer_metrics(
    results: Iterable[AnswerMetricResult],
) -> dict[str, float]:
    """Return macro accuracy and invalid-output rates."""

    rows = [result.as_dict() for result in results]
    if not rows:
        raise ValueError("At least one answer result is required.")
    return {name: fmean(row[name] for row in rows) for name in rows[0]}


def aggregate_semantic_answer_metrics(
    results: Iterable[SemanticAnswerMetricResult],
) -> dict[str, float]:
    """Return macro secondary metrics without merging them into official scores."""

    rows = [result.as_dict() for result in results]
    if not rows:
        raise ValueError("At least one semantic answer result is required.")
    return {name: fmean(row[name] for row in rows) for name in rows[0]}

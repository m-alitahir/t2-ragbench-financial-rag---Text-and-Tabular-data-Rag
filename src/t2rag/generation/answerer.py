"""Strictly parse evidence-grounded model answers into shared data models."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from decimal import Decimal, InvalidOperation
from typing import Any

from t2rag.config import AppConfig
from t2rag.models import AnswerResult, Citation, SearchHit

from .calculator import CalculationError, evaluate_arithmetic, format_decimal
from .deepinfra import DeepInfraClient
from .prompts import (
    ANSWER_PROMPT_VERSION,
    build_answer_correction_messages,
    build_answer_messages,
)


class AnswerParseError(ValueError):
    """Raised when model output violates the answer schema or evidence rules.

    ``raw_response`` and ``usage`` are retained for local failure analysis.
    They are never repaired or treated as a valid answer.
    """

    def __init__(
        self,
        message: str,
        *,
        raw_response: str = "",
        usage: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.raw_response = raw_response
        self.usage = dict(usage or {})


_ANSWER_KEYS = {
    "answer",
    "normalized_answer",
    "result_unit",
    "calculation",
    "citations",
    "abstained",
    "confidence",
}
_CITATION_KEYS = {
    "document_id",
    "context_uid",
    "alias_path",
    "page_number",
    "chunk_id",
}
_SIMPLE_NUMBER = re.compile(
    r"\s*[$£€¥₹]?\s*([-+]?(?:\d+(?:,\d{3})*(?:\.\d*)?|\.\d+))\s*"
    r"(?:%|percent(?:age)?(?:\s+points?)?|thousand|million|billion)?\s*",
    flags=re.IGNORECASE,
)
_SCALE_WORD = re.compile(
    r"\b(thousand|million|billion|trillion)s?\b",
    flags=re.IGNORECASE,
)
_PER_TARGET = re.compile(
    r"\bper\s+(?:each\s+)?"
    r"(transaction|share|employee|customer|card|account|unit|day|month|year|store|location)s?\b",
    flags=re.IGNORECASE,
)
_TOTAL_ACROSS_PERIODS = re.compile(
    r"\btotal\b.{0,160}\b(?:for|across)\b.{0,50}\b(?:years?|periods?)\b",
    flags=re.IGNORECASE | re.DOTALL,
)
_SUM_USAGE_KEYS = {
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "estimated_cost",
}


def _require_string(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str):
        raise AnswerParseError(f"Answer field '{key}' must be a string.")
    return value.strip()


def _citation_key_from_hit(hit: SearchHit) -> tuple[object, ...]:
    return (
        hit.document_id,
        hit.context_uid,
        hit.alias_path,
        hit.page_number,
        hit.chunk_id,
    )


def _parse_simple_number(value: object) -> Decimal | None:
    """Parse one displayed scalar for calculator-agreement checks."""

    match = _SIMPLE_NUMBER.fullmatch(str(value))
    if match is None:
        return None
    try:
        parsed = Decimal(match.group(1).replace(",", ""))
    except InvalidOperation:
        return None
    return parsed if parsed.is_finite() else None


def _calculator_disagrees(
    result: AnswerResult,
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> bool:
    """Return true only when both answer and calculator are numeric and differ."""

    calculator_value = _parse_simple_number(result.usage.get("calculator_result"))
    predicted_value = _parse_simple_number(result.normalized_answer)
    if predicted_value is None:
        predicted_value = _parse_simple_number(result.answer)
    if calculator_value is None or predicted_value is None:
        return False
    difference = abs(predicted_value - calculator_value)
    limit = max(
        Decimal(str(absolute_tolerance)),
        Decimal(str(relative_tolerance))
        * max(abs(predicted_value), abs(calculator_value)),
    )
    return difference > limit


def _contains_per_target(value: str, target: str) -> bool:
    """Return whether text states the requested singular or plural per-unit target."""

    pattern = re.compile(
        rf"\bper\s+(?:each\s+)?{re.escape(target)}s?\b",
        flags=re.IGNORECASE,
    )
    return pattern.search(value) is not None


def _unit_validation_problem(question: str, result: AnswerResult) -> str | None:
    """Return one high-confidence display-unit conflict without guessing units.

    The check is deliberately narrow. It verifies explicit ``per X`` wording
    and scale agreement between two model fields. It does not infer a unit from
    vague words such as ``average`` because some valid averages remain scaled.
    """

    if result.abstained:
        return None

    per_match = _PER_TARGET.search(question)
    if per_match is not None:
        target = per_match.group(1).lower()
        expected = f"per {target}"
        if not _contains_per_target(result.result_unit, target):
            return f"The result_unit must state '{expected}' because the question asks for it."
        if not _contains_per_target(result.answer, target):
            return f"The displayed answer must state '{expected}' and match result_unit."

    answer_scales = {word.lower() for word in _SCALE_WORD.findall(result.answer)}
    result_scales = {word.lower() for word in _SCALE_WORD.findall(result.result_unit)}
    if answer_scales != result_scales:
        return (
            "The displayed answer and result_unit use different scale words. "
            "Recheck whether thousand, million, billion, or trillion cancels."
        )
    return None


def _operation_validation_problem(question: str, result: AnswerResult) -> str | None:
    """Require arithmetic for an explicit total across multiple periods."""

    if result.abstained:
        return None
    if _TOTAL_ACROSS_PERIODS.search(question) and not result.calculation:
        return (
            "The question asks for a total across multiple periods. Return one "
            "final total and include the addition in calculation."
        )
    return None


def _merge_usage(total: dict[str, Any], current: dict[str, Any]) -> None:
    """Accumulate token and cost values across bounded correction requests."""

    for key, value in current.items():
        if key in _SUM_USAGE_KEYS and isinstance(value, (int, float)):
            existing = total.get(key, 0)
            total[key] = existing + value if isinstance(existing, (int, float)) else value
        else:
            total[key] = value


def _parse_citations(
    raw_citations: Any,
    evidence: Sequence[SearchHit],
) -> list[Citation]:
    """Accept citations only when every field matches supplied evidence."""

    if not isinstance(raw_citations, list):
        raise AnswerParseError("Answer field 'citations' must be a list.")
    allowed = {_citation_key_from_hit(hit) for hit in evidence}
    citations: list[Citation] = []
    seen: set[tuple[object, ...]] = set()

    for raw in raw_citations:
        if not isinstance(raw, dict) or set(raw) != _CITATION_KEYS:
            raise AnswerParseError("Each citation must contain exactly the required fields.")
        identifier_names = ("document_id", "context_uid", "alias_path")
        if not all(isinstance(raw[name], str) for name in identifier_names):
            raise AnswerParseError("Citation identifier fields must be strings.")
        page_number = raw["page_number"]
        chunk_id = raw["chunk_id"]
        if page_number is not None and (
            isinstance(page_number, bool) or not isinstance(page_number, int)
        ):
            raise AnswerParseError("Citation page_number must be an integer or null.")
        if isinstance(chunk_id, bool) or not isinstance(chunk_id, int):
            raise AnswerParseError("Citation chunk_id must be an integer.")

        key = (
            raw["document_id"],
            raw["context_uid"],
            raw["alias_path"],
            page_number,
            chunk_id,
        )
        if key not in allowed:
            raise AnswerParseError("A citation does not match any supplied evidence record.")
        if key in seen:
            raise AnswerParseError("Duplicate citations are not allowed.")
        seen.add(key)
        citations.append(
            Citation(
                document_id=raw["document_id"],
                context_uid=raw["context_uid"],
                alias_path=raw["alias_path"],
                page_number=page_number,
                chunk_id=chunk_id,
            )
        )
    return citations


def parse_answer_response(raw_response: str, evidence: Sequence[SearchHit]) -> AnswerResult:
    """Parse a model response without repairing or guessing malformed output.

    Rejecting Markdown fences and extra fields makes invalid-output rate an
    honest benchmark metric.  It also prevents a permissive parser from
    silently changing a model answer.
    """

    try:
        data = json.loads(raw_response)
    except (json.JSONDecodeError, TypeError) as exc:
        raise AnswerParseError("The model response is not one JSON object.") from exc
    if not isinstance(data, dict) or set(data) != _ANSWER_KEYS:
        raise AnswerParseError("The answer must contain exactly the required fields.")

    answer = _require_string(data, "answer")
    normalized_answer = _require_string(data, "normalized_answer")
    result_unit = _require_string(data, "result_unit")
    calculation = _require_string(data, "calculation")
    citations = _parse_citations(data["citations"], evidence)

    abstained = data["abstained"]
    if not isinstance(abstained, bool):
        raise AnswerParseError("Answer field 'abstained' must be a boolean.")

    confidence = data["confidence"]
    if confidence is not None:
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise AnswerParseError("Answer confidence must be a number or null.")
        confidence = float(confidence)
        if not 0.0 <= confidence <= 1.0:
            raise AnswerParseError("Answer confidence must be between zero and one.")

    if abstained:
        if answer or normalized_answer or result_unit or calculation or citations:
            raise AnswerParseError("An abstention must not contain an answer or citations.")
    else:
        if not answer or not normalized_answer:
            raise AnswerParseError("A non-abstained result needs an answer.")
        if not result_unit:
            raise AnswerParseError("A non-abstained result needs result_unit.")
        if not citations:
            raise AnswerParseError("A non-abstained result needs at least one citation.")

    calculator_result: str | None = None
    if calculation:
        try:
            calculator_result = format_decimal(evaluate_arithmetic(calculation))
        except CalculationError as exc:
            raise AnswerParseError("The calculation is not safe arithmetic.") from exc

    result = AnswerResult(
        answer=answer,
        normalized_answer=normalized_answer,
        calculation=calculation,
        citations=citations,
        abstained=abstained,
        confidence=confidence,
        result_unit=result_unit,
        raw_response=raw_response,
    )
    if calculator_result is not None:
        # Keep the independent arithmetic result for audits.  We do not
        # silently replace the model answer because benchmark answers can use
        # dataset-specific rounding and percentage conventions.
        result.usage["calculator_result"] = calculator_result
    return result


class EvidenceAnswerer:
    """Generate and validate one answer from retrieved source evidence."""

    def __init__(
        self,
        config: AppConfig,
        client: DeepInfraClient | None = None,
    ) -> None:
        self._client = client or DeepInfraClient(config)
        settings = getattr(config, "generation", None)
        self._correction_attempts = max(
            0,
            int(getattr(settings, "correction_attempts", 0)),
        )
        self._calculator_absolute_tolerance = float(
            getattr(settings, "calculator_absolute_tolerance", 0.01)
        )
        self._calculator_relative_tolerance = float(
            getattr(settings, "calculator_relative_tolerance", 1e-4)
        )

    def answer(self, question: str, evidence: Sequence[SearchHit]) -> AnswerResult:
        """Return one validated answer after a bounded correction sequence."""

        # No evidence means there is nothing grounded to send to a paid model.
        if not evidence:
            return AnswerResult(
                answer="",
                normalized_answer="",
                calculation="",
                citations=[],
                abstained=True,
                confidence=None,
                raw_response="",
                usage={"skipped_reason": "no_evidence"},
            )

        messages = build_answer_messages(question, evidence)
        combined_usage: dict[str, Any] = {}
        correction_history: list[dict[str, Any]] = []

        for attempt in range(self._correction_attempts + 1):
            response = self._client.complete(messages)
            _merge_usage(combined_usage, response.usage)
            if response.request_id:
                combined_usage["request_id"] = response.request_id
                combined_usage.setdefault("request_ids", []).append(response.request_id)

            try:
                result = parse_answer_response(response.content, evidence)
            except AnswerParseError as exc:
                problem = str(exc)
                calculator_result = None
                result = None
            else:
                calculator_result = result.usage.get("calculator_result")
                if _calculator_disagrees(
                    result,
                    absolute_tolerance=self._calculator_absolute_tolerance,
                    relative_tolerance=self._calculator_relative_tolerance,
                ):
                    problem = (
                        "The displayed or normalized answer disagrees with the "
                        "trusted local calculator result."
                    )
                elif unit_problem := _unit_validation_problem(question, result):
                    problem = unit_problem
                elif operation_problem := _operation_validation_problem(question, result):
                    problem = operation_problem
                else:
                    result.usage.update(combined_usage)
                    result.usage["answer_prompt_version"] = ANSWER_PROMPT_VERSION
                    result.usage["answer_attempt_count"] = attempt + 1
                    if correction_history:
                        result.usage["correction_history"] = correction_history
                    return result

            correction_history.append(
                {
                    "attempt": attempt + 1,
                    "problem": problem,
                    "raw_response": response.content,
                    "request_id": response.request_id,
                }
            )
            if attempt >= self._correction_attempts:
                combined_usage["answer_prompt_version"] = ANSWER_PROMPT_VERSION
                combined_usage["answer_attempt_count"] = attempt + 1
                combined_usage["correction_history"] = correction_history[:-1]
                raise AnswerParseError(
                    problem,
                    raw_response=response.content,
                    usage=combined_usage,
                )

            messages = build_answer_correction_messages(
                question,
                evidence,
                previous_response=response.content,
                problem=problem,
                calculator_result=(
                    str(calculator_result) if calculator_result is not None else None
                ),
            )

        raise AssertionError("The bounded answer loop ended unexpectedly.")

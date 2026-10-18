"""Evidence-grounded answer generation for T2-RAGBench."""

from .answerer import AnswerParseError, EvidenceAnswerer, parse_answer_response
from .calculator import CalculationError, evaluate_arithmetic
from .deepinfra import DeepInfraClient, DeepInfraResponse
from .presentation import (
    ComparisonPlan,
    ComparisonPresenter,
    ComputedComparison,
    PresentationError,
    PresentationFact,
    asks_for_average,
    compute_comparison,
    is_comparison_question,
)

__all__ = [
    "AnswerParseError",
    "CalculationError",
    "DeepInfraClient",
    "DeepInfraResponse",
    "EvidenceAnswerer",
    "ComparisonPlan",
    "ComparisonPresenter",
    "ComputedComparison",
    "PresentationError",
    "PresentationFact",
    "asks_for_average",
    "compute_comparison",
    "evaluate_arithmetic",
    "is_comparison_question",
    "parse_answer_response",
]

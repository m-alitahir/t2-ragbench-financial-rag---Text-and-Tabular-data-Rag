"""Save test predictions before opening gold labels, then score them later."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from t2rag.config import AppConfig
from t2rag.models import AnswerResult, Citation, SearchHit

from .answer_metrics import (
    AnswerMetricResult,
    SemanticAnswerMetricResult,
    aggregate_answer_metrics,
    aggregate_semantic_answer_metrics,
    score_answer,
    score_semantic_answer,
)
from .citations import CitationMetricResult, aggregate_citation_metrics, score_citations
from .dataset import BenchmarkExample
from .retrieval_metrics import (
    RetrievalMetricResult,
    aggregate_retrieval_metrics,
    score_ranking,
    score_search_hits,
)


@dataclass(slots=True)
class PredictionRecord:
    """One saved prediction without a gold answer."""

    question_id: str
    dataset: str
    split: str
    question_sha256: str
    hits: list[SearchHit]
    result: AnswerResult | None
    latency_seconds: float
    invalid_reason: str = ""
    invalid_raw_response: str = ""
    invalid_usage: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def question_sha256(question: str) -> str:
    """Identify a question without copying its text into run artifacts."""

    return hashlib.sha256(question.encode("utf-8")).hexdigest()


def write_predictions(path: str | Path, records: Iterable[PredictionRecord]) -> None:
    """Write prediction-only JSONL through a complete temporary file."""

    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record.as_dict(), ensure_ascii=False, sort_keys=True))
            stream.write("\n")
    temporary.replace(target)


def _hit_from_dict(value: Any) -> SearchHit:
    if not isinstance(value, dict):
        raise ValueError("Saved retrieval hit must be an object.")
    return SearchHit(**value)


def answer_result_from_dict(value: Any) -> AnswerResult | None:
    """Rebuild a validated answer dataclass from a saved JSON object."""

    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("Saved answer must be an object or null.")
    data = dict(value)
    # Prompt versions 1 and 2 did not save a separate result unit. Keep those
    # development artifacts readable for free rescoring and comparisons.
    data.setdefault("result_unit", "")
    raw_citations = data.get("citations")
    if not isinstance(raw_citations, list):
        raise ValueError("Saved answer citations must be a list.")
    data["citations"] = [Citation(**item) for item in raw_citations]
    return AnswerResult(**data)


def load_predictions(path: str | Path) -> list[PredictionRecord]:
    """Load and validate a saved prediction file."""

    records: list[PredictionRecord] = []
    with Path(path).resolve().open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid prediction JSON at line {line_number}.") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Prediction line {line_number} must be an object.")
            records.append(
                PredictionRecord(
                    question_id=str(value["question_id"]),
                    dataset=str(value["dataset"]),
                    split=str(value["split"]),
                    question_sha256=str(value["question_sha256"]),
                    hits=[_hit_from_dict(item) for item in value.get("hits", [])],
                    result=answer_result_from_dict(value.get("result")),
                    latency_seconds=float(value.get("latency_seconds", 0.0)),
                    invalid_reason=str(value.get("invalid_reason", "")),
                    invalid_raw_response=str(value.get("invalid_raw_response", "")),
                    invalid_usage=dict(value.get("invalid_usage") or {}),
                )
            )
    return records


def score_saved_predictions(
    records: Sequence[PredictionRecord],
    examples: Iterable[BenchmarkExample],
    config: AppConfig,
) -> dict[str, Any]:
    """Score saved predictions after explicit test-label authorization.

    The function verifies question hashes before reading each gold answer. This
    prevents a prediction file from being scored against a different dataset.
    """

    by_key = {(record.dataset, record.question_id): record for record in records}
    retrieval_rows: list[RetrievalMetricResult] = []
    answer_rows: list[AnswerMetricResult] = []
    semantic_answer_rows: list[SemanticAnswerMetricResult] = []
    citation_rows: list[CitationMetricResult] = []
    observed: set[tuple[str, str]] = set()

    for example in examples:
        key = (example.dataset, example.question_id)
        record = by_key.get(key)
        if record is None:
            raise ValueError(f"Missing saved prediction for {key}.")
        if record.split != example.split:
            raise ValueError(f"Prediction split mismatch for {key}.")
        if record.question_sha256 != question_sha256(example.question):
            raise ValueError(f"Prediction question hash mismatch for {key}.")
        if example.program_answer is None:
            raise ValueError("Gold labels were not enabled for scoring.")
        observed.add(key)

        ordered_hits = sorted(record.hits, key=lambda hit: hit.rank)
        if example.document_id is not None:
            retrieval_rows.append(
                score_ranking(
                    (hit.document_id for hit in ordered_hits),
                    [example.document_id],
                )
            )
        else:
            retrieval_rows.append(score_search_hits(ordered_hits, [example.context_uid]))
        answer_rows.append(
            score_answer(
                record.result,
                example.program_answer,
                absolute_tolerance=config.evaluation.numeric_absolute_tolerance,
                relative_tolerance=config.evaluation.numeric_relative_tolerance,
            )
        )
        semantic_answer_rows.append(
            score_semantic_answer(
                record.result,
                example.program_answer,
                example.original_answer,
                absolute_tolerance=config.evaluation.semantic_absolute_tolerance,
                relative_tolerance=config.evaluation.semantic_relative_tolerance,
            )
        )
        citation_rows.append(
            score_citations(
                record.result.citations if record.result is not None else [],
                ordered_hits,
                [example.context_uid],
                [example.document_id] if example.document_id is not None else None,
            )
        )

    extra = sorted(set(by_key) - observed)
    if extra:
        raise ValueError(f"Prediction file contains unexpected questions: {extra[:5]}")
    if not observed:
        raise ValueError("No predictions were scored.")
    return {
        "question_count": len(observed),
        "retrieval": aggregate_retrieval_metrics(retrieval_rows),
        "answer": aggregate_answer_metrics(answer_rows),
        "semantic_answer": aggregate_semantic_answer_metrics(semantic_answer_rows),
        "citations": aggregate_citation_metrics(citation_rows),
    }


def rescore_saved_answer_rows(
    saved_rows: Sequence[dict[str, Any]],
    examples: Iterable[BenchmarkExample],
    config: AppConfig,
) -> dict[str, Any]:
    """Recompute answer metrics from saved development outputs without models."""

    example_by_key = {
        (example.dataset, example.question_id): example for example in examples
    }
    answer_rows: list[AnswerMetricResult] = []
    semantic_rows: list[SemanticAnswerMetricResult] = []
    record_scores: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    for saved in saved_rows:
        key = (str(saved.get("dataset", "")), str(saved.get("question_id", "")))
        if key in seen:
            raise ValueError(f"Duplicate saved answer: {key}.")
        seen.add(key)
        example = example_by_key.get(key)
        if example is None:
            raise ValueError(f"Saved answer does not match the requested data: {key}.")
        if str(saved.get("split", "")) != example.split:
            raise ValueError(f"Saved answer split mismatch for {key}.")
        if example.program_answer is None:
            raise ValueError("Gold labels were not enabled for rescoring.")

        result = answer_result_from_dict(saved.get("result"))
        official = score_answer(
            result,
            example.program_answer,
            absolute_tolerance=config.evaluation.numeric_absolute_tolerance,
            relative_tolerance=config.evaluation.numeric_relative_tolerance,
        )
        semantic = score_semantic_answer(
            result,
            example.program_answer,
            example.original_answer,
            absolute_tolerance=config.evaluation.semantic_absolute_tolerance,
            relative_tolerance=config.evaluation.semantic_relative_tolerance,
        )
        answer_rows.append(official)
        semantic_rows.append(semantic)
        record_scores.append(
            {
                "question_id": example.question_id,
                "dataset": example.dataset,
                "split": example.split,
                "official_answer": official.as_dict(),
                "semantic_answer": semantic.as_dict(),
            }
        )

    if not record_scores:
        raise ValueError("No saved answers were rescored.")
    conflict_ids = [
        row["question_id"]
        for row in record_scores
        if row["semantic_answer"]["label_conflict"] == 1.0
    ]
    return {
        "diagnostic_only": True,
        "model_requests": 0,
        "question_count": len(record_scores),
        "answer": aggregate_answer_metrics(answer_rows),
        "semantic_answer": aggregate_semantic_answer_metrics(semantic_rows),
        "deterministic_label_conflicts": {
            "count": len(conflict_ids),
            "question_ids": conflict_ids,
        },
        "records": record_scores,
    }

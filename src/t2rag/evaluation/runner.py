"""Run retrieval and answer evaluation without coupling to one index backend."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import fmean, median
from typing import Any, Literal

from t2rag.config import AppConfig
from t2rag.generation.answerer import AnswerParseError
from t2rag.models import AnswerResult, SearchHit

from .answer_metrics import (
    AnswerMetricResult,
    SemanticAnswerMetricResult,
    aggregate_answer_metrics,
    aggregate_semantic_answer_metrics,
    score_answer,
    score_semantic_answer,
)
from .citations import (
    CitationMetricResult,
    aggregate_citation_metrics,
    score_citations,
)
from .dataset import BenchmarkExample
from .retrieval_metrics import (
    RetrievalMetricResult,
    aggregate_retrieval_metrics,
    score_ranking,
    score_search_hits,
)


Retriever = Callable[[str, str, str], Sequence[SearchHit]]
AnswerFunction = Callable[[str, Sequence[SearchHit]], AnswerResult]
EvaluationMode = Literal["retrieved", "oracle"]


@dataclass(slots=True)
class EvaluationRecord:
    """Auditable result for one question."""

    question_id: str
    dataset: str
    split: str
    mode: EvaluationMode
    relevant_context_uid: str
    relevant_document_id: str | None
    retrieved_context_uids: list[str]
    answer_context_uids: list[str]
    retrieval: RetrievalMetricResult
    answer: AnswerMetricResult
    semantic_answer: SemanticAnswerMetricResult
    citations: CitationMetricResult
    latency_seconds: float
    result: AnswerResult | None = None
    invalid_reason: str = ""
    invalid_raw_response: str = ""
    invalid_usage: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Convert nested dataclasses, including raw model output, to JSON data."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class EvaluationSummary:
    """Macro metrics for one complete evaluation mode."""

    mode: EvaluationMode
    question_count: int
    retrieval: dict[str, float]
    answer: dict[str, float]
    semantic_answer: dict[str, float]
    citations: dict[str, float]
    mean_latency_seconds: float
    median_latency_seconds: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class EvaluationRun:
    """Summary plus per-question records for failure analysis."""

    summary: EvaluationSummary
    records: list[EvaluationRecord]


def _oracle_hit(example: BenchmarkExample) -> SearchHit:
    """Represent the supplied gold context through the standard evidence type."""

    if example.oracle_context is None:
        raise ValueError(
            "Oracle evaluation requires examples loaded with include_context=True."
        )
    return SearchHit(
        chunk_id=-1,
        context_uid=example.context_uid,
        document_id=example.document_id or example.context_uid,
        source_text=example.oracle_context,
        score=1.0,
        rank=1,
        retrieval_source="oracle",
        dataset=example.dataset,
        split=example.split,
        page_number=example.page_number,
        alias_path=example.file_name,
    )


def run_evaluation(
    examples: Iterable[BenchmarkExample],
    *,
    answer: AnswerFunction,
    config: AppConfig,
    mode: EvaluationMode,
    retrieve: Retriever | None = None,
) -> EvaluationRun:
    """Evaluate retrieved or oracle evidence with the same answer pipeline.

    Only strict answer parse failures count as invalid model output.  Retrieval,
    service, and programming errors remain visible by propagating to the caller.
    """

    if mode == "retrieved" and retrieve is None:
        raise ValueError("Retrieved evaluation requires a retrieval function.")

    records: list[EvaluationRecord] = []
    retrieval_rows: list[RetrievalMetricResult] = []
    answer_rows: list[AnswerMetricResult] = []
    semantic_answer_rows: list[SemanticAnswerMetricResult] = []
    citation_rows: list[CitationMetricResult] = []
    latencies: list[float] = []

    for example in examples:
        if example.program_answer is None:
            raise ValueError(
                "Evaluation labels are absent. Load examples with include_labels=True."
            )

        if mode == "oracle":
            hits = [_oracle_hit(example)]
        else:
            assert retrieve is not None  # Narrow the optional callable for type checkers.
            hits = list(retrieve(example.question, example.dataset, example.split))

        ordered_hits = sorted(hits, key=lambda hit: hit.rank)
        # Retrieval evaluation uses the complete Top-K list.  The paid answer
        # model receives only the independently configured answer-context limit.
        answer_hits = ordered_hits[: config.retrieval.answer_contexts]

        if example.document_id is not None:
            retrieval_score = score_ranking(
                (hit.document_id for hit in ordered_hits),
                [example.document_id],
            )
        else:
            retrieval_score = score_search_hits(hits, [example.context_uid])
        started = time.perf_counter()
        generated: AnswerResult | None
        invalid_reason = ""
        invalid_raw_response = ""
        invalid_usage: dict[str, Any] = {}
        try:
            generated = answer(example.question, answer_hits)
        except AnswerParseError as exc:
            generated = None
            invalid_reason = str(exc)
            invalid_raw_response = exc.raw_response
            invalid_usage = exc.usage
        latency = time.perf_counter() - started

        answer_score = score_answer(
            generated,
            example.program_answer,
            absolute_tolerance=config.evaluation.numeric_absolute_tolerance,
            relative_tolerance=config.evaluation.numeric_relative_tolerance,
        )
        semantic_answer_score = score_semantic_answer(
            generated,
            example.program_answer,
            example.original_answer,
            absolute_tolerance=config.evaluation.semantic_absolute_tolerance,
            relative_tolerance=config.evaluation.semantic_relative_tolerance,
        )
        citation_score = score_citations(
            generated.citations if generated is not None else [],
            answer_hits,
            [example.context_uid],
            [example.document_id] if example.document_id is not None else None,
        )
        record = EvaluationRecord(
            question_id=example.question_id,
            dataset=example.dataset,
            split=example.split,
            mode=mode,
            relevant_context_uid=example.context_uid,
            relevant_document_id=example.document_id,
            retrieved_context_uids=[hit.context_uid for hit in ordered_hits],
            answer_context_uids=[hit.context_uid for hit in answer_hits],
            retrieval=retrieval_score,
            answer=answer_score,
            semantic_answer=semantic_answer_score,
            citations=citation_score,
            latency_seconds=latency,
            result=generated,
            invalid_reason=invalid_reason,
            invalid_raw_response=invalid_raw_response,
            invalid_usage=invalid_usage,
        )
        records.append(record)
        retrieval_rows.append(retrieval_score)
        answer_rows.append(answer_score)
        semantic_answer_rows.append(semantic_answer_score)
        citation_rows.append(citation_score)
        latencies.append(latency)

    if not records:
        raise ValueError("The evaluation input contains no questions.")
    summary = EvaluationSummary(
        mode=mode,
        question_count=len(records),
        retrieval=aggregate_retrieval_metrics(retrieval_rows),
        answer=aggregate_answer_metrics(answer_rows),
        semantic_answer=aggregate_semantic_answer_metrics(semantic_answer_rows),
        citations=aggregate_citation_metrics(citation_rows),
        mean_latency_seconds=fmean(latencies),
        median_latency_seconds=median(latencies),
    )
    return EvaluationRun(summary=summary, records=records)


def compare_retrieved_with_oracle(
    retrieved: EvaluationSummary,
    oracle: EvaluationSummary,
) -> dict[str, float | None]:
    """Report the generation gap caused by retrieval instead of the answer model."""

    if retrieved.mode != "retrieved" or oracle.mode != "oracle":
        raise ValueError("Expected retrieved and oracle summaries in that order.")
    if retrieved.question_count != oracle.question_count:
        raise ValueError("Retrieved and oracle runs must cover the same question count.")

    retrieved_accuracy = retrieved.answer["correct"]
    oracle_accuracy = oracle.answer["correct"]
    retrieved_semantic_accuracy = retrieved.semantic_answer["semantic_correct"]
    oracle_semantic_accuracy = oracle.semantic_answer["semantic_correct"]
    return {
        "retrieved_context_accuracy": retrieved_accuracy,
        "oracle_context_accuracy": oracle_accuracy,
        "retrieved_to_oracle_ratio": (
            retrieved_accuracy / oracle_accuracy if oracle_accuracy > 0 else None
        ),
        "absolute_accuracy_gap": oracle_accuracy - retrieved_accuracy,
        "retrieved_context_semantic_accuracy": retrieved_semantic_accuracy,
        "oracle_context_semantic_accuracy": oracle_semantic_accuracy,
        "semantic_retrieved_to_oracle_ratio": (
            retrieved_semantic_accuracy / oracle_semantic_accuracy
            if oracle_semantic_accuracy > 0
            else None
        ),
        "semantic_absolute_accuracy_gap": (
            oracle_semantic_accuracy - retrieved_semantic_accuracy
        ),
    }


def write_evaluation_records(path: str | Path, run: EvaluationRun) -> None:
    """Atomically save JSONL records, including raw outputs and failure reasons."""

    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in run.records:
            handle.write(
                json.dumps(record.as_dict(), ensure_ascii=False, sort_keys=True) + "\n"
            )
    temporary.replace(target)

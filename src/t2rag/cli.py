"""Manual command-line entry point for every pipeline stage.

Nothing runs at import time. Each subcommand performs only the stage named by
the user, which makes expensive model and API operations explicit.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import asdict
from itertools import islice
from pathlib import Path
from typing import Any

from t2rag.config import AppConfig, load_config
from t2rag.errors import T2RagError
from t2rag.evaluation.answer_metrics import score_answer, score_semantic_answer
from t2rag.evaluation.dataset import BenchmarkExample, BenchmarkLoader
from t2rag.evaluation.freeze import create_freeze_manifest, write_run_manifest
from t2rag.evaluation.judge import (
    OllamaEvaluationJudge,
    build_judge_payload,
    summarize_judge_reviews,
)
from t2rag.evaluation.predictions import (
    PredictionRecord,
    answer_result_from_dict,
    load_predictions,
    question_sha256,
    rescore_saved_answer_rows,
    score_saved_predictions,
    write_predictions,
)
from t2rag.evaluation.retrieval_metrics import (
    aggregate_retrieval_metrics,
    score_ranking,
    score_search_hits,
)
from t2rag.evaluation.runner import (
    compare_retrieved_with_oracle,
    run_evaluation,
    write_evaluation_records,
)
from t2rag.generation.answerer import AnswerParseError, EvidenceAnswerer
from t2rag.ingestion.database import RegistryDatabase
from t2rag.ingestion.descriptions import (
    TEXT_PROMPT_VERSION,
    VISION_PROMPT_VERSION,
    OllamaDescriptionService,
)
from t2rag.ingestion.registry import build_registry
from t2rag.ingestion.raw_pdf import extract_raw_pdfs
from t2rag.json_utils import write_json_atomic
from t2rag.logging_utils import configure_logging
from t2rag.model_manifest import capture_model_manifest
from t2rag.models import AnswerResult, SearchHit
from t2rag.retrieval.lexical import SQLiteBM25Index
from t2rag.retrieval.service import HybridSearchService
from t2rag.ui.gradio_chat import launch_gradio_chat


RetrievalFunction = Callable[[str], list[SearchHit]]


def _parser() -> argparse.ArgumentParser:
    """Create the command tree in one place so help text stays consistent."""

    parser = argparse.ArgumentParser(
        prog="t2rag",
        description="Build and evaluate the T2-RAGBench financial RAG pipeline.",
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="YAML configuration path. Default: configs/default.yaml",
    )
    parser.add_argument("--verbose", action="store_true", help="Show debug logging.")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("status", help="Show registry and index status without models.")
    commands.add_parser("build-registry", help="Build the canonical SQLite registry.")

    raw_pdf = commands.add_parser(
        "extract-raw-pdfs",
        help="Build the separate raw-PDF comparison corpus.",
    )
    raw_pdf.add_argument("--limit", type=int, default=None)
    raw_pdf.add_argument(
        "--render", choices=("none", "weak", "all"), default="weak"
    )

    describe = commands.add_parser("describe", help="Generate optional local descriptions.")
    describe.add_argument("--mode", choices=("text", "vision"), required=True)
    describe.add_argument("--limit", type=int, default=None)

    commands.add_parser("build-bm25", help="Build the lexical baseline only.")
    commands.add_parser("build-dense", help="Build BM25 and the FAISS dense index.")
    commands.add_parser(
        "capture-models", help="Save exact Ollama digests and cloud model IDs."
    )

    retrieval = commands.add_parser(
        "evaluate-retrieval", help="Evaluate one retrieval configuration."
    )
    _add_dataset_options(retrieval)
    retrieval.add_argument(
        "--mode",
        choices=("bm25", "dense", "hybrid", "reranked", "routed"),
        required=True,
    )
    retrieval.add_argument("--limit", type=int, default=None)
    retrieval.add_argument("--frozen-manifest", default=None)
    retrieval.add_argument("--output", default=None)

    answers = commands.add_parser(
        "evaluate-answers", help="Generate and evaluate retrieved or oracle answers."
    )
    _add_dataset_options(answers)
    answers.add_argument(
        "--mode", choices=("retrieved", "oracle", "both"), default="retrieved"
    )
    answers.add_argument(
        "--retrieval-mode",
        choices=("bm25", "dense", "hybrid", "reranked", "routed"),
        default="reranked",
    )
    answers.add_argument("--limit", type=int, default=None)
    answers.add_argument("--predictions-only", action="store_true")
    answers.add_argument("--frozen-manifest", default=None)
    answers.add_argument("--output", default=None)

    judge = commands.add_parser(
        "judge-answers",
        help="Use the optional local Qwen judge to review disputed saved answers.",
    )
    _add_dataset_options(judge)
    judge.add_argument("--records", required=True)
    judge.add_argument("--limit", type=int, default=None)
    judge.add_argument(
        "--all",
        action="store_true",
        help="Review all saved answers instead of deterministic disputes only.",
    )
    judge.add_argument("--frozen-manifest", default=None)
    judge.add_argument("--output", default=None)

    rescore = commands.add_parser(
        "rescore-answers",
        help="Recompute deterministic metrics for saved development answers.",
    )
    _add_dataset_options(rescore)
    rescore.add_argument("--records", required=True)
    rescore.add_argument("--output", default=None)

    score = commands.add_parser(
        "score-predictions", help="Score saved predictions after explicitly opening labels."
    )
    _add_dataset_options(score, default_split="test")
    score.add_argument("--predictions", required=True)
    score.add_argument("--frozen-manifest", required=True)
    score.add_argument("--allow-test-labels", action="store_true", required=True)
    score.add_argument("--output", default=None)

    ask = commands.add_parser("ask", help="Answer one question with citations.")
    ask.add_argument("question")
    ask.add_argument(
        "--retrieval-mode",
        choices=("bm25", "dense", "hybrid", "reranked", "routed"),
        default="reranked",
    )

    chat_ui = commands.add_parser(
        "chat-ui",
        help="Open a local Gradio chat page for manual RAG testing.",
    )
    chat_ui.add_argument("--host", default="127.0.0.1")
    chat_ui.add_argument("--port", type=int, default=7860)
    chat_ui.add_argument(
        "--retrieval-mode",
        choices=("bm25", "dense", "hybrid", "reranked", "routed"),
        default="reranked",
    )
    chat_ui.add_argument(
        "--share",
        action="store_true",
        help="Create a public Gradio share link. Disabled by default.",
    )
    chat_ui.add_argument(
        "--no-browser",
        action="store_true",
        help="Start the server without opening the browser automatically.",
    )

    freeze = commands.add_parser("freeze", help="Hash the selected system before test use.")
    freeze.add_argument("--output", default=None)
    freeze.add_argument("--notes", default="")
    return parser


def _add_dataset_options(parser: argparse.ArgumentParser, *, default_split: str = "dev") -> None:
    parser.add_argument(
        "--dataset",
        choices=("all", "finqa", "tat-dqa", "convfinqa"),
        default="all",
    )
    parser.add_argument(
        "--split", choices=("train", "dev", "test", "all"), default=default_split
    )


def _require_registry(config: AppConfig) -> None:
    if not config.paths.registry_db.is_file():
        raise FileNotFoundError(
            f"Registry does not exist: {config.paths.registry_db}. Run build-registry first."
        )


def _datasets(dataset: str, split: str) -> list[str]:
    """Expand `all` without inventing an official ConvFinQA split."""

    if dataset == "all":
        if split == "all":
            raise ValueError("Use --dataset convfinqa with --split all explicitly.")
        return ["FinQA", "TAT-DQA"]
    if dataset == "finqa":
        return ["FinQA"]
    if dataset == "tat-dqa":
        return ["TAT-DQA"]
    if split != "all":
        raise ValueError("ConvFinQA requires --split all.")
    return ["ConvFinQA"]


def _examples(
    loader: BenchmarkLoader,
    *,
    dataset: str,
    split: str,
    include_labels: bool,
    include_context: bool,
    limit: int | None,
) -> Iterator[BenchmarkExample]:
    """Stream one or two datasets and apply one global optional limit."""

    if limit is not None and limit <= 0:
        raise ValueError("--limit must be positive.")

    def combined() -> Iterator[BenchmarkExample]:
        for name in _datasets(dataset, split):
            yield from loader.load(
                name,
                split,
                include_labels=include_labels,
                include_context=include_context,
            )

    return combined() if limit is None else islice(combined(), limit)


def _service(config: AppConfig, mode: str) -> HybridSearchService:
    return HybridSearchService.from_config(
        config,
        enable_dense=mode in {"dense", "hybrid", "reranked", "routed"},
        enable_reranker=mode in {"reranked", "routed"},
    )


def _retriever(
    service: HybridSearchService,
    mode: str,
    *,
    top_k: int | None = None,
    dataset: str | None = None,
    split: str | None = None,
) -> RetrievalFunction:
    if mode == "bm25":
        return lambda question: service.search_lexical(
            question,
            top_k=top_k,
            dataset=dataset,
            split=split,
        )
    if mode == "dense":
        return lambda question: service.search_dense(
            question,
            top_k=top_k,
            dataset=dataset,
            split=split,
        )
    if mode == "hybrid":
        return lambda question: service.search(
            question,
            top_k=top_k,
            use_dense=True,
            use_reranker=False,
            dataset=dataset,
            split=split,
        )
    if mode == "reranked":
        return lambda question: service.search(
            question,
            top_k=top_k,
            use_dense=True,
            use_reranker=True,
            dataset=dataset,
            split=split,
        )
    if mode == "routed":
        return lambda question: service.search_routed(
            question,
            top_k=top_k,
            use_reranker=True,
            dataset=dataset,
            split=split,
        )[1]
    raise ValueError(f"Unknown retrieval mode: {mode}")


def _default_output(config: AppConfig, name: str) -> Path:
    return config.paths.output_dir / name


def _command_status(config: AppConfig) -> dict[str, Any]:
    registry: dict[str, int] | None = None
    if config.paths.registry_db.is_file():
        with RegistryDatabase(config.paths.registry_db) as database:
            registry = database.summary()
    lexical_path = config.paths.index_dir / "lexical.sqlite3"
    return {
        "dataset_root": str(config.paths.dataset_root),
        "registry_path": str(config.paths.registry_db),
        "registry": registry,
        "bm25_chunks": SQLiteBM25Index(lexical_path).count(),
        "dense_index_exists": (config.paths.index_dir / "dense.faiss").is_file(),
        "dense_manifest_exists": (
            config.paths.index_dir / "dense.manifest.json"
        ).is_file(),
        "model_manifest_exists": (
            config.paths.work_root / "model-manifest.json"
        ).is_file(),
    }


def _command_build_registry(config: AppConfig) -> dict[str, Any]:
    return asdict(build_registry(config))


def _command_describe(config: AppConfig, *, mode: str, limit: int | None) -> dict[str, Any]:
    _require_registry(config)
    if limit is not None and limit <= 0:
        raise ValueError("--limit must be positive.")
    generated = 0
    with RegistryDatabase(config.paths.registry_db) as database:
        with OllamaDescriptionService(config) as descriptions:
            if mode == "text":
                rows = database.iter_text_description_inputs(
                    model=config.models.text_description_model,
                    prompt_version=TEXT_PROMPT_VERSION,
                    limit=limit,
                )
                for row in rows:
                    descriptions.generate_and_store_text(
                        database,
                        document_id=str(row["document_id"]),
                        context_uid=str(row["context_uid"]),
                        alias_id=int(row["alias_id"]),
                        source_text=str(row["source_text"]),
                    )
                    generated += 1
            else:
                rows = database.iter_vision_description_inputs(
                    model=config.models.vision_model,
                    prompt_version=VISION_PROMPT_VERSION,
                    limit=limit,
                )
                for row in rows:
                    image_path = config.paths.dataset_root / str(row["relative_path"])
                    descriptions.generate_and_store_page(
                        database,
                        document_id=str(row["document_id"]),
                        alias_id=int(row["alias_id"]),
                        page_number=int(row["page_number"]),
                        image_path=image_path,
                    )
                    generated += 1
                remaining = None if limit is None else max(limit - generated, 0)
                if remaining is None or remaining > 0:
                    rendered_rows = database.iter_rendered_vision_inputs(
                        model=config.models.vision_model,
                        prompt_version=VISION_PROMPT_VERSION,
                        limit=remaining,
                    )
                    for row in rendered_rows:
                        descriptions.generate_and_store_page(
                            database,
                            document_id=str(row["document_id"]),
                            alias_id=int(row["alias_id"]),
                            page_number=int(row["page_number"]),
                            image_path=Path(str(row["image_path"])),
                        )
                        generated += 1
    return {"mode": mode, "generated": generated}


def _command_build_index(config: AppConfig, *, dense: bool) -> dict[str, Any]:
    _require_registry(config)
    with RegistryDatabase(config.paths.registry_db) as database:
        chunks = database.iter_chunks(
            include_descriptions=config.retrieval.include_descriptions
        )
        with HybridSearchService.from_config(
            config, enable_dense=dense, enable_reranker=False
        ) as service:
            return service.build_indexes(chunks)


def _command_evaluate_retrieval(
    config: AppConfig,
    *,
    dataset: str,
    split: str,
    mode: str,
    limit: int | None,
    frozen_manifest: str | None,
    output: str | None,
) -> dict[str, Any]:
    loader = BenchmarkLoader(config, frozen_manifest_path=frozen_manifest)
    examples = _examples(
        loader,
        dataset=dataset,
        split=split,
        include_labels=False,
        include_context=False,
        limit=limit,
    )
    rows = []
    scores = []
    with _service(config, mode) as service:
        for example in examples:
            retrieve = _retriever(
                service,
                mode,
                top_k=config.evaluation.retrieval_top_k,
                dataset=example.dataset,
                split=example.split,
            )
            hits = retrieve(example.question)
            if example.document_id is not None:
                score = score_ranking(
                    (hit.document_id for hit in sorted(hits, key=lambda item: item.rank)),
                    [example.document_id],
                )
            else:
                score = score_search_hits(hits, [example.context_uid])
            scores.append(score)
            rows.append(
                {
                    "question_id": example.question_id,
                    "dataset": example.dataset,
                    "split": example.split,
                    "relevant_context_uid": example.context_uid,
                    "relevant_document_id": example.document_id,
                    "retrieved": [
                        {
                            "rank": hit.rank,
                            "chunk_id": hit.chunk_id,
                            "context_uid": hit.context_uid,
                            "document_id": hit.document_id,
                            "score": hit.score,
                            "source": hit.retrieval_source,
                        }
                        for hit in hits
                    ],
                    "metrics": asdict(score),
                }
            )
    if not scores:
        raise ValueError("No retrieval questions were evaluated.")
    result = {
        "mode": mode,
        "dataset": dataset,
        "split": split,
        "question_count": len(scores),
        "metrics": aggregate_retrieval_metrics(scores),
        "records": rows,
    }
    target = Path(output) if output else _default_output(
        config, f"retrieval-{dataset}-{split}-{mode}.json"
    )
    write_json_atomic(target.resolve(), result)
    return {**result, "records": f"saved:{target.resolve()}"}


def _prediction_run(
    config: AppConfig,
    *,
    examples: Iterable[BenchmarkExample],
    retrieval_mode: str,
    output: Path,
) -> dict[str, Any]:
    """Generate predictions without opening labels or calculating metrics."""

    records: list[PredictionRecord] = []
    answerer = EvidenceAnswerer(config)
    with _service(config, retrieval_mode) as service:
        for example in examples:
            retrieve = _retriever(
                service,
                retrieval_mode,
                top_k=config.evaluation.retrieval_top_k,
                dataset=example.dataset,
                split=example.split,
            )
            hits = retrieve(example.question)
            answer_hits = sorted(hits, key=lambda hit: hit.rank)[
                : config.retrieval.answer_contexts
            ]
            started = time.perf_counter()
            result: AnswerResult | None
            invalid_reason = ""
            invalid_raw_response = ""
            invalid_usage: dict[str, Any] = {}
            try:
                result = answerer.answer(example.question, answer_hits)
            except AnswerParseError as exc:
                result = None
                invalid_reason = str(exc)
                invalid_raw_response = exc.raw_response
                invalid_usage = exc.usage
            records.append(
                PredictionRecord(
                    question_id=example.question_id,
                    dataset=example.dataset,
                    split=example.split,
                    question_sha256=question_sha256(example.question),
                    hits=hits,
                    result=result,
                    latency_seconds=time.perf_counter() - started,
                    invalid_reason=invalid_reason,
                    invalid_raw_response=invalid_raw_response,
                    invalid_usage=invalid_usage,
                )
            )
    write_predictions(output, records)
    return {"prediction_count": len(records), "output": str(output)}


def _command_evaluate_answers(
    config: AppConfig,
    *,
    dataset: str,
    split: str,
    mode: str,
    retrieval_mode: str,
    limit: int | None,
    predictions_only: bool,
    frozen_manifest: str | None,
    output: str | None,
) -> dict[str, Any]:
    # Prediction-only runs never open labels or oracle contexts. Treat the
    # flag as an explicit request for retrieved mode even when the parser's
    # normal evaluation default is ``both``.
    if predictions_only:
        mode = "retrieved"
    loader = BenchmarkLoader(config, frozen_manifest_path=frozen_manifest)
    examples = list(
        _examples(
            loader,
            dataset=dataset,
            split=split,
            include_labels=not predictions_only,
            include_context=mode in {"oracle", "both"},
            limit=limit,
        )
    )
    if not examples:
        raise ValueError("No answer questions were loaded.")

    base = Path(output).resolve() if output else _default_output(
        config, f"answers-{dataset}-{split}"
    ).resolve()
    if predictions_only:
        target = base if base.suffix else base.with_suffix(".predictions.jsonl")
        return _prediction_run(
            config,
            examples=examples,
            retrieval_mode=retrieval_mode,
            output=target,
        )

    answerer = EvidenceAnswerer(config)
    summaries: dict[str, Any] = {}
    retrieved_summary = None
    oracle_summary = None
    if mode in {"retrieved", "both"}:
        with _service(config, retrieval_mode) as service:
            run = run_evaluation(
                examples,
                answer=answerer.answer,
                config=config,
                mode="retrieved",
                retrieve=lambda question, dataset_name, split_name: _retriever(
                    service,
                    retrieval_mode,
                    top_k=config.evaluation.retrieval_top_k,
                    dataset=dataset_name,
                    split=split_name,
                )(question),
            )
        retrieved_summary = run.summary
        summaries["retrieved"] = run.summary.as_dict()
        write_evaluation_records(base.with_suffix(".retrieved.jsonl"), run)
    if mode in {"oracle", "both"}:
        run = run_evaluation(
            examples,
            answer=answerer.answer,
            config=config,
            mode="oracle",
        )
        oracle_summary = run.summary
        summaries["oracle"] = run.summary.as_dict()
        write_evaluation_records(base.with_suffix(".oracle.jsonl"), run)
    if retrieved_summary is not None and oracle_summary is not None:
        summaries["oracle_gap"] = compare_retrieved_with_oracle(
            retrieved_summary, oracle_summary
        )

    summary_path = base.with_suffix(".summary.json")
    write_json_atomic(summary_path, summaries)
    return {"summaries": summaries, "output": str(summary_path)}


def _load_jsonl_objects(path: str | Path) -> list[dict[str, Any]]:
    """Load a local JSONL artifact and reject malformed rows."""

    source = Path(path).resolve()
    rows: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid evaluation JSON at {source}:{line_number}."
                ) from exc
            if not isinstance(value, dict):
                raise ValueError(
                    f"Evaluation row {line_number} must be a JSON object."
                )
            rows.append(value)
    if not rows:
        raise ValueError("The evaluation record file is empty.")
    return rows


def _command_judge_answers(
    config: AppConfig,
    *,
    dataset: str,
    split: str,
    records_path: str,
    limit: int | None,
    review_all: bool,
    frozen_manifest: str | None,
    output: str | None,
) -> dict[str, Any]:
    """Review saved answers locally without changing deterministic metrics."""

    if limit is not None and limit <= 0:
        raise ValueError("--limit must be positive.")
    loader = BenchmarkLoader(config, frozen_manifest_path=frozen_manifest)
    examples = {
        (example.dataset, example.question_id): example
        for example in _examples(
            loader,
            dataset=dataset,
            split=split,
            include_labels=True,
            include_context=True,
            limit=None,
        )
    }
    saved_rows = _load_jsonl_objects(records_path)
    reviews: list[dict[str, Any]] = []
    skipped_uncontested = 0

    with OllamaEvaluationJudge(config) as judge:
        for saved in saved_rows:
            key = (str(saved.get("dataset", "")), str(saved.get("question_id", "")))
            example = examples.get(key)
            if example is None:
                raise ValueError(f"Saved answer does not match the requested data: {key}.")
            if example.program_answer is None or example.original_answer is None:
                raise ValueError("Judge review requires both answer labels.")
            if example.oracle_context is None:
                raise ValueError("Judge review requires the source context.")

            generated = answer_result_from_dict(saved.get("result"))
            official = score_answer(
                generated,
                example.program_answer,
                absolute_tolerance=config.evaluation.numeric_absolute_tolerance,
                relative_tolerance=config.evaluation.numeric_relative_tolerance,
            )
            semantic = score_semantic_answer(
                generated,
                example.program_answer,
                example.original_answer,
                absolute_tolerance=config.evaluation.semantic_absolute_tolerance,
                relative_tolerance=config.evaluation.semantic_relative_tolerance,
            )
            disputed = bool(
                official.correct != semantic.semantic_correct
                or semantic.label_conflict
                or official.invalid_output
            )
            if not review_all and not disputed:
                skipped_uncontested += 1
                continue
            if limit is not None and len(reviews) >= limit:
                break

            invalid_raw_response = str(saved.get("invalid_raw_response", ""))
            decision = judge.judge(
                build_judge_payload(
                    question=example.question,
                    source_context=example.oracle_context,
                    program_answer=example.program_answer,
                    original_answer=example.original_answer,
                    generated=generated,
                    invalid_raw_response=invalid_raw_response,
                )
            )
            reviews.append(
                {
                    "question_id": example.question_id,
                    "dataset": example.dataset,
                    "split": example.split,
                    "mode": saved.get("mode", "unknown"),
                    "official_answer": official.as_dict(),
                    "semantic_answer": semantic.as_dict(),
                    "judge": decision.as_dict(),
                }
            )

    target = Path(output).resolve() if output else _default_output(
        config, f"judge-{dataset}-{split}.json"
    ).resolve()
    review_summary = summarize_judge_reviews(reviews)
    artifact = {
        "diagnostic_only": True,
        "official_scores_changed": False,
        "judge_model": config.models.evaluation_judge_model,
        "source_records": str(Path(records_path).resolve()),
        "reviewed_count": len(reviews),
        "skipped_uncontested": skipped_uncontested,
        **review_summary,
        "reviews": reviews,
    }
    write_json_atomic(target, artifact)
    return {
        "diagnostic_only": True,
        "official_scores_changed": False,
        "judge_model": config.models.evaluation_judge_model,
        "reviewed_count": len(reviews),
        "skipped_uncontested": skipped_uncontested,
        **review_summary,
        "output": str(target),
    }


def _command_score_predictions(
    config: AppConfig,
    *,
    dataset: str,
    split: str,
    predictions: str,
    frozen_manifest: str,
    output: str | None,
) -> dict[str, Any]:
    if split != "test":
        raise ValueError("score-predictions is reserved for the final test split.")
    # The command-line flag is an explicit one-run authorization. The frozen
    # YAML file remains unchanged, so its integrity hash still verifies.
    config.evaluation.allow_test_questions = True
    config.evaluation.allow_test_labels = True
    loader = BenchmarkLoader(config, frozen_manifest_path=frozen_manifest)
    examples = _examples(
        loader,
        dataset=dataset,
        split=split,
        include_labels=True,
        include_context=False,
        limit=None,
    )
    result = score_saved_predictions(load_predictions(predictions), examples, config)
    target = Path(output).resolve() if output else _default_output(
        config, f"final-test-{dataset}-metrics.json"
    ).resolve()
    write_json_atomic(target, result)
    write_run_manifest(
        target.with_name(target.stem + ".run.json"),
        freeze_manifest_path=frozen_manifest,
        dataset=dataset,
        split=split,
        settings={"predictions": str(Path(predictions).resolve())},
        metrics=result,
    )
    return {**result, "output": str(target)}


def _command_rescore_answers(
    config: AppConfig,
    *,
    dataset: str,
    split: str,
    records_path: str,
    output: str | None,
) -> dict[str, Any]:
    """Rescore saved train or dev answers without retrieval or model calls."""

    if split not in {"train", "dev"}:
        raise ValueError(
            "rescore-answers supports train or dev only; use the frozen test scorer "
            "for test predictions."
        )
    loader = BenchmarkLoader(config)
    examples = _examples(
        loader,
        dataset=dataset,
        split=split,
        include_labels=True,
        include_context=False,
        limit=None,
    )
    summary = rescore_saved_answer_rows(
        _load_jsonl_objects(records_path),
        examples,
        config,
    )
    target = Path(output).resolve() if output else _default_output(
        config, f"rescore-{dataset}-{split}.json"
    ).resolve()
    artifact = {
        **summary,
        "source_records": str(Path(records_path).resolve()),
    }
    write_json_atomic(target, artifact)
    compact = {key: value for key, value in artifact.items() if key != "records"}
    return {**compact, "output": str(target)}


def _command_ask(config: AppConfig, *, question: str, retrieval_mode: str) -> dict[str, Any]:
    with _service(config, retrieval_mode) as service:
        hits = _retriever(service, retrieval_mode)(question)
    result = EvidenceAnswerer(config).answer(question, hits)
    return {
        "answer": asdict(result),
        "retrieval": [
            {
                "rank": hit.rank,
                "chunk_id": hit.chunk_id,
                "context_uid": hit.context_uid,
                "document_id": hit.document_id,
                "alias_path": hit.alias_path,
                "page_number": hit.page_number,
                "score": hit.score,
                "source": hit.retrieval_source,
            }
            for hit in hits
        ],
    }


def _command_freeze(
    config: AppConfig,
    *,
    config_path: str,
    output: str | None,
    notes: str,
) -> dict[str, Any]:
    # Resolve the source tree from this installed module. This remains correct
    # when the user stores a custom YAML file outside the configs directory.
    project_root = Path(__file__).resolve().parents[2]
    model_manifest = config.paths.work_root / "model-manifest.json"
    if not model_manifest.is_file():
        raise FileNotFoundError(
            "Model manifest does not exist. Run capture-models before freeze."
        )
    target = Path(output).resolve() if output else _default_output(
        config, "freeze-manifest.json"
    ).resolve()
    manifest = create_freeze_manifest(
        target,
        config_path=config_path,
        artifact_paths={
            "source": project_root / "src",
            "dependency_specification": project_root / "pyproject.toml",
            "registry": config.paths.registry_db,
            "indexes": config.paths.index_dir,
            "models": model_manifest,
        },
        notes=notes,
    )
    return asdict(manifest)


def _dispatch(config: AppConfig, args: argparse.Namespace) -> dict[str, Any]:
    if args.command == "status":
        return _command_status(config)
    if args.command == "build-registry":
        return _command_build_registry(config)
    if args.command == "extract-raw-pdfs":
        _require_registry(config)
        return asdict(
            extract_raw_pdfs(config, limit=args.limit, render_mode=args.render)
        )
    if args.command == "describe":
        return _command_describe(config, mode=args.mode, limit=args.limit)
    if args.command == "build-bm25":
        return _command_build_index(config, dense=False)
    if args.command == "build-dense":
        return _command_build_index(config, dense=True)
    if args.command == "capture-models":
        return capture_model_manifest(
            config, config.paths.work_root / "model-manifest.json"
        )
    if args.command == "evaluate-retrieval":
        return _command_evaluate_retrieval(
            config,
            dataset=args.dataset,
            split=args.split,
            mode=args.mode,
            limit=args.limit,
            frozen_manifest=args.frozen_manifest,
            output=args.output,
        )
    if args.command == "evaluate-answers":
        return _command_evaluate_answers(
            config,
            dataset=args.dataset,
            split=args.split,
            mode=args.mode,
            retrieval_mode=args.retrieval_mode,
            limit=args.limit,
            predictions_only=args.predictions_only,
            frozen_manifest=args.frozen_manifest,
            output=args.output,
        )
    if args.command == "judge-answers":
        return _command_judge_answers(
            config,
            dataset=args.dataset,
            split=args.split,
            records_path=args.records,
            limit=args.limit,
            review_all=args.all,
            frozen_manifest=args.frozen_manifest,
            output=args.output,
        )
    if args.command == "rescore-answers":
        return _command_rescore_answers(
            config,
            dataset=args.dataset,
            split=args.split,
            records_path=args.records,
            output=args.output,
        )
    if args.command == "score-predictions":
        return _command_score_predictions(
            config,
            dataset=args.dataset,
            split=args.split,
            predictions=args.predictions,
            frozen_manifest=args.frozen_manifest,
            output=args.output,
        )
    if args.command == "ask":
        return _command_ask(
            config, question=args.question, retrieval_mode=args.retrieval_mode
        )
    if args.command == "chat-ui":
        _require_registry(config)
        return launch_gradio_chat(
            config,
            host=args.host,
            port=args.port,
            share=args.share,
            inbrowser=not args.no_browser,
            retrieval_mode=args.retrieval_mode,
        )
    if args.command == "freeze":
        return _command_freeze(
            config,
            config_path=args.config,
            output=args.output,
            notes=args.notes,
        )
    raise ValueError(f"Unsupported command: {args.command}")


def main(argv: Sequence[str] | None = None) -> int:
    """Parse one manual command, print JSON output, and return an exit code."""

    parser = _parser()
    args = parser.parse_args(argv)
    configure_logging(args.verbose)
    try:
        config = load_config(args.config)
        result = _dispatch(config, args)
    except (T2RagError, FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

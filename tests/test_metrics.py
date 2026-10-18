"""Unit tests for ranking, answer, citation, and strict-output metrics."""

import json
import math
from dataclasses import asdict
from decimal import Decimal

import pytest

from t2rag.evaluation.answer_metrics import (
    labels_semantically_conflict,
    numeric_answers_match,
    parse_numeric_answer,
    score_answer,
    score_semantic_answer,
)
from t2rag.evaluation.dataset import BenchmarkExample
from t2rag.evaluation.predictions import answer_result_from_dict, rescore_saved_answer_rows
from t2rag.evaluation.citations import score_citations
from t2rag.evaluation.retrieval_metrics import score_ranking
from t2rag.evaluation.runner import run_evaluation
from t2rag.generation.answerer import (
    AnswerParseError,
    EvidenceAnswerer,
    parse_answer_response,
)
from t2rag.generation.deepinfra import DeepInfraResponse
from t2rag.generation.prompts import ANSWER_PROMPT_VERSION, build_answer_messages
from t2rag.config import (
    AppConfig,
    GenerationSettings,
    PathSettings,
    RetrievalSettings,
)
from t2rag.models import AnswerResult, Citation, SearchHit


def _hit(context_uid: str = "gold") -> SearchHit:
    return SearchHit(
        chunk_id=7,
        context_uid=context_uid,
        document_id="doc-1",
        source_text="Revenue was $532 million.",
        score=1.0,
        rank=1,
        retrieval_source="test",
        page_number=28,
        alias_path="pdf/report/page_28.pdf",
    )


def _citation(context_uid: str = "gold") -> Citation:
    return Citation(
        document_id="doc-1",
        context_uid=context_uid,
        alias_path="pdf/report/page_28.pdf",
        page_number=28,
        chunk_id=7,
    )


def _answer_payload(
    hit: SearchHit,
    *,
    answer: str,
    normalized_answer: str,
    calculation: str,
    result_unit: str = "percent",
) -> str:
    return json.dumps(
        {
            "answer": answer,
            "normalized_answer": normalized_answer,
            "result_unit": result_unit,
            "calculation": calculation,
            "citations": [
                {
                    "document_id": hit.document_id,
                    "context_uid": hit.context_uid,
                    "alias_path": hit.alias_path,
                    "page_number": hit.page_number,
                    "chunk_id": hit.chunk_id,
                }
            ],
            "abstained": False,
            "confidence": 0.9,
        }
    )


def test_single_relevant_retrieval_metrics_deduplicate_contexts() -> None:
    result = score_ranking(["wrong", "gold", "gold", "later"], ["gold"])

    assert result.recall_at_1 == 0.0
    assert result.recall_at_5 == 1.0
    assert result.hit_at_1 == 0.0
    assert result.hit_at_5 == 1.0
    assert result.mrr_at_10 == 0.5
    assert result.ndcg_at_10 == pytest.approx(1.0 / math.log2(3))


def test_multiple_relevant_recall_and_ndcg() -> None:
    result = score_ranking(["a", "wrong", "b"], ["a", "b"])
    expected_dcg = 1.0 + 1.0 / math.log2(4)
    ideal_dcg = 1.0 + 1.0 / math.log2(3)

    assert result.recall_at_1 == 0.5
    assert result.recall_at_5 == 1.0
    assert result.mrr_at_10 == 1.0
    assert result.ndcg_at_10 == pytest.approx(expected_dcg / ideal_dcg)


def test_numeric_normalization_preserves_benchmark_percentage_scale() -> None:
    assert parse_numeric_answer("$1,234.50 million") == Decimal("1234.50")
    assert parse_numeric_answer("(12.5%)") == Decimal("-12.5")
    assert parse_numeric_answer("13.2%") == Decimal("13.2")
    assert parse_numeric_answer("from 10 to 12") is None


def test_numeric_match_uses_absolute_and_relative_tolerances() -> None:
    assert numeric_answers_match("1000.05", "1000", relative_tolerance=1e-4)
    assert not numeric_answers_match("1000.2", "1000", relative_tolerance=1e-4)


def test_answer_metrics_distinguish_invalid_output_and_abstention() -> None:
    invalid = score_answer(None, "10")
    abstention = score_answer(
        AnswerResult("", "", "", [], True),
        "10",
    )

    assert invalid.invalid_output == 1.0
    assert invalid.abstained == 0.0
    assert abstention.invalid_output == 0.0
    assert abstention.abstained == 1.0


def test_semantic_metric_handles_percent_scale_without_changing_official_score() -> None:
    prediction = AnswerResult(
        answer="93.5%",
        normalized_answer="93.5",
        calculation="(193.5 - 100) / 100 * 100",
        citations=[],
        abstained=False,
    )

    official = score_answer(prediction, "0.935")
    semantic = score_semantic_answer(prediction, "0.935", "93.5%")

    assert official.correct == 0.0
    assert semantic.semantic_correct == 1.0
    assert semantic.label_conflict == 0.0


def test_semantic_metric_uses_verified_calculator_precision() -> None:
    prediction = AnswerResult(
        answer="24.7%",
        normalized_answer="24.7",
        calculation="60 / 243 * 100",
        citations=[],
        abstained=False,
        usage={"calculator_result": "24.691358024691358"},
    )

    semantic = score_semantic_answer(
        prediction,
        "24.691358024691358",
        "24.69%",
    )

    assert semantic.semantic_correct == 1.0
    assert semantic.calculator_numeric_match == 1.0
    assert semantic.calculator_consistent == 1.0


def test_calculator_does_not_rescue_a_contradictory_display_answer() -> None:
    prediction = AnswerResult(
        answer="10%",
        normalized_answer="10",
        calculation="20",
        citations=[],
        abstained=False,
        usage={"calculator_result": "20"},
    )

    semantic = score_semantic_answer(prediction, "20", "20%")

    assert semantic.calculator_numeric_match == 1.0
    assert semantic.calculator_consistent == 0.0
    assert semantic.semantic_correct == 0.0


def test_semantic_metric_flags_conflicting_dataset_labels() -> None:
    prediction = AnswerResult("18.6 million", "18.6", "555.3 - 536.7", [], False)

    assert labels_semantically_conflict("19.2", "18.6")
    semantic = score_semantic_answer(prediction, "19.2", "18.6")

    assert semantic.semantic_correct == 1.0
    assert semantic.label_conflict == 1.0


def test_semantic_metric_rejects_wrong_percent_despite_conflicting_labels() -> None:
    prediction = AnswerResult(
        answer="19.41%",
        normalized_answer="19.41",
        calculation="(1703 - 1371) / 1703 * 100",
        citations=[],
        abstained=False,
        usage={"calculator_result": "19.495008807985907"},
    )

    semantic = score_semantic_answer(
        prediction,
        "0.19495008807985909",
        "25%",
    )

    assert semantic.label_conflict == 1.0
    assert semantic.calculator_consistent == 0.0
    assert semantic.semantic_correct == 0.0


def test_citation_must_be_retrieved_and_point_to_gold_context() -> None:
    hit = _hit()
    correct = score_citations([_citation()], [hit], ["gold"])
    wrong_gold = score_citations([_citation()], [hit], ["different-gold"])
    canonical_page_match = score_citations(
        [_citation()],
        [hit],
        ["different-context-on-same-page"],
        ["doc-1"],
    )

    assert correct.citation_validity == 1.0
    assert correct.citation_correct == 1.0
    assert wrong_gold.citation_validity == 1.0
    assert wrong_gold.citation_correct == 0.0
    assert canonical_page_match.citation_correct == 1.0


def test_strict_answer_parser_accepts_exact_evidence_citation() -> None:
    hit = _hit()
    payload = {
        "answer": "$532 million",
        "normalized_answer": "532",
        "result_unit": "USD million",
        "calculation": "",
        "citations": [
            {
                "document_id": hit.document_id,
                "context_uid": hit.context_uid,
                "alias_path": hit.alias_path,
                "page_number": hit.page_number,
                "chunk_id": hit.chunk_id,
            }
        ],
        "abstained": False,
        "confidence": 0.9,
    }

    result = parse_answer_response(json.dumps(payload), [hit])

    assert result.normalized_answer == "532"
    assert result.result_unit == "USD million"
    assert result.citations == [_citation()]


def test_strict_answer_parser_rejects_markdown_and_invented_citations() -> None:
    hit = _hit()
    with pytest.raises(AnswerParseError):
        parse_answer_response('```json\n{"answer": "532"}\n```', [hit])

    payload = {
        "answer": "532",
        "normalized_answer": "532",
        "result_unit": "USD million",
        "calculation": "",
        "citations": [
            {
                "document_id": "invented",
                "context_uid": hit.context_uid,
                "alias_path": hit.alias_path,
                "page_number": hit.page_number,
                "chunk_id": hit.chunk_id,
            }
        ],
        "abstained": False,
        "confidence": None,
    }
    with pytest.raises(AnswerParseError, match="supplied evidence"):
        parse_answer_response(json.dumps(payload), [hit])


def test_answerer_preserves_invalid_raw_response_for_audit() -> None:
    class FakeClient:
        def complete(self, messages: list[dict[str, str]]) -> DeepInfraResponse:
            return DeepInfraResponse(
                content='{"unexpected": true}',
                usage={"prompt_tokens": 12},
                request_id="request-1",
            )

    answerer = EvidenceAnswerer(object(), client=FakeClient())  # type: ignore[arg-type]

    with pytest.raises(AnswerParseError) as captured:
        answerer.answer("What was revenue?", [_hit()])

    assert captured.value.raw_response == '{"unexpected": true}'
    assert captured.value.usage["prompt_tokens"] == 12
    assert captured.value.usage["request_id"] == "request-1"
    assert captured.value.usage["answer_attempt_count"] == 1


def test_answer_prompt_v3_contains_operation_and_unit_rules() -> None:
    messages = build_answer_messages("What was the percentage change?", [_hit()])

    assert "percentage change" in messages[0]["content"]
    assert "new minus old" in messages[0]["content"]
    assert "Matching scale words cancel" in messages[0]["content"]
    assert '"result_unit"' in messages[0]["content"]
    assert ANSWER_PROMPT_VERSION in messages[1]["content"]


def test_answerer_corrects_one_calculation_mismatch_and_preserves_audit(
    tmp_path,
) -> None:
    hit = _hit()
    responses = [
        DeepInfraResponse(
            content=_answer_payload(
                hit,
                answer="19.41%",
                normalized_answer="19.41",
                calculation="(1703 - 1371) / 1703 * 100",
            ),
            usage={"prompt_tokens": 10, "completion_tokens": 2},
            request_id="request-1",
        ),
        DeepInfraResponse(
            content=_answer_payload(
                hit,
                answer="19.5%",
                normalized_answer="19.5",
                calculation="(1703 - 1371) / 1703 * 100",
            ),
            usage={"prompt_tokens": 12, "completion_tokens": 2},
            request_id="request-2",
        ),
    ]
    captured_messages: list[list[dict[str, str]]] = []

    class FakeClient:
        def complete(self, messages: list[dict[str, str]]) -> DeepInfraResponse:
            captured_messages.append(messages)
            return responses.pop(0)

    config = AppConfig(
        paths=PathSettings(
            dataset_root=tmp_path / "dataset",
            work_root=tmp_path / "work",
            registry_db=tmp_path / "work" / "registry.sqlite3",
            index_dir=tmp_path / "work" / "indexes",
            output_dir=tmp_path / "work" / "outputs",
        ),
        generation=GenerationSettings(correction_attempts=1),
    )
    answerer = EvidenceAnswerer(config, client=FakeClient())  # type: ignore[arg-type]

    result = answerer.answer("What was the decline?", [hit])

    assert result.answer == "19.5%"
    assert result.usage["answer_attempt_count"] == 2
    assert result.usage["prompt_tokens"] == 22
    assert result.usage["request_ids"] == ["request-1", "request-2"]
    assert result.usage["correction_history"][0]["raw_response"].startswith("{")
    assert len(captured_messages) == 2
    assert "trusted_local_calculator_result" in captured_messages[1][-1]["content"]


def test_answerer_stops_after_configured_correction_limit(tmp_path) -> None:
    hit = _hit()
    call_count = 0

    class FakeClient:
        def complete(self, messages: list[dict[str, str]]) -> DeepInfraResponse:
            nonlocal call_count
            call_count += 1
            return DeepInfraResponse(
                content=_answer_payload(
                    hit,
                    answer="10%",
                    normalized_answer="10",
                    calculation="20",
                ),
                usage={"prompt_tokens": 3},
            )

    config = AppConfig(
        paths=PathSettings(
            dataset_root=tmp_path / "dataset",
            work_root=tmp_path / "work",
            registry_db=tmp_path / "work" / "registry.sqlite3",
            index_dir=tmp_path / "work" / "indexes",
            output_dir=tmp_path / "work" / "outputs",
        ),
        generation=GenerationSettings(correction_attempts=1),
    )
    answerer = EvidenceAnswerer(config, client=FakeClient())  # type: ignore[arg-type]

    with pytest.raises(AnswerParseError, match="disagrees") as captured:
        answerer.answer("What was the percentage?", [hit])

    assert call_count == 2
    assert captured.value.usage["answer_attempt_count"] == 2


def test_answerer_corrects_per_transaction_display_unit(tmp_path) -> None:
    hit = _hit()
    responses = [
        DeepInfraResponse(
            content=_answer_payload(
                hit,
                answer="$127.4 billion",
                normalized_answer="127.4",
                result_unit="USD per transaction",
                calculation="637 / 5",
            ),
            usage={},
        ),
        DeepInfraResponse(
            content=_answer_payload(
                hit,
                answer="$127.40 per transaction",
                normalized_answer="127.4",
                result_unit="USD per transaction",
                calculation="637 / 5",
            ),
            usage={},
        ),
    ]
    correction_messages: list[list[dict[str, str]]] = []

    class FakeClient:
        def complete(self, messages: list[dict[str, str]]) -> DeepInfraResponse:
            correction_messages.append(messages)
            return responses.pop(0)

    config = AppConfig(
        paths=PathSettings(
            dataset_root=tmp_path / "dataset",
            work_root=tmp_path / "work",
            registry_db=tmp_path / "work" / "registry.sqlite3",
            index_dir=tmp_path / "work" / "indexes",
            output_dir=tmp_path / "work" / "outputs",
        ),
        generation=GenerationSettings(correction_attempts=1),
    )
    answerer = EvidenceAnswerer(config, client=FakeClient())  # type: ignore[arg-type]

    result = answerer.answer(
        "What was the average payment volume per transaction?",
        [hit],
    )

    assert result.answer == "$127.40 per transaction"
    assert result.result_unit == "USD per transaction"
    assert result.usage["answer_attempt_count"] == 2
    assert "displayed answer must state" in correction_messages[1][-1]["content"]


def test_answerer_allows_average_requested_in_billions(tmp_path) -> None:
    hit = _hit()
    call_count = 0

    class FakeClient:
        def complete(self, messages: list[dict[str, str]]) -> DeepInfraResponse:
            nonlocal call_count
            call_count += 1
            return DeepInfraResponse(
                content=_answer_payload(
                    hit,
                    answer="$2.4 billion",
                    normalized_answer="2.4",
                    result_unit="USD billion",
                    calculation="12 / 5",
                ),
                usage={},
            )

    config = AppConfig(
        paths=PathSettings(
            dataset_root=tmp_path / "dataset",
            work_root=tmp_path / "work",
            registry_db=tmp_path / "work" / "registry.sqlite3",
            index_dir=tmp_path / "work" / "indexes",
            output_dir=tmp_path / "work" / "outputs",
        ),
        generation=GenerationSettings(correction_attempts=1),
    )
    answerer = EvidenceAnswerer(config, client=FakeClient())  # type: ignore[arg-type]

    result = answerer.answer("What was average annual revenue in billions?", [hit])

    assert result.answer == "$2.4 billion"
    assert call_count == 1


def test_answerer_requires_one_total_across_multiple_years(tmp_path) -> None:
    hit = _hit()
    responses = [
        DeepInfraResponse(
            content=_answer_payload(
                hit,
                answer="$1,356 million for 2013 and $2,220 million for 2012",
                normalized_answer="1356 and 2220",
                result_unit="USD million",
                calculation="",
            )
        ),
        DeepInfraResponse(
            content=_answer_payload(
                hit,
                answer="$3,576 million",
                normalized_answer="3576",
                result_unit="USD million",
                calculation="1356 + 2220",
            )
        ),
    ]
    correction_messages: list[list[dict[str, str]]] = []

    class FakeClient:
        def complete(self, messages: list[dict[str, str]]) -> DeepInfraResponse:
            correction_messages.append(messages)
            return responses.pop(0)

    config = AppConfig(
        paths=PathSettings(
            dataset_root=tmp_path / "dataset",
            work_root=tmp_path / "work",
            registry_db=tmp_path / "work" / "registry.sqlite3",
            index_dir=tmp_path / "work" / "indexes",
            output_dir=tmp_path / "work" / "outputs",
        ),
        generation=GenerationSettings(correction_attempts=1),
    )
    answerer = EvidenceAnswerer(config, client=FakeClient())  # type: ignore[arg-type]

    result = answerer.answer(
        "What was the total balance for the years 2013 and 2012?",
        [hit],
    )

    assert result.normalized_answer == "3576"
    assert result.calculation == "1356 + 2220"
    assert result.usage["answer_attempt_count"] == 2
    assert "total across multiple periods" in correction_messages[1][-1]["content"]


def test_old_saved_answer_without_result_unit_remains_readable() -> None:
    saved = asdict(
        AnswerResult(
            answer="93.5%",
            normalized_answer="93.5",
            calculation="93.5",
            citations=[_citation()],
            abstained=False,
        )
    )
    saved.pop("result_unit")

    restored = answer_result_from_dict(saved)

    assert restored is not None
    assert restored.result_unit == ""


def test_saved_dev_answers_can_be_rescored_without_model_requests(tmp_path) -> None:
    config = AppConfig(
        paths=PathSettings(
            dataset_root=tmp_path / "dataset",
            work_root=tmp_path / "work",
            registry_db=tmp_path / "work" / "registry.sqlite3",
            index_dir=tmp_path / "work" / "indexes",
            output_dir=tmp_path / "work" / "outputs",
        )
    )
    example = BenchmarkExample(
        question_id="finqa_dev_1",
        dataset="FinQA",
        split="dev",
        question="What was the return?",
        context_uid="gold",
        document_id="doc-1",
        file_name="page.pdf",
        page_number=1,
        program_answer="0.935",
        original_answer="93.5%",
    )
    result = AnswerResult(
        answer="93.5%",
        normalized_answer="93.5",
        calculation="93.5",
        citations=[_citation()],
        abstained=False,
        usage={"calculator_result": "93.5"},
    )
    saved = {
        "question_id": example.question_id,
        "dataset": example.dataset,
        "split": example.split,
        "result": asdict(result),
    }

    summary = rescore_saved_answer_rows([saved], [example], config)

    assert summary["model_requests"] == 0
    assert summary["answer"]["correct"] == 0.0
    assert summary["semantic_answer"]["semantic_correct"] == 1.0


def test_answer_evaluation_scores_top_ten_but_sends_only_eight_contexts(
    tmp_path,
) -> None:
    config = AppConfig(
        paths=PathSettings(
            dataset_root=tmp_path / "dataset",
            work_root=tmp_path / "work",
            registry_db=tmp_path / "work" / "registry.sqlite3",
            index_dir=tmp_path / "work" / "indexes",
            output_dir=tmp_path / "work" / "outputs",
        ),
        retrieval=RetrievalSettings(answer_contexts=8),
    )
    example = BenchmarkExample(
        question_id="q1",
        dataset="FinQA",
        split="dev",
        question="What was the value?",
        context_uid="gold-context",
        document_id="gold-document",
        file_name="page.pdf",
        page_number=1,
        program_answer="1",
        original_answer="1",
    )
    hits = [
        SearchHit(
            chunk_id=index,
            context_uid=f"context-{index}",
            document_id=("gold-document" if index == 10 else f"document-{index}"),
            source_text=f"evidence {index}",
            score=1.0 / index,
            rank=index,
            retrieval_source="test",
        )
        for index in range(1, 11)
    ]
    observed_context_counts: list[int] = []
    observed_scopes: list[tuple[str, str]] = []

    def answer(question: str, evidence: list[SearchHit]) -> AnswerResult:
        observed_context_counts.append(len(evidence))
        return AnswerResult("1", "1", "", [], False)

    def retrieve(
        question: str,
        dataset: str,
        split: str,
    ) -> list[SearchHit]:
        observed_scopes.append((dataset, split))
        return hits

    run = run_evaluation(
        [example],
        answer=answer,
        config=config,
        mode="retrieved",
        retrieve=retrieve,
    )

    assert observed_context_counts == [8]
    assert observed_scopes == [("FinQA", "dev")]
    assert run.summary.retrieval["hit_at_5"] == 0.0
    assert run.summary.retrieval["hit_at_10"] == 1.0
    assert len(run.records[0].retrieved_context_uids) == 10
    assert len(run.records[0].answer_context_uids) == 8

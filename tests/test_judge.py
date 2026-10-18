"""Tests for the optional local diagnostic judge without live model calls."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from t2rag.config import AppConfig, ModelSettings, PathSettings
from t2rag.evaluation.judge import (
    JUDGE_PROMPT_VERSION,
    OllamaEvaluationJudge,
    build_judge_payload,
    parse_judge_response,
    summarize_judge_reviews,
)
from t2rag.models import AnswerResult


def _config(tmp_path: Path) -> AppConfig:
    return AppConfig(
        paths=PathSettings(
            dataset_root=tmp_path / "dataset",
            work_root=tmp_path / "work",
            registry_db=tmp_path / "work" / "registry.sqlite3",
            index_dir=tmp_path / "work" / "indexes",
            output_dir=tmp_path / "work" / "outputs",
        ),
        models=ModelSettings(evaluation_judge_model="qwen3:8b"),
    )


def test_judge_payload_contains_labels_context_and_calculator() -> None:
    generated = AnswerResult(
        answer="93.5%",
        normalized_answer="93.5",
        calculation="93.5 / 100",
        citations=[],
        abstained=False,
        result_unit="percent",
        usage={"calculator_result": "0.935"},
    )

    payload = build_judge_payload(
        question="What was the return?",
        source_context="The ending index was 193.5 from a base of 100.",
        program_answer="0.935",
        original_answer="93.5%",
        generated=generated,
    )

    assert payload["program_answer"] == "0.935"
    assert payload["original_answer"] == "93.5%"
    assert payload["generated_answer"]["result_unit"] == "percent"
    assert payload["generated_answer"]["calculator_result"] == "0.935"


def test_ollama_judge_is_deterministic_and_diagnostic_only(tmp_path: Path) -> None:
    captured: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "message": {
                    "content": json.dumps(
                        {
                            "verdict": "correct",
                            "equivalent": True,
                            "supported_answer": "93.5%",
                            "reason": "0.935 equals 93.5 percent.",
                            "confidence": 0.98,
                        }
                    )
                }
            },
        )

    client = httpx.Client(
        base_url="http://127.0.0.1:11434",
        transport=httpx.MockTransport(handler),
    )
    with OllamaEvaluationJudge(_config(tmp_path), client=client) as judge:
        decision = judge.judge({"example": "data"})

    assert decision.verdict == "correct"
    assert decision.prompt_version == JUDGE_PROMPT_VERSION
    assert captured[0]["model"] == "qwen3:8b"
    assert captured[0]["think"] is False
    assert captured[0]["format"] == "json"
    assert captured[0]["options"] == {"temperature": 0, "seed": 42}
    client.close()


def test_judge_parser_rejects_invalid_or_ambiguous_schema() -> None:
    with pytest.raises(ValueError, match="exactly"):
        parse_judge_response('{"verdict":"correct"}', model="qwen3:8b")

    invalid_ambiguous = json.dumps(
        {
            "verdict": "ambiguous",
            "equivalent": True,
            "supported_answer": "",
            "reason": "The source is incomplete.",
            "confidence": 0.4,
        }
    )
    with pytest.raises(ValueError, match="equivalent=null"):
        parse_judge_response(invalid_ambiguous, model="qwen3:8b")

    contradictory_incorrect = json.dumps(
        {
            "verdict": "incorrect",
            "equivalent": True,
            "supported_answer": "19.495%",
            "reason": "The generated value does not match the calculation.",
            "confidence": 0.9,
        }
    )
    with pytest.raises(ValueError, match="equivalent=false"):
        parse_judge_response(contradictory_incorrect, model="qwen3:8b")


def test_judge_summary_separates_model_verdicts_from_deterministic_conflicts() -> None:
    reviews = [
        {
            "question_id": "finqa_dev_1",
            "semantic_answer": {"label_conflict": 0.0},
            "judge": {"verdict": "correct"},
        },
        {
            "question_id": "finqa_dev_3",
            "semantic_answer": {"label_conflict": 1.0},
            "judge": {"verdict": "correct"},
        },
    ]

    summary = summarize_judge_reviews(reviews)

    assert summary["judge_verdict_counts"] == {
        "correct": 2,
        "incorrect": 0,
        "ambiguous": 0,
        "label_conflict": 0,
    }
    assert summary["deterministic_label_conflicts"] == {
        "count": 1,
        "question_ids": ["finqa_dev_3"],
    }

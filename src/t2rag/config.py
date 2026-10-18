"""Load validated configuration without placing secrets in source files."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from .errors import ConfigurationError


@dataclass(slots=True)
class PathSettings:
    """Locations for read-only input and generated project data."""

    dataset_root: Path
    work_root: Path
    registry_db: Path
    index_dir: Path
    output_dir: Path


@dataclass(slots=True)
class ModelSettings:
    """Local and cloud model identifiers."""

    ollama_base_url: str = "http://127.0.0.1:11434"
    text_description_model: str = "qwen3:8b"
    vision_model: str = "qwen3-vl:8b-instruct"
    embedding_provider: str = "ollama"
    embedding_model: str = "qwen3-embedding:4b"
    deepinfra_base_url: str = "https://api.deepinfra.com/v1/openai"
    deepinfra_inference_url: str = "https://api.deepinfra.com/v1/inference"
    answer_model: str = "Qwen/Qwen3-235B-A22B-Instruct-2507"
    reranker_model: str = "Qwen/Qwen3-Reranker-4B"
    evaluation_judge_model: str = "qwen3:8b"


@dataclass(slots=True)
class RetrievalSettings:
    """Chunking and ranking controls tuned only on the development split."""

    max_chunk_words: int = 500
    overlap_words: int = 80
    lexical_candidates: int = 30
    dense_candidates: int = 30
    fused_candidates: int = 30
    rerank_candidates: int = 30
    answer_contexts: int = 8
    rrf_k: int = 60
    include_descriptions: bool = False
    typo_minimum_length: int = 4
    typo_similarity_threshold: float = 85.0
    typo_ambiguity_margin: float = 5.0
    typo_max_edits: int = 2


@dataclass(slots=True)
class RuntimeSettings:
    """Network behavior and reproducibility controls."""

    request_timeout_seconds: float = 120.0
    max_retries: int = 3
    embedding_batch_size: int = 16
    random_seed: int = 42
    ollama_keep_alive: str = "2m"


@dataclass(slots=True)
class GenerationSettings:
    """Answer-generation checks and bounded correction behavior."""

    correction_attempts: int = 1
    calculator_absolute_tolerance: float = 0.01
    calculator_relative_tolerance: float = 1e-4


@dataclass(slots=True)
class EvaluationSettings:
    """Safety settings for benchmark evaluation."""

    allow_test_questions: bool = False
    allow_test_labels: bool = False
    require_frozen_manifest_for_test: bool = True
    numeric_absolute_tolerance: float = 1e-4
    numeric_relative_tolerance: float = 1e-4
    semantic_absolute_tolerance: float = 1e-4
    semantic_relative_tolerance: float = 1e-3
    retrieval_top_k: int = 10


@dataclass(slots=True)
class RawPdfSettings:
    """Controls for the optional raw-PDF ingestion comparison."""

    minimum_text_characters: int = 200
    render_dpi: int = 150
    extract_tables: bool = True


@dataclass(slots=True)
class AppConfig:
    """Complete application configuration."""

    paths: PathSettings
    models: ModelSettings = field(default_factory=ModelSettings)
    retrieval: RetrievalSettings = field(default_factory=RetrievalSettings)
    runtime: RuntimeSettings = field(default_factory=RuntimeSettings)
    generation: GenerationSettings = field(default_factory=GenerationSettings)
    evaluation: EvaluationSettings = field(default_factory=EvaluationSettings)
    raw_pdf: RawPdfSettings = field(default_factory=RawPdfSettings)

    @property
    def deepinfra_api_key(self) -> str:
        """Read the API key only when a cloud operation needs it."""

        value = os.getenv("DEEPINFRA_API_KEY", "").strip()
        if not value:
            raise ConfigurationError(
                "DEEPINFRA_API_KEY is missing. Add it to the ignored .env file."
            )
        return value


def _resolve_path(value: str, project_root: Path) -> Path:
    """Resolve relative generated-data paths against the project folder."""

    path = Path(os.path.expandvars(value)).expanduser()
    return path if path.is_absolute() else (project_root / path).resolve()


def _section(data: dict[str, Any], name: str) -> dict[str, Any]:
    """Return one mapping section and reject invalid scalar values."""

    value = data.get(name, {})
    if not isinstance(value, dict):
        raise ConfigurationError(f"Configuration section '{name}' must be a mapping.")
    return value


def load_config(config_path: str | Path) -> AppConfig:
    """Load YAML settings and the project-local ``.env`` file.

    Loading configuration does not create folders or contact any service.
    """

    path = Path(config_path).resolve()
    if not path.is_file():
        raise ConfigurationError(f"Configuration file does not exist: {path}")

    project_root = path.parent.parent
    load_dotenv(project_root / ".env", override=False)

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ConfigurationError("The configuration root must be a mapping.")

    paths_raw = _section(raw, "paths")
    dataset_value = paths_raw.get("dataset_root")
    if not dataset_value:
        raise ConfigurationError("paths.dataset_root is required.")

    work_root = _resolve_path(paths_raw.get("work_root", "work"), project_root)
    paths = PathSettings(
        dataset_root=_resolve_path(str(dataset_value), project_root),
        work_root=work_root,
        registry_db=_resolve_path(
            paths_raw.get("registry_db", str(work_root / "registry.sqlite3")),
            project_root,
        ),
        index_dir=_resolve_path(
            paths_raw.get("index_dir", str(work_root / "indexes")), project_root
        ),
        output_dir=_resolve_path(
            paths_raw.get("output_dir", str(work_root / "outputs")), project_root
        ),
    )

    try:
        return AppConfig(
            paths=paths,
            models=ModelSettings(**_section(raw, "models")),
            retrieval=RetrievalSettings(**_section(raw, "retrieval")),
            runtime=RuntimeSettings(**_section(raw, "runtime")),
            generation=GenerationSettings(**_section(raw, "generation")),
            evaluation=EvaluationSettings(**_section(raw, "evaluation")),
            raw_pdf=RawPdfSettings(**_section(raw, "raw_pdf")),
        )
    except TypeError as exc:
        raise ConfigurationError(f"Unknown or invalid configuration field: {exc}") from exc

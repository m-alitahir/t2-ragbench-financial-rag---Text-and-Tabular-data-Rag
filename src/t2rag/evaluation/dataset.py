"""Stream benchmark questions while keeping the test split sealed by default."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from t2rag.config import AppConfig
from t2rag.errors import TestSplitSealedError

from .freeze import load_and_verify_freeze_manifest


@dataclass(frozen=True, slots=True)
class BenchmarkExample:
    """One evaluation query and only the fields explicitly requested by a caller."""

    question_id: str
    dataset: str
    split: str
    question: str
    context_uid: str
    document_id: str | None
    file_name: str
    page_number: int | None
    program_answer: str | None = None
    original_answer: str | None = None
    oracle_context: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


_DATASET_NAMES = {
    "finqa": "FinQA",
    "convfinqa": "ConvFinQA",
    "tat-dqa": "TAT-DQA",
    "tatdqa": "TAT-DQA",
}


class BenchmarkLoader:
    """Load train/dev data and require an explicit freeze for all test access.

    The repository stores questions and labels in the same JSONL row.  This
    loader cannot change the source files, so it enforces the boundary at the
    API: answer fields are never copied into returned objects unless
    ``include_labels`` is explicitly enabled.
    """

    def __init__(
        self,
        config: AppConfig,
        *,
        frozen_manifest_path: str | Path | None = None,
    ) -> None:
        self._config = config
        self._dataset_root = config.paths.dataset_root.resolve()
        self._frozen_manifest_path = (
            Path(frozen_manifest_path).resolve() if frozen_manifest_path else None
        )
        self._context_documents: dict[str, str] | None = None

    def load_train(
        self,
        dataset: str,
        *,
        include_labels: bool = True,
        include_context: bool = False,
    ) -> Iterator[BenchmarkExample]:
        """Stream official training examples."""

        return self.load(
            dataset,
            "train",
            include_labels=include_labels,
            include_context=include_context,
        )

    def load_dev(
        self,
        dataset: str,
        *,
        include_labels: bool = True,
        include_context: bool = False,
    ) -> Iterator[BenchmarkExample]:
        """Stream official development examples for tuning and comparison."""

        return self.load(
            dataset,
            "dev",
            include_labels=include_labels,
            include_context=include_context,
        )

    def load_test(
        self,
        dataset: str,
        *,
        include_labels: bool = False,
        include_context: bool = False,
    ) -> Iterator[BenchmarkExample]:
        """Stream test examples only after explicit settings and freeze checks."""

        return self.load(
            dataset,
            "test",
            include_labels=include_labels,
            include_context=include_context,
        )

    def load(
        self,
        dataset: str,
        split: str,
        *,
        include_labels: bool = False,
        include_context: bool = False,
    ) -> Iterator[BenchmarkExample]:
        """Return a streaming iterator for one dataset split.

        ConvFinQA contains one ``all`` file instead of official train/dev/test
        files.  Callers must request ``split='all'`` and must not present it as
        an official held-out evaluation.
        """

        canonical_name = _canonical_dataset_name(dataset)
        clean_split = split.strip().lower()
        self._authorize(canonical_name, clean_split, include_labels)
        path = self._metadata_path(canonical_name, clean_split)
        return self._iter_file(
            path,
            dataset=canonical_name,
            requested_split=clean_split,
            include_labels=include_labels,
            include_context=include_context,
        )

    def _authorize(self, dataset: str, split: str, include_labels: bool) -> None:
        """Check test flags before resolving or opening a test metadata file."""

        if split != "test":
            return
        settings = self._config.evaluation
        if dataset == "ConvFinQA":
            raise ValueError("ConvFinQA has no official test split in this repository.")
        if not settings.allow_test_questions:
            raise TestSplitSealedError(
                "Test questions are sealed. Set evaluation.allow_test_questions only "
                "after the system is frozen."
            )
        if include_labels and not settings.allow_test_labels:
            raise TestSplitSealedError(
                "Test labels are sealed. Enable them only for the final evaluation run."
            )
        if settings.require_frozen_manifest_for_test:
            if self._frozen_manifest_path is None:
                raise TestSplitSealedError(
                    "Test access requires the path to a frozen system manifest."
                )
            load_and_verify_freeze_manifest(self._frozen_manifest_path)

    def _metadata_path(self, dataset: str, split: str) -> Path:
        if dataset == "ConvFinQA":
            if split != "all":
                raise ValueError("ConvFinQA supports only split='all'.")
            path = self._dataset_root / "data" / "ConvFinQA" / "turn_0.jsonl"
        else:
            if split not in {"train", "dev", "test"}:
                raise ValueError("Split must be train, dev, or test.")
            path = self._dataset_root / "data" / dataset / split / "metadata.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"Benchmark metadata does not exist: {path}")
        return path

    def _iter_file(
        self,
        path: Path,
        *,
        dataset: str,
        requested_split: str,
        include_labels: bool,
        include_context: bool,
    ) -> Iterator[BenchmarkExample]:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}.") from exc
                context_uid = None
                if isinstance(row, dict) and "context_id" in row:
                    context_uid = f"{dataset.lower()}:{row['context_id']}"
                yield _example_from_row(
                    row,
                    dataset=dataset,
                    requested_split=requested_split,
                    include_labels=include_labels,
                    include_context=include_context,
                    document_id=(
                        self._document_id(context_uid) if context_uid is not None else None
                    ),
                    source=f"{path}:{line_number}",
                )

    def _document_id(self, context_uid: str) -> str | None:
        """Resolve a context to the registry's canonical physical PDF hash."""

        if self._context_documents is None:
            registry_path = self._config.paths.registry_db
            if not registry_path.is_file():
                self._context_documents = {}
            else:
                # The registry is read only during evaluation.  Loading 7,318
                # small mappings once avoids one database query per question.
                with sqlite3.connect(registry_path) as connection:
                    rows = connection.execute(
                        "SELECT context_uid, document_id FROM context_records"
                    ).fetchall()
                self._context_documents = {
                    str(row[0]): str(row[1]) for row in rows
                }
        return self._context_documents.get(context_uid)


def _canonical_dataset_name(dataset: str) -> str:
    key = dataset.strip().lower()
    try:
        return _DATASET_NAMES[key]
    except KeyError as exc:
        raise ValueError(f"Unsupported dataset: {dataset}") from exc


def _optional_page_number(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid page number: {value!r}") from exc


def _example_from_row(
    row: Any,
    *,
    dataset: str,
    requested_split: str,
    include_labels: bool,
    include_context: bool,
    document_id: str | None,
    source: str,
) -> BenchmarkExample:
    if not isinstance(row, dict):
        raise ValueError(f"Metadata row must be an object at {source}.")
    required = {"id", "context_id", "split", "question", "file_name"}
    missing = sorted(required - set(row))
    if missing:
        raise ValueError(f"Missing fields {missing} at {source}.")

    row_split = str(row["split"]).strip().lower()
    if requested_split != "all" and row_split != requested_split:
        raise ValueError(
            f"Row split {row_split!r} does not match {requested_split!r} at {source}."
        )
    metadata_names = (
        "company_name",
        "company_symbol",
        "report_year",
        "company_sector",
        "company_industry",
    )
    metadata = {name: row[name] for name in metadata_names if name in row}
    return BenchmarkExample(
        question_id=str(row["id"]),
        dataset=dataset,
        split=row_split,
        question=str(row["question"]),
        # Match ingestion.metadata.normalize_row exactly so relevance labels
        # compare directly with SearchHit.context_uid values.
        context_uid=f"{dataset.lower()}:{row['context_id']}",
        document_id=document_id,
        file_name=str(row["file_name"]).replace("\\", "/"),
        page_number=_optional_page_number(row.get("page_number")),
        program_answer=str(row["program_answer"]) if include_labels else None,
        original_answer=str(row["original_answer"]) if include_labels else None,
        oracle_context=str(row["context"]) if include_context else None,
        metadata=metadata,
    )

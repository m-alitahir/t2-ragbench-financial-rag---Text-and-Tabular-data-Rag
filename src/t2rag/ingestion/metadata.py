"""Stream and normalize benchmark metadata without retaining QA labels.

Each source JSONL row contains both indexable document evidence and sealed
evaluation fields.  Normalization uses an explicit allow-list.  A new field in
an upstream dataset therefore cannot silently enter the search index.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping

from t2rag.errors import LeakageError

# These fields may be read while streaming the source row, but must never be
# copied into registry metadata, chunks, embeddings, prompts, or descriptions.
FORBIDDEN_FIELDS = frozenset(
    {
        "question",
        "questions",
        "answer",
        "answers",
        "gold_answer",
        "gold_answers",
        "program_answer",
        "original_answer",
        "qa",
        "qas",
    }
)

# Only descriptive provenance fields are copied.  In particular, the QA row
# identifier is omitted because it has no retrieval value and encourages code
# to join indexed evidence back to labels.
SAFE_METADATA_FIELDS = (
    "company_name",
    "company_symbol",
    "report_year",
    "company_sector",
    "company_industry",
    "company_headquarters",
    "company_date_added",
    "company_cik",
    "company_founded",
)

_FIELD_NORMALIZER = re.compile(r"[^a-z0-9]+")


class MetadataFormatError(ValueError):
    """Raised when one source row cannot be normalized safely."""


@dataclass(frozen=True, slots=True)
class MetadataFile:
    """One expected JSONL file and its dataset identity."""

    path: Path
    dataset: str
    default_split: str


@dataclass(frozen=True, slots=True)
class NormalizedContext:
    """Source-only context shared by one or more benchmark questions.

    The class intentionally has no question or answer attributes.  Callers can
    pass it to chunking and indexing code without carrying sealed labels along.
    """

    context_uid: str
    context_id: str
    dataset: str
    split: str
    source_file: str
    source_text: str
    table_text: str = ""
    pre_text: str = ""
    post_text: str = ""
    page_number: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def _normalized_key(value: object) -> str:
    """Convert a mapping key to the comparison form used by leakage checks."""

    return _FIELD_NORMALIZER.sub("_", str(value).strip().lower()).strip("_")


def assert_no_forbidden_fields(value: Any, *, location: str = "payload") -> None:
    """Reject forbidden field names anywhere inside an outgoing structure.

    This check is for normalized or generated payloads.  Raw benchmark rows are
    expected to contain sealed fields and must be passed through ``normalize_row``
    before this function is used.
    """

    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = _normalized_key(key)
            if normalized in FORBIDDEN_FIELDS:
                raise LeakageError(f"Forbidden field '{key}' found in {location}.")
            assert_no_forbidden_fields(child, location=f"{location}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            assert_no_forbidden_fields(child, location=f"{location}[{index}]")


def stream_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield one JSON object at a time without loading a JSONL file into RAM."""

    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise MetadataFormatError(
                    f"Invalid JSON in {source} at line {line_number}: {exc.msg}"
                ) from exc
            if not isinstance(value, dict):
                raise MetadataFormatError(
                    f"Expected an object in {source} at line {line_number}."
                )
            yield value


def _data_root(dataset_root: Path) -> Path:
    """Accept either the cloned repository root or its ``data`` directory."""

    candidate = dataset_root / "data"
    return candidate if candidate.is_dir() else dataset_root


def iter_metadata_files(dataset_root: str | Path) -> Iterator[MetadataFile]:
    """Yield the seven benchmark metadata files in deterministic order."""

    data_root = _data_root(Path(dataset_root))
    specifications = (
        ("FinQA", "train", Path("FinQA/train/metadata.jsonl")),
        ("FinQA", "dev", Path("FinQA/dev/metadata.jsonl")),
        ("FinQA", "test", Path("FinQA/test/metadata.jsonl")),
        ("ConvFinQA", "all", Path("ConvFinQA/turn_0.jsonl")),
        ("TAT-DQA", "train", Path("TAT-DQA/train/metadata.jsonl")),
        ("TAT-DQA", "dev", Path("TAT-DQA/dev/metadata.jsonl")),
        ("TAT-DQA", "test", Path("TAT-DQA/test/metadata.jsonl")),
    )
    for dataset, split, relative_path in specifications:
        path = data_root / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"Required metadata file does not exist: {path}")
        yield MetadataFile(path=path, dataset=dataset, default_split=split)


def _required_string(row: Mapping[str, Any], field_name: str) -> str:
    """Return one required non-empty string without exposing its value in errors."""

    value = row.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise MetadataFormatError(f"Metadata field '{field_name}' must be a non-empty string.")
    return value.strip()


def _optional_string(row: Mapping[str, Any], field_name: str) -> str:
    """Normalize a nullable source text field."""

    value = row.get(field_name, "")
    if value is None:
        return ""
    if not isinstance(value, str):
        raise MetadataFormatError(f"Metadata field '{field_name}' must be a string.")
    return value.strip()


def _page_number(value: Any) -> int | None:
    """Normalize integer and numeric-string page identifiers."""

    if value in (None, ""):
        return None
    try:
        page = int(value)
    except (TypeError, ValueError) as exc:
        raise MetadataFormatError("Metadata field 'page_number' must be an integer.") from exc
    return page if page >= 0 else None


def normalize_row(
    row: Mapping[str, Any],
    *,
    dataset: str,
    default_split: str,
) -> NormalizedContext:
    """Convert a FinQA, ConvFinQA, or TAT-DQA row to source-only form.

    ``context`` is the authoritative source evidence supplied by T2-RAGBench.
    FinQA and ConvFinQA also expose its narrative and table components.  TAT-DQA
    contains the combined Markdown context only.
    """

    if dataset not in {"FinQA", "ConvFinQA", "TAT-DQA"}:
        raise MetadataFormatError(f"Unsupported dataset: {dataset}")

    context_id = _required_string(row, "context_id")
    source_text = _required_string(row, "context")
    source_file = _required_string(row, "file_name").replace("\\", "/")
    split_value = row.get("split", default_split)
    split = str(split_value).strip().lower() or default_split
    if default_split != "all" and split != default_split:
        raise MetadataFormatError(
            f"Row split '{split}' does not match metadata directory '{default_split}'."
        )

    # Use a dataset-qualified key.  TAT-DQA uses hexadecimal IDs while the two
    # FinQA variants use human-readable prefixes, but qualification makes the
    # uniqueness rule explicit and future-proof.
    context_uid = f"{dataset.lower()}:{context_id}"
    metadata = {
        key: row[key]
        for key in SAFE_METADATA_FIELDS
        if key in row and row[key] is not None and row[key] != ""
    }
    metadata["source_file"] = source_file

    normalized = NormalizedContext(
        context_uid=context_uid,
        context_id=context_id,
        dataset=dataset,
        split=split,
        source_file=source_file,
        source_text=source_text,
        table_text=_optional_string(row, "table"),
        pre_text=_optional_string(row, "pre_text"),
        post_text=_optional_string(row, "post_text"),
        page_number=_page_number(row.get("page_number")),
        metadata=metadata,
    )
    assert_no_forbidden_fields(normalized.metadata, location="normalized metadata")
    return normalized


def resolve_source_path(metadata_file: MetadataFile, source_file: str) -> Path:
    """Resolve a row's PDF path according to its dataset layout."""

    # FinQA metadata lives inside its split directory.  The other metadata files
    # also use paths relative to their own containing directory, so one rule is
    # sufficient for all current schemas.
    source = (metadata_file.path.parent / Path(source_file)).resolve()
    if source.suffix.lower() != ".pdf":
        raise MetadataFormatError(f"Expected a PDF source path, got: {source_file}")
    return source

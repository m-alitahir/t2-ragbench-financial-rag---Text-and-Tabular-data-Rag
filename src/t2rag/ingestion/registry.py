"""Build the canonical PDF, alias, context, and chunk registry."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from pypdf import PdfReader

from t2rag.config import AppConfig

from .chunking import build_context_chunks
from .database import RegistryDatabase
from .metadata import (
    MetadataFile,
    MetadataFormatError,
    iter_metadata_files,
    normalize_row,
    resolve_source_path,
    stream_jsonl,
)


@dataclass(slots=True)
class RegistryStats:
    """Counters returned after an idempotent registry build."""

    metadata_rows: int = 0
    unique_contexts: int = 0
    physical_documents: int = 0
    pdf_aliases: int = 0
    raw_json_aliases: int = 0
    png_aliases: int = 0
    chunks: int = 0
    tat_blocks: int = 0
    tat_distractors: int = 0


def sha256_file(path: str | Path, *, block_size: int = 1024 * 1024) -> str:
    """Return a streaming SHA-256 digest without loading a large PDF into RAM."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


class RegistryBuilder:
    """Ingest benchmark metadata and source assets into a canonical registry."""

    def __init__(
        self,
        *,
        dataset_root: str | Path,
        database: RegistryDatabase,
        max_chunk_words: int = 500,
        overlap_words: int = 80,
    ) -> None:
        self.dataset_root = Path(dataset_root).resolve()
        candidate = self.dataset_root / "data"
        self.data_root = candidate if candidate.is_dir() else self.dataset_root
        self.database = database
        self.max_chunk_words = max_chunk_words
        self.overlap_words = overlap_words
        self.stats = RegistryStats()
        # Multiple QA rows reference the same file.  Caching prevents repeated
        # hashing and PDF parsing while retaining streaming metadata behavior.
        self._pdf_cache: dict[Path, tuple[str, int]] = {}
        self._known_documents: set[str] = set()
        self._known_alias_paths: set[str] = set()

    def _relative_path(self, path: Path) -> str:
        """Return a stable POSIX path rooted at the configured dataset folder."""

        resolved = path.resolve()
        try:
            return resolved.relative_to(self.dataset_root).as_posix()
        except ValueError as exc:
            raise ValueError(f"Source asset is outside dataset_root: {resolved}") from exc

    @staticmethod
    def _pdf_page_count(path: Path) -> int:
        """Read the real PDF page tree instead of trusting filename conventions."""

        try:
            count = len(PdfReader(str(path), strict=False).pages)
        except Exception as exc:
            raise MetadataFormatError(f"Cannot read PDF structure: {path}") from exc
        if count <= 0:
            raise MetadataFormatError(f"PDF has no pages: {path}")
        return count

    def _ensure_pdf_alias(
        self,
        path: Path,
        *,
        dataset: str,
        split: str,
        page_number: int | None,
        is_distractor: bool,
    ) -> tuple[str, int]:
        """Canonicalize one PDF and create its dataset-specific alias."""

        resolved = path.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"Referenced PDF does not exist: {resolved}")
        if resolved in self._pdf_cache:
            return self._pdf_cache[resolved]

        document_id = sha256_file(resolved)
        if document_id not in self._known_documents:
            existing = self.database.connection.execute(
                "SELECT 1 FROM physical_documents WHERE document_id = ?", (document_id,)
            ).fetchone()
            self.database.upsert_document(
                document_id,
                resolved.stat().st_size,
                self._pdf_page_count(resolved),
            )
            if existing is None:
                self.stats.physical_documents += 1
            self._known_documents.add(document_id)

        relative_path = self._relative_path(resolved)
        existing_alias = self.database.alias_by_path(relative_path)
        alias_id = self.database.upsert_alias(
            document_id=document_id,
            dataset=dataset,
            split=split,
            relative_path=relative_path,
            alias_kind="pdf",
            page_number=page_number,
            content_sha256=document_id,
            is_distractor=is_distractor,
        )
        if existing_alias is None and relative_path not in self._known_alias_paths:
            self.stats.pdf_aliases += 1
        self._known_alias_paths.add(relative_path)
        self._pdf_cache[resolved] = (document_id, alias_id)
        return document_id, alias_id

    def ingest_metadata_file(self, metadata_file: MetadataFile) -> None:
        """Stream one JSONL file and persist each distinct source context once."""

        with self.database.transaction():
            for row in stream_jsonl(metadata_file.path):
                self.stats.metadata_rows += 1
                context = normalize_row(
                    row,
                    dataset=metadata_file.dataset,
                    default_split=metadata_file.default_split,
                )
                source_path = resolve_source_path(metadata_file, context.source_file)
                document_id, alias_id = self._ensure_pdf_alias(
                    source_path,
                    dataset=context.dataset,
                    split=context.split,
                    page_number=context.page_number,
                    is_distractor=False,
                )
                inserted = self.database.upsert_context(
                    context,
                    document_id=document_id,
                    alias_id=alias_id,
                )
                if not inserted:
                    continue

                self.stats.unique_contexts += 1
                chunks = build_context_chunks(
                    context,
                    document_id=document_id,
                    alias_id=alias_id,
                    max_words=self.max_chunk_words,
                    overlap_words=self.overlap_words,
                )
                for chunk in chunks:
                    position = int(chunk.metadata["position"])
                    self.database.insert_chunk(chunk, position=position)
                    self.stats.chunks += 1

    def _register_asset_alias(
        self,
        *,
        path: Path,
        document_id: str,
        dataset: str,
        split: str,
        alias_kind: str,
        page_number: int | None,
        is_distractor: bool,
    ) -> int:
        """Register a JSON or PNG as an alias of its canonical source PDF."""

        relative_path = self._relative_path(path)
        existing = self.database.alias_by_path(relative_path)
        alias_id = self.database.upsert_alias(
            document_id=document_id,
            dataset=dataset,
            split=split,
            relative_path=relative_path,
            alias_kind=alias_kind,
            page_number=page_number,
            content_sha256=sha256_file(path),
            is_distractor=is_distractor,
        )
        if existing is None:
            if alias_kind == "raw_json":
                self.stats.raw_json_aliases += 1
            elif alias_kind == "png":
                self.stats.png_aliases += 1
        return alias_id

    def _ingest_tat_layout(
        self,
        *,
        json_path: Path,
        pdf_alias_id: int,
    ) -> int:
        """Persist TAT-DQA ordered blocks and word-level coordinates."""

        try:
            raw = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise MetadataFormatError(f"Cannot read TAT-DQA layout JSON: {json_path}") from exc
        pages = raw.get("pages") if isinstance(raw, dict) else None
        if not isinstance(pages, list) or not pages:
            raise MetadataFormatError(f"TAT-DQA layout has no pages: {json_path}")

        inserted = 0
        for page_number, page in enumerate(pages, start=1):
            if not isinstance(page, dict) or not isinstance(page.get("blocks"), list):
                raise MetadataFormatError(f"Invalid TAT-DQA page layout: {json_path}")
            blocks = page["blocks"]
            ordered = sorted(
                enumerate(blocks),
                key=lambda item: (
                    item[1].get("order", item[0])
                    if isinstance(item[1], dict)
                    else item[0]
                ),
            )
            for fallback_order, block in ordered:
                if not isinstance(block, dict):
                    raise MetadataFormatError(f"Invalid TAT-DQA block: {json_path}")
                order = block.get("order", fallback_order)
                try:
                    block_order = int(order)
                except (TypeError, ValueError) as exc:
                    raise MetadataFormatError(f"Invalid TAT-DQA block order: {json_path}") from exc
                bbox = block.get("bbox", [])
                words = block.get("words", {})
                if not isinstance(bbox, list) or not isinstance(words, dict):
                    raise MetadataFormatError(f"Invalid TAT-DQA block coordinates: {json_path}")
                self.database.insert_tat_block(
                    alias_id=pdf_alias_id,
                    page_number=page_number,
                    block_order=block_order,
                    block_uuid=str(block.get("uuid")) if block.get("uuid") else None,
                    bbox=bbox,
                    text=str(block.get("text", "")),
                    words=words,
                )
                inserted += 1
        return inserted

    @staticmethod
    def _png_page_number(path: Path, stem: str) -> int | None:
        """Parse ``<document-id>_<page>.png`` without guessing other names."""

        prefix = f"{stem}_"
        suffix = path.stem[len(prefix) :] if path.stem.startswith(prefix) else ""
        return int(suffix) if suffix.isdigit() and int(suffix) > 0 else None

    def register_tat_assets(self) -> None:
        """Register every TAT-DQA PDF, layout JSON, PNG, and distractor.

        A distractor is a train PDF with no row in ``metadata.jsonl``.  It still
        has layout JSON and may be byte-identical to a labeled alias, so the flag
        belongs to the alias rather than to the physical document.
        """

        tat_root = self.data_root / "TAT-DQA"
        with self.database.transaction():
            for split in ("train", "dev", "test"):
                raw_dir = tat_root / split / "raw"
                for pdf_path in sorted(raw_dir.glob("*.pdf")):
                    relative_path = self._relative_path(pdf_path)
                    existing = self.database.alias_by_path(relative_path)
                    has_context = existing is not None and self.database.alias_has_context(
                        int(existing["alias_id"])
                    )
                    is_distractor = split == "train" and not has_context
                    if split != "train" and not has_context:
                        raise MetadataFormatError(
                            f"Unexpected unreferenced TAT-DQA {split} PDF: {pdf_path}"
                        )
                    document_id, pdf_alias_id = self._ensure_pdf_alias(
                        pdf_path,
                        dataset="TAT-DQA",
                        split=split,
                        page_number=None,
                        is_distractor=is_distractor,
                    )
                    if is_distractor:
                        self.stats.tat_distractors += 1

                    json_path = pdf_path.with_suffix(".json")
                    if not json_path.is_file():
                        raise FileNotFoundError(f"Missing TAT-DQA layout JSON: {json_path}")
                    self._register_asset_alias(
                        path=json_path,
                        document_id=document_id,
                        dataset="TAT-DQA",
                        split=split,
                        alias_kind="raw_json",
                        page_number=None,
                        is_distractor=is_distractor,
                    )
                    self.stats.tat_blocks += self._ingest_tat_layout(
                        json_path=json_path,
                        pdf_alias_id=pdf_alias_id,
                    )

                    png_paths = sorted(raw_dir.glob(f"{pdf_path.stem}_*.png"))
                    for png_path in png_paths:
                        page_number = self._png_page_number(png_path, pdf_path.stem)
                        if page_number is None:
                            continue
                        self._register_asset_alias(
                            path=png_path,
                            document_id=document_id,
                            dataset="TAT-DQA",
                            split=split,
                            alias_kind="png",
                            page_number=page_number,
                            is_distractor=is_distractor,
                        )

    def build(self) -> RegistryStats:
        """Initialize and ingest all seven metadata files and TAT-DQA assets."""

        self.database.initialize()
        for metadata_file in iter_metadata_files(self.dataset_root):
            self.ingest_metadata_file(metadata_file)
        self.register_tat_assets()
        return self.stats


def build_registry(config: AppConfig) -> RegistryStats:
    """Build the configured registry without contacting any model service."""

    with RegistryDatabase(config.paths.registry_db) as database:
        builder = RegistryBuilder(
            dataset_root=config.paths.dataset_root,
            database=database,
            max_chunk_words=config.retrieval.max_chunk_words,
            overlap_words=config.retrieval.overlap_words,
        )
        return builder.build()

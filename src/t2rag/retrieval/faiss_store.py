"""Normalized FAISS storage with strict, reproducible manifest checks."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from t2rag.models import ChunkRecord

from .embeddings import normalize_rows

try:  # Keep import errors actionable for minimal or documentation environments.
    import faiss  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover - exercised only without the required dependency.
    faiss = None  # type: ignore[assignment]


_MANIFEST_VERSION = 1


def _require_faiss() -> Any:
    if faiss is None:
        raise RuntimeError("faiss-cpu is required for dense retrieval.")
    return faiss


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _id_sha256(ids: NDArray[np.int64]) -> str:
    # Explicit little-endian encoding keeps the digest stable across platforms.
    encoded = np.asarray(ids, dtype="<i8").tobytes(order="C")
    return hashlib.sha256(encoded).hexdigest()


def fingerprint_chunks(
    chunks: Sequence[ChunkRecord],
    *,
    include_descriptions: bool = False,
) -> str:
    """Hash the exact ordered ID/text input used to create retrieval indexes."""

    records: list[dict[str, Any]] = []
    seen: set[int] = set()
    for chunk in chunks:
        if chunk.chunk_id is None:
            raise ValueError("Cannot fingerprint a chunk without chunk_id.")
        if chunk.chunk_id in seen:
            raise ValueError(f"Duplicate chunk_id in source fingerprint: {chunk.chunk_id}")
        seen.add(chunk.chunk_id)
        records.append(
            {
                "chunk_id": chunk.chunk_id,
                "context_uid": chunk.context_uid,
                "document_id": chunk.document_id,
                "dataset": chunk.dataset,
                "split": chunk.split,
                "kind": chunk.kind,
                "page_number": chunk.page_number,
                "text": chunk.text_for_index(include_descriptions),
                # Provenance changes must invalidate the dense/lexical join,
                # even when a numeric chunk ID and its text are reused.
                "metadata": chunk.metadata,
            }
        )
    records.sort(key=lambda item: item["chunk_id"])
    payload = json.dumps(
        records,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class FaissIndexManifest:
    """Fields required to reject stale or incompatible vector indexes."""

    schema_version: int
    index_type: str
    metric: str
    normalized: bool
    dimension: int
    vector_count: int
    embedding_model: str
    source_fingerprint: str
    id_sha256: str
    index_sha256: str

    @classmethod
    def from_json(cls, value: Any) -> "FaissIndexManifest":
        if not isinstance(value, dict):
            raise ValueError("FAISS manifest must be a JSON object.")
        try:
            return cls(**value)
        except TypeError as exc:
            raise ValueError(f"FAISS manifest has missing or unknown fields: {exc}") from exc


class FaissVectorStore:
    """Persist a cosine-similarity FAISS ``IndexIDMap2(IndexFlatIP)`` index."""

    def __init__(
        self,
        index_path: str | Path,
        manifest_path: str | Path | None = None,
    ) -> None:
        self.index_path = Path(index_path)
        self.manifest_path = (
            Path(manifest_path)
            if manifest_path is not None
            else self.index_path.with_suffix(self.index_path.suffix + ".manifest.json")
        )
        self._index: Any | None = None
        self._manifest: FaissIndexManifest | None = None

    @property
    def manifest(self) -> FaissIndexManifest | None:
        return self._manifest

    @property
    def is_loaded(self) -> bool:
        return self._index is not None

    def build(
        self,
        vectors: Any,
        chunk_ids: Sequence[int],
        *,
        embedding_model: str,
        source_fingerprint: str,
    ) -> FaissIndexManifest:
        """Build and atomically replace the index and validation manifest."""

        faiss_module = _require_faiss()
        matrix = normalize_rows(vectors)
        ids = np.asarray(chunk_ids, dtype=np.int64)
        if matrix.shape[0] == 0:
            raise ValueError("A FAISS index requires at least one vector.")
        if ids.ndim != 1 or len(ids) != matrix.shape[0]:
            raise ValueError("chunk_ids must contain one ID for each embedding vector.")
        if len(np.unique(ids)) != len(ids):
            raise ValueError("FAISS chunk IDs must be unique.")
        if not embedding_model.strip():
            raise ValueError("embedding_model cannot be empty.")
        if not source_fingerprint.strip():
            raise ValueError("source_fingerprint cannot be empty.")

        index = faiss_module.IndexIDMap2(faiss_module.IndexFlatIP(matrix.shape[1]))
        index.add_with_ids(matrix, ids)

        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        index_temp_handle = tempfile.NamedTemporaryFile(
            prefix=self.index_path.name + ".",
            suffix=".tmp",
            dir=self.index_path.parent,
            delete=False,
        )
        index_temp = Path(index_temp_handle.name)
        index_temp_handle.close()
        manifest_temp = self.manifest_path.with_suffix(self.manifest_path.suffix + ".tmp")
        try:
            faiss_module.write_index(index, str(index_temp))
            manifest = FaissIndexManifest(
                schema_version=_MANIFEST_VERSION,
                index_type="IndexIDMap2(IndexFlatIP)",
                metric="cosine_via_normalized_inner_product",
                normalized=True,
                dimension=int(matrix.shape[1]),
                vector_count=int(matrix.shape[0]),
                embedding_model=embedding_model,
                source_fingerprint=source_fingerprint,
                id_sha256=_id_sha256(ids),
                index_sha256=_sha256_file(index_temp),
            )
            manifest_temp.write_text(
                json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            # Replacing the index first is deliberate.  If a crash occurs
            # between replacements, the old manifest hash rejects the new
            # index instead of allowing an inconsistent pair to be used.
            os.replace(index_temp, self.index_path)
            os.replace(manifest_temp, self.manifest_path)
        finally:
            index_temp.unlink(missing_ok=True)
            manifest_temp.unlink(missing_ok=True)

        self._index = index
        self._manifest = manifest
        return manifest

    def load(
        self,
        *,
        expected_embedding_model: str | None = None,
        expected_source_fingerprint: str | None = None,
        expected_dimension: int | None = None,
    ) -> FaissIndexManifest:
        """Load the index only after every available compatibility check passes."""

        faiss_module = _require_faiss()
        if not self.index_path.is_file() or not self.manifest_path.is_file():
            raise FileNotFoundError(
                f"FAISS index or manifest is missing: {self.index_path}, {self.manifest_path}"
            )
        manifest = FaissIndexManifest.from_json(
            json.loads(self.manifest_path.read_text(encoding="utf-8"))
        )
        if manifest.schema_version != _MANIFEST_VERSION:
            raise ValueError(
                f"Unsupported FAISS manifest version: {manifest.schema_version}."
            )
        if manifest.index_type != "IndexIDMap2(IndexFlatIP)":
            raise ValueError(f"Unexpected FAISS index type: {manifest.index_type}.")
        if manifest.metric != "cosine_via_normalized_inner_product" or not manifest.normalized:
            raise ValueError("FAISS manifest does not describe normalized cosine retrieval.")
        if expected_embedding_model and manifest.embedding_model != expected_embedding_model:
            raise ValueError(
                "Embedding model mismatch: index uses "
                f"{manifest.embedding_model!r}, expected {expected_embedding_model!r}."
            )
        if (
            expected_source_fingerprint
            and manifest.source_fingerprint != expected_source_fingerprint
        ):
            raise ValueError("FAISS source fingerprint does not match the current chunks.")
        if expected_dimension is not None and manifest.dimension != expected_dimension:
            raise ValueError(
                f"Embedding dimension mismatch: {manifest.dimension} != {expected_dimension}."
            )
        if _sha256_file(self.index_path) != manifest.index_sha256:
            raise ValueError("FAISS index file hash does not match its manifest.")

        index = faiss_module.read_index(str(self.index_path))
        if not isinstance(index, faiss_module.IndexIDMap2):
            raise ValueError("FAISS file is not an IndexIDMap2 index.")
        if int(index.d) != manifest.dimension or int(index.ntotal) != manifest.vector_count:
            raise ValueError("FAISS index dimensions or vector count differ from the manifest.")
        if int(index.metric_type) != int(faiss_module.METRIC_INNER_PRODUCT):
            raise ValueError("FAISS index does not use inner-product search.")
        stored_ids = faiss_module.vector_to_array(index.id_map).astype(np.int64, copy=False)
        if _id_sha256(stored_ids) != manifest.id_sha256:
            raise ValueError("FAISS stored IDs do not match the manifest.")

        self._index = index
        self._manifest = manifest
        return manifest

    def search(self, query_vector: Any, *, top_k: int) -> list[tuple[int, float]]:
        """Return ``(chunk_id, cosine_score)`` pairs in descending score order."""

        if top_k <= 0:
            return []
        if self._index is None or self._manifest is None:
            raise RuntimeError("Load or build the FAISS index before searching it.")
        query = normalize_rows(query_vector)
        if query.shape[0] != 1:
            raise ValueError("Dense search accepts exactly one query vector.")
        if query.shape[1] != self._manifest.dimension:
            raise ValueError(
                f"Query dimension {query.shape[1]} does not match index dimension "
                f"{self._manifest.dimension}."
            )
        scores, ids = self._index.search(query, min(top_k, self._manifest.vector_count))
        return [
            (int(chunk_id), float(score))
            for chunk_id, score in zip(ids[0], scores[0], strict=True)
            if int(chunk_id) >= 0
        ]

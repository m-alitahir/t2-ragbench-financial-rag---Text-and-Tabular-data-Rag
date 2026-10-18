"""Pluggable local embedding backends with normalized vector output."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

import httpx
import numpy as np
from numpy.typing import NDArray

from t2rag.config import AppConfig
from t2rag.errors import ExternalServiceError


FloatMatrix = NDArray[np.float32]


def normalize_rows(vectors: Any) -> FloatMatrix:
    """Return a finite, two-dimensional, unit-normalized float32 matrix."""

    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"Embeddings must form a 2-D matrix; received shape {matrix.shape}.")
    if matrix.shape[0] and matrix.shape[1] == 0:
        raise ValueError("Embedding vectors cannot have zero dimensions.")
    if not np.isfinite(matrix).all():
        raise ValueError("Embeddings contain NaN or infinite values.")
    if matrix.shape[0] == 0:
        return np.ascontiguousarray(matrix)

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("Embedding vectors must have non-zero length.")
    return np.ascontiguousarray(matrix / norms, dtype=np.float32)


@runtime_checkable
class EmbeddingBackend(Protocol):
    """Small interface shared by Ollama and Sentence Transformers."""

    @property
    def model_id(self) -> str:
        """Return the exact model identifier recorded in the index manifest."""

    def embed_documents(self, texts: Sequence[str]) -> FloatMatrix:
        """Embed document chunks in input order."""

    def embed_query(self, text: str) -> FloatMatrix:
        """Embed one query as a one-row matrix."""

    def close(self) -> None:
        """Release network or model resources."""


class OllamaEmbeddingBackend:
    """Call the local Ollama ``/api/embed`` endpoint in bounded batches.

    Ollama may return normalized vectors for some models, but this class
    normalizes every response.  That makes inner-product FAISS search equal to
    cosine similarity and prevents model-specific behavior from changing the
    metric.
    """

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        timeout_seconds: float = 120.0,
        batch_size: int = 16,
        max_retries: int = 3,
        keep_alive: str = "2m",
        client: httpx.Client | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if max_retries < 0:
            raise ValueError("max_retries cannot be negative.")
        self._model = model
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.keep_alive = keep_alive
        root = base_url.rstrip("/")
        self.endpoint = f"{root}/embed" if root.endswith("/api") else f"{root}/api/embed"
        self._owns_client = client is None
        self._client = client or httpx.Client(timeout=timeout_seconds)

    @property
    def model_id(self) -> str:
        return self._model

    def _post_batch(self, texts: Sequence[str]) -> list[list[float]]:
        """Post one batch and retry only transient failures."""

        payload = {
            "model": self._model,
            "input": list(texts),
            "truncate": False,
            # A short keep-alive avoids reloading the model for every batch.
            # The manual runbook shows how to unload it between stages.
            "keep_alive": self.keep_alive,
        }
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self._client.post(self.endpoint, json=payload)
                if response.status_code == 429 or response.status_code >= 500:
                    response.raise_for_status()
                if response.is_error:
                    detail = ""
                    try:
                        error_body = response.json()
                        if isinstance(error_body, dict):
                            detail = str(error_body.get("error", ""))
                    except ValueError:
                        detail = ""
                    detail = " ".join(detail.split())[:300]
                    input_lengths = [len(text) for text in texts]
                    raise ExternalServiceError(
                        "Ollama embedding request failed with HTTP "
                        f"{response.status_code}; batch size {len(texts)}; "
                        f"longest input {max(input_lengths, default=0)} characters. "
                        f"Provider detail: {detail or 'not supplied'}."
                    )
                data = response.json()
                vectors = data.get("embeddings") if isinstance(data, dict) else None
                if not isinstance(vectors, list) or len(vectors) != len(texts):
                    raise ExternalServiceError(
                        "Ollama returned an invalid number of embedding vectors."
                    )
                return vectors
            except ExternalServiceError:
                raise
            except (httpx.HTTPError, ValueError) as exc:
                last_error = exc
                if attempt == self.max_retries:
                    break
        raise ExternalServiceError(
            f"Ollama embedding request failed after {self.max_retries + 1} attempts."
        ) from last_error

    def embed_documents(self, texts: Sequence[str]) -> FloatMatrix:
        """Embed non-empty documents while preserving their input order."""

        values = list(texts)
        if any(not isinstance(text, str) or not text.strip() for text in values):
            raise ValueError("Every embedding input must be a non-empty string.")
        if not values:
            return np.empty((0, 0), dtype=np.float32)

        batches: list[FloatMatrix] = []
        expected_dimension: int | None = None
        for start in range(0, len(values), self.batch_size):
            matrix = normalize_rows(self._post_batch(values[start : start + self.batch_size]))
            if expected_dimension is None:
                expected_dimension = matrix.shape[1]
            elif matrix.shape[1] != expected_dimension:
                raise ExternalServiceError(
                    "Ollama changed embedding dimensions between batches."
                )
            batches.append(matrix)
        return np.ascontiguousarray(np.vstack(batches), dtype=np.float32)

    def embed_query(self, text: str) -> FloatMatrix:
        return self.embed_documents([text])

    def close(self) -> None:
        if self._owns_client:
            self._client.close()


class SentenceTransformerEmbeddingBackend:
    """Optional in-process embedding backend.

    The dependency is imported only when this backend is selected.  The base
    installation can therefore use Ollama without downloading PyTorch or a
    Sentence Transformers model.
    """

    def __init__(
        self,
        *,
        model: str,
        batch_size: int = 16,
        device: str | None = None,
        model_instance: Any | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        self._model_id = model
        self.batch_size = batch_size
        if model_instance is not None:
            self._model = model_instance
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Install the 'sentence-transformers' project extra to use this backend."
            ) from exc
        self._model = SentenceTransformer(model, device=device)

    @property
    def model_id(self) -> str:
        return self._model_id

    def embed_documents(self, texts: Sequence[str]) -> FloatMatrix:
        values = list(texts)
        if any(not isinstance(text, str) or not text.strip() for text in values):
            raise ValueError("Every embedding input must be a non-empty string.")
        if not values:
            return np.empty((0, 0), dtype=np.float32)
        vectors = self._model.encode(
            values,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return normalize_rows(vectors)

    def embed_query(self, text: str) -> FloatMatrix:
        return self.embed_documents([text])

    def close(self) -> None:
        # SentenceTransformer has no required close operation.
        return None


def create_embedding_backend(config: AppConfig) -> EmbeddingBackend:
    """Create the configured local embedding backend without doing inference."""

    provider = config.models.embedding_provider.strip().casefold().replace("-", "_")
    if provider == "ollama":
        return OllamaEmbeddingBackend(
            base_url=config.models.ollama_base_url,
            model=config.models.embedding_model,
            timeout_seconds=config.runtime.request_timeout_seconds,
            batch_size=config.runtime.embedding_batch_size,
            max_retries=config.runtime.max_retries,
            keep_alive=config.runtime.ollama_keep_alive,
        )
    if provider in {"sentence_transformer", "sentence_transformers"}:
        return SentenceTransformerEmbeddingBackend(
            model=config.models.embedding_model,
            batch_size=config.runtime.embedding_batch_size,
        )
    raise ValueError(
        "models.embedding_provider must be 'ollama' or 'sentence_transformers'; "
        f"received {config.models.embedding_provider!r}."
    )

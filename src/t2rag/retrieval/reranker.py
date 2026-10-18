"""DeepInfra adapter for the Qwen3 pairwise reranking endpoint."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Any
from urllib.parse import quote

import httpx

from t2rag.errors import ExternalServiceError
from t2rag.models import SearchHit


class DeepInfraReranker:
    """Rerank text candidates with ``Qwen/Qwen3-Reranker-4B`` by default.

    DeepInfra's reranker is not served through its OpenAI-compatible URL.  It
    uses ``POST /v1/inference/{model}`` and accepts aligned ``queries`` and
    ``documents`` arrays.  This adapter repeats one query for every document,
    validates the returned score count, and restores a stable global order
    across request batches.
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.deepinfra.com/v1/inference",
        model: str = "Qwen/Qwen3-Reranker-4B",
        timeout_seconds: float = 120.0,
        max_retries: int = 3,
        batch_size: int = 16,
        client: httpx.Client | None = None,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if not api_key.strip():
            raise ValueError("A non-empty DeepInfra API key is required.")
        if max_retries < 0:
            raise ValueError("max_retries cannot be negative.")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        self.model = model
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.endpoint = f"{base_url.rstrip('/')}/{quote(model, safe='/')}"
        self._sleeper = sleeper
        self._owns_client = client is None
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self._client = client or httpx.Client(
            timeout=timeout_seconds,
            headers=self._headers,
        )

    def _post_scores(self, query: str, documents: Sequence[str]) -> list[float]:
        payload = {
            "queries": [query] * len(documents),
            "documents": list(documents),
        }
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                # Pass headers explicitly so injected test or application
                # clients receive the same authenticated request.
                response = self._client.post(self.endpoint, json=payload, headers=self._headers)
                if response.status_code == 429 or response.status_code >= 500:
                    response.raise_for_status()
                if response.is_error:
                    # Do not include the response body.  Provider error bodies
                    # can echo input documents, which would pollute logs.
                    raise ExternalServiceError(
                        f"DeepInfra reranker returned HTTP {response.status_code}."
                    )
                data: Any = response.json()
                scores = data.get("scores") if isinstance(data, dict) else None
                if not isinstance(scores, list) or len(scores) != len(documents):
                    raise ExternalServiceError(
                        "DeepInfra returned an invalid number of reranker scores."
                    )
                converted = [float(score) for score in scores]
                if not all(math.isfinite(score) for score in converted):
                    raise ExternalServiceError("DeepInfra returned a non-finite reranker score.")
                return converted
            except ExternalServiceError:
                raise
            except (httpx.HTTPError, TypeError, ValueError) as exc:
                last_error = exc
                if attempt == self.max_retries:
                    break
                # Short bounded backoff protects the service during transient
                # rate limits.  Tests inject a no-op sleeper.
                self._sleeper(min(2.0**attempt, 8.0))
        raise ExternalServiceError(
            f"DeepInfra reranking failed after {self.max_retries + 1} attempts."
        ) from last_error

    def rerank(
        self,
        query: str,
        hits: Sequence[SearchHit],
        *,
        top_k: int | None = None,
    ) -> list[SearchHit]:
        """Return candidates sorted by descending Qwen3 reranker score."""

        if not query.strip():
            raise ValueError("Reranking query cannot be empty.")
        candidates = list(hits)
        if not candidates or (top_k is not None and top_k <= 0):
            return []

        scores: list[float] = []
        for start in range(0, len(candidates), self.batch_size):
            batch = candidates[start : start + self.batch_size]
            scores.extend(self._post_scores(query, [hit.source_text for hit in batch]))

        indexed = list(zip(candidates, scores, strict=True))
        indexed.sort(key=lambda item: (-item[1], item[0].rank, item[0].chunk_id))
        if top_k is not None:
            indexed = indexed[:top_k]
        return [
            replace(
                hit,
                score=score,
                rank=rank,
                retrieval_source="qwen3_reranker",
                component_scores={
                    **hit.component_scores,
                    "qwen3_reranker": score,
                },
            )
            for rank, (hit, score) in enumerate(indexed, start=1)
        ]

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

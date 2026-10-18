"""Minimal OpenAI-compatible client for DeepInfra answer generation.

The module has no import-time network behavior.  A request occurs only when
``complete`` is called, which keeps evaluation and offline tests predictable.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from t2rag.config import AppConfig
from t2rag.errors import ExternalServiceError


@dataclass(frozen=True, slots=True)
class DeepInfraResponse:
    """The small part of an OpenAI chat response needed by the answerer."""

    content: str
    usage: dict[str, Any] = field(default_factory=dict)
    request_id: str = ""


class DeepInfraClient:
    """Call a DeepInfra model through its OpenAI-compatible chat endpoint."""

    def __init__(self, config: AppConfig) -> None:
        self._base_url = config.models.deepinfra_base_url.rstrip("/")
        self._model = config.models.answer_model
        self._api_key = config.deepinfra_api_key
        self._timeout = config.runtime.request_timeout_seconds
        self._max_retries = config.runtime.max_retries

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> DeepInfraResponse:
        """Return one JSON-mode answer, retrying only transient failures."""

        if not messages:
            raise ValueError("At least one chat message is required.")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive.")

        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            # JSON mode reduces formatting failures.  The answerer still uses
            # strict local parsing because a service response is untrusted.
            "response_format": {"type": "json_object"},
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        last_error: Exception | None = None
        attempts = max(1, self._max_retries + 1)
        for attempt in range(attempts):
            try:
                with httpx.Client(timeout=self._timeout) as client:
                    response = client.post(
                        f"{self._base_url}/chat/completions",
                        headers=headers,
                        json=payload,
                    )
                if response.status_code == 429 or response.status_code >= 500:
                    raise _TransientResponseError(response.status_code)
                if response.is_error:
                    raise ExternalServiceError(
                        f"DeepInfra returned HTTP {response.status_code}."
                    )
                return _parse_response(response)
            except (httpx.TimeoutException, httpx.NetworkError, _TransientResponseError) as exc:
                last_error = exc
                if attempt + 1 >= attempts:
                    break
                # The short capped backoff helps with rate limits without
                # making a failed benchmark run wait indefinitely.
                time.sleep(min(2**attempt, 8))

        raise ExternalServiceError(
            f"DeepInfra did not return a usable response after {attempts} attempt(s)."
        ) from last_error


class _TransientResponseError(Exception):
    """Internal marker for HTTP responses that are safe to retry."""

    def __init__(self, status_code: int) -> None:
        super().__init__(f"Transient HTTP status {status_code}")
        self.status_code = status_code


def _parse_response(response: httpx.Response) -> DeepInfraResponse:
    """Validate the external response before returning model text."""

    try:
        data = response.json()
        choices = data["choices"]
        content = choices[0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise ExternalServiceError("DeepInfra returned an invalid chat response.") from exc

    if not isinstance(content, str) or not content.strip():
        raise ExternalServiceError("DeepInfra returned empty answer content.")
    usage = data.get("usage", {})
    if not isinstance(usage, dict):
        usage = {}
    return DeepInfraResponse(
        content=content,
        usage=usage,
        request_id=response.headers.get("x-request-id", ""),
    )

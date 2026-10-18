"""Optional Ollama descriptions kept separate from source evidence.

Descriptions can add retrieval vocabulary, but they are model output rather
than admissible evidence.  This module stores them in ``document_descriptions``
and never mutates ``chunks.source_text`` or the default FTS index.
"""

from __future__ import annotations

import base64
import hashlib
from pathlib import Path
from typing import Any

import httpx

from t2rag.config import AppConfig
from t2rag.errors import ExternalServiceError

from .database import RegistryDatabase

TEXT_PROMPT_VERSION = "financial-source-description-v1"
VISION_PROMPT_VERSION = "financial-page-description-v1"

TEXT_SYSTEM_PROMPT = """You describe financial evidence for document retrieval.
Use only the supplied source text. Mention the company, period, accounting topic,
table measures, and units when present. Do not calculate an answer. Do not invent
facts. Return one concise paragraph without Markdown headings."""

VISION_PROMPT = """Describe this financial-report page for retrieval. Identify the
page layout, visible section titles, table subjects, chart subjects, periods, units,
and distinctive financial terms. Do not infer values that are not legible. Do not
answer any benchmark question. Return one concise paragraph."""


class OllamaDescriptionService:
    """Generate optional text and page-image descriptions through local Ollama."""

    def __init__(
        self,
        config: AppConfig,
        *,
        client: httpx.Client | None = None,
    ) -> None:
        self.config = config
        self._owns_client = client is None
        self.client = client or httpx.Client(
            base_url=config.models.ollama_base_url.rstrip("/"),
            timeout=config.runtime.request_timeout_seconds,
        )

    def close(self) -> None:
        """Close a client created by this service."""

        if self._owns_client:
            self.client.close()

    def __enter__(self) -> OllamaDescriptionService:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _generate(self, payload: dict[str, Any]) -> str:
        """Call Ollama's non-streaming generate endpoint with bounded retries."""

        last_error: Exception | None = None
        for _ in range(self.config.runtime.max_retries + 1):
            try:
                response = self.client.post("/api/generate", json=payload)
                response.raise_for_status()
                body = response.json()
                text = body.get("response", "") if isinstance(body, dict) else ""
                if not isinstance(text, str) or not text.strip():
                    raise ExternalServiceError("Ollama returned an empty description.")
                return text.strip()
            except (httpx.HTTPError, ValueError, ExternalServiceError) as exc:
                last_error = exc
        raise ExternalServiceError(f"Ollama description request failed: {last_error}")

    def describe_text(self, source_text: str) -> tuple[str, str]:
        """Describe source evidence with qwen3:8b and return text plus input hash."""

        if not source_text.strip():
            raise ValueError("Source text cannot be empty.")
        digest = hashlib.sha256(source_text.encode("utf-8")).hexdigest()
        payload = {
            "model": self.config.models.text_description_model,
            "system": TEXT_SYSTEM_PROMPT,
            "prompt": source_text,
            "stream": False,
            "think": False,
            "keep_alive": self.config.runtime.ollama_keep_alive,
            "options": {"temperature": 0},
        }
        return self._generate(payload), digest

    def describe_page_image(self, image_path: str | Path) -> tuple[str, str]:
        """Describe one local page image with qwen3-vl:8b-instruct."""

        path = Path(image_path)
        image_bytes = path.read_bytes()
        digest = hashlib.sha256(image_bytes).hexdigest()
        payload = {
            "model": self.config.models.vision_model,
            "prompt": VISION_PROMPT,
            "images": [base64.b64encode(image_bytes).decode("ascii")],
            "stream": False,
            "think": False,
            "keep_alive": self.config.runtime.ollama_keep_alive,
            "options": {"temperature": 0},
        }
        return self._generate(payload), digest

    def generate_and_store_text(
        self,
        database: RegistryDatabase,
        *,
        document_id: str,
        context_uid: str,
        alias_id: int,
        source_text: str,
    ) -> int:
        """Generate and store a context description outside the source index."""

        description, digest = self.describe_text(source_text)
        with database.transaction():
            return database.store_description(
                document_id=document_id,
                context_uid=context_uid,
                alias_id=alias_id,
                page_number=None,
                modality="text",
                model=self.config.models.text_description_model,
                prompt_version=TEXT_PROMPT_VERSION,
                input_sha256=digest,
                description_text=description,
            )

    def generate_and_store_page(
        self,
        database: RegistryDatabase,
        *,
        document_id: str,
        alias_id: int,
        page_number: int,
        image_path: str | Path,
    ) -> int:
        """Generate and store a page-image description outside the source index."""

        description, digest = self.describe_page_image(image_path)
        with database.transaction():
            return database.store_description(
                document_id=document_id,
                context_uid=None,
                alias_id=alias_id,
                page_number=page_number,
                modality="vision",
                model=self.config.models.vision_model,
                prompt_version=VISION_PROMPT_VERSION,
                input_sha256=digest,
                description_text=description,
            )

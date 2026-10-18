"""Record exact local model digests and configured cloud model identifiers."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from .config import AppConfig
from .errors import ExternalServiceError
from .json_utils import write_json_atomic


def capture_model_manifest(config: AppConfig, output_path: str | Path) -> dict[str, Any]:
    """Query local Ollama tags and save only models used by this project."""

    endpoint = f"{config.models.ollama_base_url.rstrip('/')}/api/tags"
    try:
        response = httpx.get(endpoint, timeout=config.runtime.request_timeout_seconds)
        response.raise_for_status()
        payload = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise ExternalServiceError("Could not read the local Ollama model list.") from exc
    raw_models = payload.get("models") if isinstance(payload, dict) else None
    if not isinstance(raw_models, list):
        raise ExternalServiceError("Ollama returned an invalid model list.")

    expected = {
        config.models.text_description_model,
        config.models.vision_model,
        config.models.embedding_model,
        config.models.evaluation_judge_model,
    }
    selected: dict[str, dict[str, Any]] = {}
    for item in raw_models:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or item.get("model") or "")
        if name not in expected:
            continue
        selected[name] = {
            "digest": str(item.get("digest", "")),
            "size": item.get("size"),
            "modified_at": item.get("modified_at"),
            "details": item.get("details", {}),
        }
    missing = sorted(expected - set(selected))
    if missing:
        raise ExternalServiceError(f"Required Ollama models are missing: {missing}")

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "ollama_base_url": config.models.ollama_base_url,
        "local_models": selected,
        "local_model_roles": {
            "text_description": config.models.text_description_model,
            "vision": config.models.vision_model,
            "embedding": config.models.embedding_model,
            "evaluation_judge": config.models.evaluation_judge_model,
        },
        "cloud_models": {
            "answer": config.models.answer_model,
            "reranker": config.models.reranker_model,
        },
    }
    write_json_atomic(Path(output_path).resolve(), manifest)
    return manifest

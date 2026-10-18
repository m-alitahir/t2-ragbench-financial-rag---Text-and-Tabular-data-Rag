"""Freeze system artifacts before test questions or labels are opened.

A freeze manifest is an integrity record.  It records hashes for configuration,
prompts, code, and indexes chosen on development data.  Test loading verifies
this record so accidental post-test tuning is visible and reproducible.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from t2rag.errors import TestSplitSealedError


FREEZE_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class FrozenArtifact:
    """One named file or directory locked by a freeze manifest."""

    path: str
    sha256: str


@dataclass(frozen=True, slots=True)
class FreezeManifest:
    """Validated immutable system snapshot used to authorize test access."""

    schema_version: int
    frozen: bool
    created_at_utc: str
    config_path: str
    config_sha256: str
    artifacts: dict[str, FrozenArtifact]
    notes: str
    manifest_sha256: str


def _canonical_json(data: Mapping[str, Any]) -> bytes:
    return json.dumps(
        data,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def hash_mapping(data: Mapping[str, Any]) -> str:
    """Return a deterministic SHA-256 for JSON-compatible configuration."""

    return hashlib.sha256(_canonical_json(data)).hexdigest()


def hash_file(path: str | Path) -> str:
    """Stream a file into SHA-256 so large indexes do not enter memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_path(path: str | Path) -> str:
    """Hash a file or a directory tree in stable relative-path order."""

    target = Path(path).resolve()
    if target.is_file():
        return hash_file(target)
    if not target.is_dir():
        raise FileNotFoundError(f"Freeze artifact does not exist: {target}")

    digest = hashlib.sha256()
    # Imports can create bytecode after freezing. Those generated files do not
    # change program behavior and must not invalidate an otherwise fixed run.
    files = sorted(
        item
        for item in target.rglob("*")
        if item.is_file()
        and "__pycache__" not in item.parts
        and item.suffix.casefold() not in {".pyc", ".pyo"}
    )
    for item in files:
        relative = item.relative_to(target).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(hash_file(item)))
    return digest.hexdigest()


def _write_json_atomic(path: Path, data: Mapping[str, Any]) -> None:
    """Write a manifest completely before replacing an older manifest."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def create_freeze_manifest(
    output_path: str | Path,
    *,
    config_path: str | Path,
    artifact_paths: Mapping[str, str | Path],
    notes: str = "",
) -> FreezeManifest:
    """Hash the selected system and write the manifest that seals it."""

    config_target = Path(config_path).resolve()
    if not config_target.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {config_target}")
    if not artifact_paths:
        raise ValueError("At least one artifact must be frozen.")

    artifacts: dict[str, dict[str, str]] = {}
    for name, raw_path in sorted(artifact_paths.items()):
        if not name.strip():
            raise ValueError("Freeze artifact names must not be empty.")
        target = Path(raw_path).resolve()
        artifacts[name] = {"path": str(target), "sha256": hash_path(target)}

    payload: dict[str, Any] = {
        "schema_version": FREEZE_SCHEMA_VERSION,
        "frozen": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_target),
        "config_sha256": hash_file(config_target),
        "artifacts": artifacts,
        "notes": notes,
    }
    payload["manifest_sha256"] = hash_mapping(payload)
    _write_json_atomic(Path(output_path).resolve(), payload)
    return _manifest_from_payload(payload)


def _manifest_from_payload(payload: Mapping[str, Any]) -> FreezeManifest:
    """Convert checked JSON data into a typed manifest."""

    required = {
        "schema_version",
        "frozen",
        "created_at_utc",
        "config_path",
        "config_sha256",
        "artifacts",
        "notes",
        "manifest_sha256",
    }
    if set(payload) != required:
        raise TestSplitSealedError("The freeze manifest has an invalid schema.")
    if payload["schema_version"] != FREEZE_SCHEMA_VERSION or payload["frozen"] is not True:
        raise TestSplitSealedError("The system is not recorded as frozen.")
    if not isinstance(payload["artifacts"], dict) or not payload["artifacts"]:
        raise TestSplitSealedError("The freeze manifest contains no artifacts.")

    artifacts: dict[str, FrozenArtifact] = {}
    try:
        for name, item in payload["artifacts"].items():
            if not isinstance(name, str) or not isinstance(item, dict):
                raise TypeError
            if set(item) != {"path", "sha256"}:
                raise TypeError
            artifacts[name] = FrozenArtifact(path=item["path"], sha256=item["sha256"])
        return FreezeManifest(
            schema_version=payload["schema_version"],
            frozen=payload["frozen"],
            created_at_utc=payload["created_at_utc"],
            config_path=payload["config_path"],
            config_sha256=payload["config_sha256"],
            artifacts=artifacts,
            notes=payload["notes"],
            manifest_sha256=payload["manifest_sha256"],
        )
    except (KeyError, TypeError) as exc:
        raise TestSplitSealedError("The freeze manifest contains invalid values.") from exc


def load_and_verify_freeze_manifest(
    manifest_path: str | Path,
    *,
    verify_artifacts: bool = True,
) -> FreezeManifest:
    """Verify manifest integrity and optionally re-hash every artifact."""

    path = Path(manifest_path).resolve()
    if not path.is_file():
        raise TestSplitSealedError(f"A frozen manifest is required: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TestSplitSealedError("The freeze manifest cannot be read.") from exc
    if not isinstance(payload, dict):
        raise TestSplitSealedError("The freeze manifest root must be an object.")

    stated_hash = payload.get("manifest_sha256")
    content = dict(payload)
    content.pop("manifest_sha256", None)
    if not isinstance(stated_hash, str) or hash_mapping(content) != stated_hash:
        raise TestSplitSealedError("The freeze manifest integrity check failed.")
    manifest = _manifest_from_payload(payload)

    if verify_artifacts:
        try:
            current_config_hash = hash_file(manifest.config_path)
        except FileNotFoundError as exc:
            raise TestSplitSealedError("The frozen configuration is missing.") from exc
        if current_config_hash != manifest.config_sha256:
            raise TestSplitSealedError(
                "The configuration changed after the system freeze."
            )
        for name, artifact in manifest.artifacts.items():
            try:
                current_hash = hash_path(artifact.path)
            except FileNotFoundError as exc:
                raise TestSplitSealedError(
                    f"Frozen artifact '{name}' is missing."
                ) from exc
            if current_hash != artifact.sha256:
                raise TestSplitSealedError(
                    f"Frozen artifact '{name}' changed after the freeze."
                )
    return manifest


def write_run_manifest(
    output_path: str | Path,
    *,
    freeze_manifest_path: str | Path,
    dataset: str,
    split: str,
    settings: Mapping[str, Any],
    metrics: Mapping[str, Any],
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write an auditable record for one evaluation run."""

    frozen = load_and_verify_freeze_manifest(freeze_manifest_path)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "freeze_manifest_sha256": frozen.manifest_sha256,
        "dataset": dataset,
        "split": split,
        "settings": dict(settings),
        "metrics": dict(metrics),
        "extra": dict(extra or {}),
    }
    payload["run_sha256"] = hash_mapping(payload)
    _write_json_atomic(Path(output_path).resolve(), payload)
    return payload

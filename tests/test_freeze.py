"""Tests for the immutable test-evaluation boundary."""

from pathlib import Path

import pytest

from t2rag.errors import TestSplitSealedError as SplitSealedError
from t2rag.evaluation.freeze import (
    create_freeze_manifest,
    load_and_verify_freeze_manifest,
)


def test_freeze_detects_changed_artifact(tmp_path: Path) -> None:
    config = tmp_path / "frozen.yaml"
    source = tmp_path / "source.py"
    manifest = tmp_path / "freeze.json"
    config.write_text("random_seed: 42\n", encoding="utf-8")
    source.write_text("VALUE = 1\n", encoding="utf-8")

    create_freeze_manifest(
        manifest,
        config_path=config,
        artifact_paths={"source": source},
    )
    assert load_and_verify_freeze_manifest(manifest).frozen is True

    source.write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(SplitSealedError, match="changed after the freeze"):
        load_and_verify_freeze_manifest(manifest)


def test_freeze_detects_changed_configuration(tmp_path: Path) -> None:
    config = tmp_path / "frozen.yaml"
    artifact = tmp_path / "index.bin"
    manifest = tmp_path / "freeze.json"
    config.write_text("answer_contexts: 8\n", encoding="utf-8")
    artifact.write_bytes(b"index")
    create_freeze_manifest(
        manifest,
        config_path=config,
        artifact_paths={"index": artifact},
    )

    config.write_text("answer_contexts: 10\n", encoding="utf-8")
    with pytest.raises(SplitSealedError, match="configuration changed"):
        load_and_verify_freeze_manifest(manifest)


def test_freeze_ignores_generated_python_bytecode(tmp_path: Path) -> None:
    config = tmp_path / "frozen.yaml"
    source_dir = tmp_path / "src"
    source_dir.mkdir()
    (source_dir / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    config.write_text("seed: 42\n", encoding="utf-8")
    manifest = tmp_path / "freeze.json"
    create_freeze_manifest(
        manifest,
        config_path=config,
        artifact_paths={"source": source_dir},
    )

    cache = source_dir / "__pycache__"
    cache.mkdir()
    (cache / "module.cpython-311.pyc").write_bytes(b"generated")

    assert load_and_verify_freeze_manifest(manifest).frozen is True

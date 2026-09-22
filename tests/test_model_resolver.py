"""Tests for model directory auto-resolution and friendly error reporting."""

from __future__ import annotations

from pathlib import Path

import pytest

from redsan_doc_trimmer.model_resolver import (
    get_user_cache_dirs,
    resolve_model_dir,
)


def test_resolve_model_dir_from_explicit_path(tmp_path: Path):
    """When a valid explicit directory containing model.onnx is passed, it is resolved."""
    fake_onnx = tmp_path / "model.onnx"
    fake_onnx.write_text("fake onnx content")

    resolved = resolve_model_dir(tmp_path)
    assert resolved == tmp_path.resolve()


def test_resolve_model_dir_from_edsan_trimmer_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """When EDSAN_TRIMMER_PATH points to a valid model directory, it is resolved."""
    fake_onnx = tmp_path / "model.onnx"
    fake_onnx.write_text("fake onnx content")

    monkeypatch.setenv("EDSAN_TRIMMER_PATH", str(tmp_path))
    resolved = resolve_model_dir()
    assert resolved == tmp_path.resolve()


def test_resolve_model_dir_from_redsan_trimmer_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """When REDSAN_TRIMMER_PATH points to a valid model directory, it is resolved."""
    fake_onnx = tmp_path / "model.onnx"
    fake_onnx.write_text("fake onnx content")

    monkeypatch.delenv("EDSAN_TRIMMER_PATH", raising=False)
    monkeypatch.setenv("REDSAN_TRIMMER_PATH", str(tmp_path))
    resolved = resolve_model_dir()
    assert resolved == tmp_path.resolve()


def test_resolve_model_dir_from_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """When model is installed in user cache, it is resolved."""
    cache_dir = tmp_path / "cache" / "R" / "edsan_doc_trimmer" / "v1"
    cache_dir.mkdir(parents=True)
    fake_onnx = cache_dir / "model.onnx"
    fake_onnx.write_text("fake onnx in cache")

    monkeypatch.delenv("EDSAN_TRIMMER_PATH", raising=False)
    monkeypatch.delenv("REDSAN_TRIMMER_PATH", raising=False)
    monkeypatch.setattr("redsan_doc_trimmer.model_resolver.get_user_cache_dirs", lambda: [cache_dir])

    resolved = resolve_model_dir()
    assert resolved == cache_dir.resolve()


def test_resolve_model_dir_default_repo_fallback(monkeypatch: pytest.MonkeyPatch):
    """Default repo artifacts/active_learning/onnx_export is resolved in repo checkout."""
    monkeypatch.delenv("EDSAN_TRIMMER_PATH", raising=False)
    monkeypatch.delenv("REDSAN_TRIMMER_PATH", raising=False)
    monkeypatch.setattr("redsan_doc_trimmer.model_resolver.get_user_cache_dirs", list)

    resolved = resolve_model_dir()
    assert (resolved / "model.onnx").is_file()
    assert resolved.name == "onnx_export"


def test_resolve_model_dir_not_found_raises_informative_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """When model cannot be found, a FileNotFoundError with clear instructions is raised."""
    # Isolate environment
    monkeypatch.delenv("EDSAN_TRIMMER_PATH", raising=False)
    monkeypatch.delenv("REDSAN_TRIMMER_PATH", raising=False)
    nonexistent = tmp_path / "nowhere"

    with pytest.raises(FileNotFoundError) as excinfo:
        monkeypatch.chdir(tmp_path)
        resolve_model_dir(nonexistent)

    err_msg = str(excinfo.value)
    assert "[ERROR] edsan-doc-trimmer model not found!" in err_msg
    assert "edsan-doc-trimmer-v1.2.0.zip" in err_msg
    assert "EDSAN_TRIMMER_PATH" in err_msg
    assert "edsan_install_trimmer" in err_msg
    assert str(nonexistent.resolve()) in err_msg


def test_resolve_model_dir_from_direct_model_file(tmp_path: Path):
    """When a direct path to model.onnx file is passed, it unwraps to the parent directory."""
    fake_onnx = tmp_path / "model.onnx"
    fake_onnx.write_text("fake onnx content")

    resolved = resolve_model_dir(fake_onnx)
    assert resolved == tmp_path.resolve()


def test_resolve_model_dir_empty_string_falls_back(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Passing an empty string should fall back to auto-resolution rather than treating it as explicit path."""
    fake_onnx = tmp_path / "model.onnx"
    fake_onnx.write_text("fake onnx content")

    monkeypatch.setenv("EDSAN_TRIMMER_PATH", str(tmp_path))
    resolved = resolve_model_dir("")
    assert resolved == tmp_path.resolve()


def test_resolve_model_dir_from_argv_script_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """When running a script in an extracted folder, model.onnx next to the script is resolved."""
    fake_onnx = tmp_path / "model.onnx"
    fake_onnx.write_text("fake onnx content")
    fake_script = tmp_path / "trim_batch_service.py"
    fake_script.write_text("# script")

    monkeypatch.delenv("EDSAN_TRIMMER_PATH", raising=False)
    monkeypatch.delenv("REDSAN_TRIMMER_PATH", raising=False)
    monkeypatch.setattr("redsan_doc_trimmer.model_resolver.get_user_cache_dirs", list)
    monkeypatch.setattr("sys.argv", [str(fake_script)])

    resolved = resolve_model_dir()
    assert resolved == tmp_path.resolve()


def test_get_user_cache_dirs_returns_paths():
    """Cache directories list should return at least one plausible path on all platforms."""
    cache_dirs = get_user_cache_dirs()
    assert len(cache_dirs) > 0
    assert all(isinstance(p, Path) for p in cache_dirs)


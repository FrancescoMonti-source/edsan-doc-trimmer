"""The worker ships without the package, so it carries a copy of the resolver.

These tests run the same discovery scenarios through both implementations so the
copy cannot drift from `redsan_doc_trimmer.model_resolver`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("onnxruntime")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import trim_batch_service as worker
from redsan_doc_trimmer.model_resolver import resolve_model_dir as package_resolver

from test_model_resolver import make_artifact

RESOLVERS = {
    "package": package_resolver,
    "worker_standalone": worker._standalone_resolve_model_dir,
}


@pytest.fixture(params=sorted(RESOLVERS))
def resolve(request, tmp_path, monkeypatch):
    for var in ("EDSAN_TRIMMER_PATH", "REDSAN_TRIMMER_PATH", "EDSAN_TRIMMER_VERSION", "XDG_CACHE_HOME"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("R_USER_CACHE_DIR", str(tmp_path / "usercache"))
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    return RESOLVERS[request.param]


@pytest.fixture
def cache_root(tmp_path):
    return tmp_path / "usercache" / "R" / "edsan_doc_trimmer"


def test_highest_version_wins(resolve, cache_root):
    make_artifact(cache_root / "1.9.0", "1.9.0")
    make_artifact(cache_root / "1.10.0", "1.10.0")
    make_artifact(cache_root / "v1.11", "1.11.0")  # misnamed, ignored
    assert resolve() == (cache_root / "1.10.0").resolve()


def test_pin_and_unknown_pin(resolve, cache_root, monkeypatch):
    make_artifact(cache_root / "1.2.0", "1.2.0")
    make_artifact(cache_root / "1.3.0", "1.3.0")
    monkeypatch.setenv("EDSAN_TRIMMER_VERSION", "1.2.0")
    assert resolve() == (cache_root / "1.2.0").resolve()

    monkeypatch.setenv("EDSAN_TRIMMER_VERSION", "9.9.9")
    with pytest.raises(FileNotFoundError, match=r"1\.3\.0, 1\.2\.0"):
        resolve()


def test_env_path_and_explicit_directory_precedence(resolve, cache_root, tmp_path, monkeypatch):
    make_artifact(cache_root / "1.3.0", "1.3.0")
    env_dir = tmp_path / "env"
    explicit = tmp_path / "explicit"
    for d in (env_dir, explicit):
        d.mkdir()
        (d / "model.onnx").write_text("fake")
    monkeypatch.setenv("EDSAN_TRIMMER_PATH", str(env_dir))
    assert resolve() == env_dir.resolve()
    assert resolve(explicit) == explicit.resolve()


def test_legacy_slot_used_with_warning(resolve, cache_root):
    make_artifact(cache_root / "v1", "1.3.0")
    with pytest.warns(UserWarning, match="legacy"):
        assert resolve() == (cache_root / "v1").resolve()


def test_dev_checkout_never_selected_and_not_found_is_release_agnostic(resolve, cache_root):
    make_artifact(Path.cwd() / "artifacts" / "active_learning" / "onnx_export", "1.3.0")
    with pytest.raises(FileNotFoundError) as excinfo:
        resolve()
    message = str(excinfo.value)
    assert "EDSAN_TRIMMER_PATH" in message
    assert "edsan_install_trimmer" in message
    assert "1.3.0" not in message

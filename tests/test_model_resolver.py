"""Tests for model directory auto-resolution against redsan's versioned trimmer cache."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from redsan_doc_trimmer.model_resolver import (
    REQUIRED_ARTIFACT_FILES,
    get_trimmer_cache_root,
    get_user_cache_dirs,
    list_installed_artifacts,
    resolve_model_dir,
)


def make_artifact(
    folder: Path,
    version: str | None = "1.3.0",
    worker_contract: str = "model-only-v1",
    omit: tuple[str, ...] = (),
) -> Path:
    """Write a minimal artifact folder shaped like an unpacked release archive."""
    folder.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED_ARTIFACT_FILES:
        if name in omit:
            continue
        if name == "artifact.json":
            manifest = {"artifact_name": "edsan-doc-trimmer", "worker_contract": worker_contract}
            if version is not None:
                manifest["artifact_version"] = version
            (folder / name).write_text(json.dumps(manifest), encoding="utf-8")
        else:
            (folder / name).write_text(f"fake {name}", encoding="utf-8")
    return folder


@pytest.fixture
def cache_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect the trimmer cache to a temp dir, as redsan's tests do with R_USER_CACHE_DIR."""
    for var in (
        "EDSAN_TRIMMER_PATH",
        "REDSAN_TRIMMER_PATH",
        "EDSAN_TRIMMER_VERSION",
        "XDG_CACHE_HOME",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("R_USER_CACHE_DIR", str(tmp_path / "usercache"))
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    monkeypatch.setattr(sys, "argv", ["pytest"])
    root = tmp_path / "usercache" / "R" / "edsan_doc_trimmer"
    assert get_trimmer_cache_root() == root
    return root


# --- explicit directory ------------------------------------------------------


def test_explicit_path_resolves(tmp_path: Path, cache_root: Path):
    (tmp_path / "model.onnx").write_text("fake")
    assert resolve_model_dir(tmp_path) == tmp_path.resolve()


def test_explicit_model_file_unwraps_to_parent(tmp_path: Path, cache_root: Path):
    (tmp_path / "model.onnx").write_text("fake")
    assert resolve_model_dir(tmp_path / "model.onnx") == tmp_path.resolve()


def test_explicit_path_wins_over_everything(tmp_path: Path, cache_root: Path, monkeypatch):
    explicit = tmp_path / "explicit"
    explicit.mkdir()
    (explicit / "model.onnx").write_text("fake")
    env_dir = make_artifact(tmp_path / "envdir")
    make_artifact(cache_root / "1.3.0", "1.3.0")
    monkeypatch.setenv("EDSAN_TRIMMER_PATH", str(env_dir))
    monkeypatch.setenv("EDSAN_TRIMMER_VERSION", "1.3.0")

    assert resolve_model_dir(explicit) == explicit.resolve()


def test_invalid_explicit_path_raises_without_falling_back(tmp_path: Path, cache_root: Path):
    make_artifact(cache_root / "1.3.0", "1.3.0")
    nowhere = tmp_path / "nowhere"

    with pytest.raises(FileNotFoundError) as excinfo:
        resolve_model_dir(nowhere)
    assert str(nowhere.resolve()) in str(excinfo.value)


def test_empty_string_falls_back_to_auto_resolution(cache_root: Path):
    make_artifact(cache_root / "1.3.0", "1.3.0")
    assert resolve_model_dir("") == (cache_root / "1.3.0").resolve()


# --- environment path --------------------------------------------------------


def test_edsan_trimmer_path_wins_over_cache(tmp_path: Path, cache_root: Path, monkeypatch):
    env_dir = tmp_path / "extracted"
    env_dir.mkdir()
    (env_dir / "model.onnx").write_text("fake")
    make_artifact(cache_root / "1.3.0", "1.3.0")
    monkeypatch.setenv("EDSAN_TRIMMER_PATH", str(env_dir))

    assert resolve_model_dir() == env_dir.resolve()


def test_edsan_trimmer_path_wins_over_version_pin(tmp_path: Path, cache_root: Path, monkeypatch):
    env_dir = tmp_path / "extracted"
    env_dir.mkdir()
    (env_dir / "model.onnx").write_text("fake")
    make_artifact(cache_root / "1.2.0", "1.2.0")
    monkeypatch.setenv("EDSAN_TRIMMER_PATH", str(env_dir))
    monkeypatch.setenv("EDSAN_TRIMMER_VERSION", "1.2.0")

    assert resolve_model_dir() == env_dir.resolve()


def test_legacy_redsan_trimmer_path_is_honoured(tmp_path: Path, cache_root: Path, monkeypatch):
    (tmp_path / "model.onnx").write_text("fake")
    monkeypatch.setenv("REDSAN_TRIMMER_PATH", str(tmp_path))

    assert resolve_model_dir() == tmp_path.resolve()


def test_legacy_variable_ignored_when_edsan_variable_is_set(tmp_path: Path, cache_root: Path, monkeypatch):
    legacy_dir = tmp_path / "legacy"
    legacy_dir.mkdir()
    (legacy_dir / "model.onnx").write_text("fake")
    monkeypatch.setenv("EDSAN_TRIMMER_PATH", str(tmp_path / "missing"))
    monkeypatch.setenv("REDSAN_TRIMMER_PATH", str(legacy_dir))

    with pytest.warns(UserWarning, match="EDSAN_TRIMMER_PATH"):
        with pytest.raises(FileNotFoundError):
            resolve_model_dir()


def test_invalid_env_path_warns_and_falls_back_to_cache(tmp_path: Path, cache_root: Path, monkeypatch):
    make_artifact(cache_root / "1.3.0", "1.3.0")
    monkeypatch.setenv("EDSAN_TRIMMER_PATH", str(tmp_path / "missing"))

    with pytest.warns(UserWarning, match="EDSAN_TRIMMER_PATH"):
        resolved = resolve_model_dir()
    assert resolved == (cache_root / "1.3.0").resolve()


# --- versioned cache ---------------------------------------------------------


def test_highest_installed_version_is_selected(cache_root: Path):
    make_artifact(cache_root / "1.2.0", "1.2.0")
    make_artifact(cache_root / "1.3.0", "1.3.0")

    assert resolve_model_dir() == (cache_root / "1.3.0").resolve()


def test_versions_compare_numerically_not_lexically(cache_root: Path):
    make_artifact(cache_root / "1.9.0", "1.9.0")
    make_artifact(cache_root / "1.10.0", "1.10.0")

    assert resolve_model_dir() == (cache_root / "1.10.0").resolve()


def test_version_pin_selects_that_version(cache_root: Path, monkeypatch):
    make_artifact(cache_root / "1.2.0", "1.2.0")
    make_artifact(cache_root / "1.3.0", "1.3.0")
    monkeypatch.setenv("EDSAN_TRIMMER_VERSION", "1.2.0")

    assert resolve_model_dir() == (cache_root / "1.2.0").resolve()


def test_unknown_pin_raises_listing_installed_versions(cache_root: Path, monkeypatch):
    make_artifact(cache_root / "1.2.0", "1.2.0")
    make_artifact(cache_root / "1.3.0", "1.3.0")
    monkeypatch.setenv("EDSAN_TRIMMER_VERSION", "9.9.9")

    with pytest.raises(FileNotFoundError) as excinfo:
        resolve_model_dir()
    message = str(excinfo.value)
    assert "9.9.9" in message
    assert "1.3.0" in message and "1.2.0" in message


def test_pin_with_nothing_installed_says_none(cache_root: Path, monkeypatch):
    monkeypatch.setenv("EDSAN_TRIMMER_VERSION", "1.3.0")

    with pytest.raises(FileNotFoundError, match="Installed versions: none"):
        resolve_model_dir()


def test_pin_does_not_fall_back_to_legacy_slot(cache_root: Path, monkeypatch):
    make_artifact(cache_root / "v1", "1.3.0")
    monkeypatch.setenv("EDSAN_TRIMMER_VERSION", "1.3.0")

    with pytest.raises(FileNotFoundError):
        resolve_model_dir()


@pytest.mark.parametrize(
    "folder, manifest_version",
    [
        ("v1.3", "1.3.0"),  # hand-made name that is not a version
        ("1.3.0", "1.2.0"),  # manifest disagrees with the folder name
        ("1.3.0", None),  # manifest carries no version
        ("1.3.0-rc", "1.3.0-rc"),  # not a numeric version
    ],
)
def test_misnamed_or_inconsistent_folders_are_ignored(cache_root: Path, folder, manifest_version):
    make_artifact(cache_root / "1.2.0", "1.2.0")
    make_artifact(cache_root / folder, manifest_version)

    assert [a.version for a in list_installed_artifacts(cache_root)] == ["1.2.0"]
    assert resolve_model_dir() == (cache_root / "1.2.0").resolve()


def test_incomplete_or_incompatible_artifacts_are_ignored(cache_root: Path):
    make_artifact(cache_root / "1.2.0", "1.2.0")
    make_artifact(cache_root / "1.4.0", "1.4.0", omit=("trim_batch_service.py",))
    make_artifact(cache_root / "1.5.0", "1.5.0", worker_contract="something-else")
    make_artifact(cache_root / "1.1.0", "1.1.0")  # below the redsan floor

    assert resolve_model_dir() == (cache_root / "1.2.0").resolve()


def test_unreadable_manifest_is_ignored(cache_root: Path):
    make_artifact(cache_root / "1.2.0", "1.2.0")
    bad = make_artifact(cache_root / "1.3.0", "1.3.0")
    (bad / "artifact.json").write_text("{not json", encoding="utf-8")

    assert resolve_model_dir() == (cache_root / "1.2.0").resolve()


# --- legacy v1 slot ----------------------------------------------------------


def test_legacy_v1_slot_is_used_when_no_versioned_install_exists(cache_root: Path):
    make_artifact(cache_root / "v1", "1.3.0")

    with pytest.warns(UserWarning, match="legacy"):
        resolved = resolve_model_dir()
    assert resolved == (cache_root / "v1").resolve()


def test_versioned_install_wins_over_legacy_slot(cache_root: Path, recwarn):
    make_artifact(cache_root / "v1", "1.2.0")
    make_artifact(cache_root / "1.3.0", "1.3.0")

    assert resolve_model_dir() == (cache_root / "1.3.0").resolve()
    assert not [w for w in recwarn if "legacy" in str(w.message)]


def test_invalid_legacy_slot_is_not_used(cache_root: Path):
    legacy = cache_root / "v1"
    legacy.mkdir(parents=True)
    (legacy / "model.onnx").write_text("fake")  # no manifest, no worker

    with pytest.raises(FileNotFoundError):
        resolve_model_dir()


# --- no development-checkout fallback ---------------------------------------


def test_dev_checkout_export_folder_is_never_selected(cache_root: Path):
    export = Path.cwd() / "artifacts" / "active_learning" / "onnx_export"
    make_artifact(export, "1.3.0")

    with pytest.raises(FileNotFoundError):
        resolve_model_dir()


def test_dev_checkout_export_folder_is_used_only_when_named_explicitly(cache_root: Path, monkeypatch):
    export = make_artifact(Path.cwd() / "artifacts" / "active_learning" / "onnx_export", "1.3.0")
    monkeypatch.setenv("EDSAN_TRIMMER_PATH", str(export))

    assert resolve_model_dir() == export.resolve()


# --- extracted release run in place -----------------------------------------


def test_script_directory_is_last_resort(tmp_path: Path, cache_root: Path, monkeypatch):
    make_artifact(tmp_path / "extracted", "1.3.0")
    script = tmp_path / "extracted" / "trim_batch_service.py"
    monkeypatch.setattr(sys, "argv", [str(script)])

    assert resolve_model_dir() == (tmp_path / "extracted").resolve()


def test_installed_version_wins_over_script_directory(tmp_path: Path, cache_root: Path, monkeypatch):
    make_artifact(tmp_path / "extracted", "1.2.0")
    make_artifact(cache_root / "1.3.0", "1.3.0")
    monkeypatch.setattr(sys, "argv", [str(tmp_path / "extracted" / "trim_batch_service.py")])

    assert resolve_model_dir() == (cache_root / "1.3.0").resolve()


def test_current_directory_is_last_resort(cache_root: Path):
    (Path.cwd() / "model.onnx").write_text("fake")

    assert resolve_model_dir() == Path.cwd().resolve()


# --- not-found message -------------------------------------------------------


def test_not_found_message_is_release_agnostic_and_actionable(cache_root: Path):
    with pytest.raises(FileNotFoundError) as excinfo:
        resolve_model_dir()

    message = str(excinfo.value)
    assert "[ERROR] edsan-doc-trimmer model not found!" in message
    assert "EDSAN_TRIMMER_PATH" in message
    assert "EDSAN_TRIMMER_VERSION" in message
    assert "edsan_install_trimmer" in message
    assert str(cache_root) in message
    assert "1.3.0" not in message
    assert "1.2.0" not in message
    assert "onnx_export" not in message


# --- cache root per OS -------------------------------------------------------


@pytest.fixture
def clean_cache_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    for var in ("R_USER_CACHE_DIR", "XDG_CACHE_HOME", "LOCALAPPDATA"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))
    return tmp_path


def test_cache_root_prefers_r_user_cache_dir_over_xdg(clean_cache_env, monkeypatch):
    monkeypatch.setenv("R_USER_CACHE_DIR", str(clean_cache_env / "r"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(clean_cache_env / "xdg"))

    assert get_trimmer_cache_root() == clean_cache_env / "r" / "R" / "edsan_doc_trimmer"


def test_cache_root_honours_xdg_cache_home(clean_cache_env, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(clean_cache_env / "xdg"))

    assert get_trimmer_cache_root() == clean_cache_env / "xdg" / "R" / "edsan_doc_trimmer"


def test_cache_root_linux_default(clean_cache_env, monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")

    assert get_trimmer_cache_root() == clean_cache_env / "home" / ".cache" / "R" / "edsan_doc_trimmer"


def test_cache_root_macos_default(clean_cache_env, monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")

    expected = (
        clean_cache_env / "home" / "Library" / "Caches" / "org.R-project.R" / "R" / "edsan_doc_trimmer"
    )
    assert get_trimmer_cache_root() == expected


def test_cache_root_windows_default(clean_cache_env, monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("LOCALAPPDATA", str(clean_cache_env / "lad"))

    assert get_trimmer_cache_root() == clean_cache_env / "lad" / "R" / "cache" / "R" / "edsan_doc_trimmer"


def test_get_user_cache_dirs_returns_the_cache_root(cache_root: Path):
    assert get_user_cache_dirs() == [cache_root]

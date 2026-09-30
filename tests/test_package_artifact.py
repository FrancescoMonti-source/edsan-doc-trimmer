from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import package_artifact as packager
from package_artifact import (
    ALLOWLISTED_EXPORT_FILES,
    package_artifact,
    read_worker_contract,
    verify_archive,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
TESTED_WORKER = REPO_ROOT / "scripts" / "trim_batch_service.py"
MODEL_MTIME = datetime(2026, 9, 21, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def export_dir(tmp_path):
    """A training export folder, including leftovers a release must not ship."""

    folder = tmp_path / "export"
    folder.mkdir()
    for filename in ALLOWLISTED_EXPORT_FILES:
        (folder / filename).write_bytes(f"content of {filename}".encode())
    (folder / "model.safetensors").write_bytes(b"training checkpoint")
    os.utime(folder / "model.onnx", (MODEL_MTIME.timestamp(),) * 2)
    return folder


def snapshot(folder: Path) -> dict[str, tuple[int, int, bytes]]:
    return {
        str(p.relative_to(folder)): (
            p.stat().st_size,
            p.stat().st_mtime_ns,
            p.read_bytes(),
        )
        for p in sorted(folder.rglob("*"))
        if p.is_file()
    }


def package(export_dir, output_dir, **kwargs):
    kwargs.setdefault("version", "1.3.0")
    return package_artifact(
        onnx_dir=str(export_dir),
        worker_script=str(TESTED_WORKER),
        output_dir=str(output_dir),
        **kwargs,
    )


def test_packaging_leaves_export_folder_untouched(export_dir, tmp_path):
    before = snapshot(export_dir)

    package(export_dir, tmp_path / "out")

    assert snapshot(export_dir) == before


def test_archive_contains_exactly_allowlisted_files_with_manifest(export_dir, tmp_path):
    archive = package(export_dir, tmp_path / "out", version="1.3.0")

    assert archive == (tmp_path / "out" / "edsan-doc-trimmer-v1.3.0.zip").resolve()
    worker_bytes = TESTED_WORKER.read_bytes()
    with zipfile.ZipFile(archive) as packaged:
        assert sorted(packaged.namelist()) == sorted(ALLOWLISTED_EXPORT_FILES)
        assert packaged.read("trim_batch_service.py") == worker_bytes
        assert packaged.read("model.onnx") == (export_dir / "model.onnx").read_bytes()
        manifest = json.loads(packaged.read("artifact.json"))

    assert manifest["artifact_name"] == "edsan-doc-trimmer"
    assert manifest["artifact_version"] == "1.3.0"
    assert manifest["worker_contract"] == read_worker_contract(TESTED_WORKER)
    assert manifest["worker_contract"] == "model-only-v1"
    assert manifest["exported_at"] == MODEL_MTIME.strftime("%Y-%m-%d")


def test_exported_at_follows_model_file_date(export_dir, tmp_path):
    other = datetime(2025, 1, 2, 12, 0, 0, tzinfo=timezone.utc)
    os.utime(export_dir / "model.onnx", (other.timestamp(),) * 2)

    archive = package(export_dir, tmp_path / "out")

    with zipfile.ZipFile(archive) as packaged:
        assert json.loads(packaged.read("artifact.json"))["exported_at"] == "2025-01-02"


@pytest.mark.parametrize(
    "version",
    ["", "1.3", "v1.3.0", "1.3.0.0", "01.3.0", "latest", "2.0.0-rc.1", "1.0.0+build.5", "1.3.0\n"],
)
def test_invalid_version_is_rejected_before_any_work(export_dir, tmp_path, version):
    with pytest.raises(ValueError, match="not a release version"):
        package(export_dir, tmp_path / "out", version=version)

    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("version", ["1.3.0", "0.0.1", "10.20.30"])
def test_release_versions_are_accepted(export_dir, tmp_path, version):
    archive = package(export_dir, tmp_path / "out", version=version)

    assert archive.name == f"edsan-doc-trimmer-v{version}.zip"


def test_cli_requires_version(export_dir, tmp_path, capsys):
    with pytest.raises(SystemExit) as excinfo:
        packager.main(["--onnx_dir", str(export_dir), "--output_dir", str(tmp_path / "out")])

    assert excinfo.value.code != 0
    assert "--version" in capsys.readouterr().err


def test_cli_rejects_non_semver_with_clear_message(export_dir, tmp_path, capsys):
    with pytest.raises(SystemExit) as excinfo:
        packager.main(
            ["--version", "1.3", "--onnx_dir", str(export_dir), "--output_dir", str(tmp_path / "out")]
        )

    assert excinfo.value.code != 0
    assert "not a release version" in capsys.readouterr().err


def test_cli_prints_redsan_install_command(export_dir, tmp_path, capsys):
    packager.main(
        ["--version", "1.3.0", "--onnx_dir", str(export_dir), "--output_dir", str(tmp_path / "out")]
    )

    archive = (tmp_path / "out" / "edsan-doc-trimmer-v1.3.0.zip").resolve()
    assert f'redsan::edsan_install_trimmer("{archive.as_posix()}")' in capsys.readouterr().out


def test_existing_archive_is_refused_without_force(export_dir, tmp_path):
    archive = package(export_dir, tmp_path / "out")
    archive.write_bytes(b"precious earlier release")

    with pytest.raises(FileExistsError, match="--force"):
        package(export_dir, tmp_path / "out")
    assert archive.read_bytes() == b"precious earlier release"

    package(export_dir, tmp_path / "out", force=True)
    assert zipfile.is_zipfile(archive)


def test_cli_reports_existing_archive(export_dir, tmp_path, capsys):
    args = ["--version", "1.3.0", "--onnx_dir", str(export_dir), "--output_dir", str(tmp_path / "out")]
    packager.main(args)

    with pytest.raises(SystemExit) as excinfo:
        packager.main(args)

    assert excinfo.value.code != 0
    assert "--force" in capsys.readouterr().err
    packager.main([*args, "--force"])


def test_nothing_is_written_besides_archive_and_temp_dir(export_dir, tmp_path, monkeypatch):
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setenv("TMPDIR", str(scratch))
    monkeypatch.setenv("TEMP", str(scratch))
    monkeypatch.setenv("TMP", str(scratch))
    monkeypatch.setattr("tempfile.tempdir", None)
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("LOCALAPPDATA", str(home / "local"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(home / "xdg"))

    package(export_dir, tmp_path / "out")

    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == ["edsan-doc-trimmer-v1.3.0.zip"]
    assert list(scratch.iterdir()) == []
    assert list(home.iterdir()) == []


def test_cache_install_is_gone():
    assert "install_to_cache" not in inspect.signature(package_artifact).parameters
    assert not hasattr(packager, "get_user_cache_dirs")
    with pytest.raises(SystemExit):
        packager.main(["--version", "1.3.0", "--no_cache_install"])


def test_verify_archive_rejects_unexpected_members(tmp_path):
    bloated = tmp_path / "bloated.zip"
    with zipfile.ZipFile(bloated, "w") as zf:
        for name in ALLOWLISTED_EXPORT_FILES:
            zf.writestr(name, b"x")
        zf.writestr("model.safetensors", b"duplicate weights")

    with pytest.raises(RuntimeError, match="model.safetensors"):
        verify_archive(bloated)


def test_verify_archive_rejects_missing_members(tmp_path):
    short = tmp_path / "short.zip"
    with zipfile.ZipFile(short, "w") as zf:
        for name in ALLOWLISTED_EXPORT_FILES[1:]:
            zf.writestr(name, b"x")

    with pytest.raises(RuntimeError, match=ALLOWLISTED_EXPORT_FILES[0]):
        verify_archive(short)


def test_failed_verification_leaves_no_archive(export_dir, tmp_path, monkeypatch):
    def reject(archive):
        raise RuntimeError("bloated")

    monkeypatch.setattr(packager, "verify_archive", reject)

    with pytest.raises(RuntimeError, match="bloated"):
        package(export_dir, tmp_path / "out")

    assert list((tmp_path / "out").iterdir()) == []


def test_missing_export_file_is_reported(export_dir, tmp_path):
    (export_dir / "tokenizer.json").unlink()

    with pytest.raises(FileNotFoundError, match="tokenizer.json"):
        package(export_dir, tmp_path / "out")


def test_packaging_rejects_worker_without_declared_contract(tmp_path):
    worker = tmp_path / "worker.py"
    worker.write_text("print('worker')\n", encoding="utf-8")

    with pytest.raises(ValueError, match="WORKER_CONTRACT"):
        read_worker_contract(worker)


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.org", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


@pytest.fixture
def git_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    (repo / "a.txt").write_text("a")
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "first")
    return repo


def test_warns_when_release_tag_does_not_point_at_head(export_dir, tmp_path, git_repo, capsys):
    git(git_repo, "tag", "v1.3.0")
    (git_repo / "b.txt").write_text("b")
    git(git_repo, "add", ".")
    git(git_repo, "commit", "-q", "-m", "second")

    package(export_dir, tmp_path / "out", repo_root=git_repo)

    assert "v1.3.0" in capsys.readouterr().err


def test_no_warning_when_tag_matches_head_or_is_absent(export_dir, tmp_path, git_repo, capsys):
    package(export_dir, tmp_path / "out", repo_root=git_repo)
    assert capsys.readouterr().err == ""

    git(git_repo, "tag", "-a", "v1.3.0", "-m", "annotated")
    package(export_dir, tmp_path / "out", repo_root=git_repo, force=True)
    assert capsys.readouterr().err == ""


def test_warns_when_packaged_worker_has_uncommitted_changes(export_dir, tmp_path, git_repo, capsys):
    worker = git_repo / "trim_batch_service.py"
    worker.write_text(TESTED_WORKER.read_text(encoding="utf-8"), encoding="utf-8")
    git(git_repo, "add", ".")
    git(git_repo, "commit", "-q", "-m", "worker")

    def build(force=False):
        package_artifact(
            version="1.3.0",
            onnx_dir=str(export_dir),
            worker_script=str(worker),
            output_dir=str(tmp_path / "out"),
            force=force,
            repo_root=git_repo,
        )

    build()
    assert "uncommitted" not in capsys.readouterr().err

    worker.write_text(worker.read_text(encoding="utf-8") + "\n# local edit\n", encoding="utf-8")
    build(force=True)
    assert "uncommitted changes" in capsys.readouterr().err


def test_packaging_outside_git_repo_does_not_fail(export_dir, tmp_path):
    not_a_repo = tmp_path / "plain"
    not_a_repo.mkdir()

    package(export_dir, tmp_path / "out", repo_root=not_a_repo)

#!/usr/bin/env python3
"""Packages an immutable edsan-doc-trimmer release archive.

The training export folder is only ever read. The archive is assembled in a
temporary staging folder from an allowlist of export files, the worker script
and a generated ``artifact.json``, then re-opened and checked before it is
moved into place. Installation is left to ``redsan::edsan_install_trimmer()``.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ARCHIVE_PREFIX = "edsan-doc-trimmer"
MODEL_FILES = (
    "model.onnx",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "config.json",
)
WORKER_FILE = "trim_batch_service.py"
MANIFEST_FILE = "artifact.json"
ALLOWLISTED_EXPORT_FILES = (*MODEL_FILES, MANIFEST_FILE, WORKER_FILE)

_IDENT = r"(?:0|[1-9]\d*|\d*[A-Za-z-][0-9A-Za-z-]*)"
_SEMVER = re.compile(
    r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
    rf"(?:-{_IDENT}(?:\.{_IDENT})*)?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$"
)


def read_worker_contract(worker_path: Path) -> str:
    """Read the literal WORKER_CONTRACT implemented by the packaged worker."""

    module = ast.parse(
        worker_path.read_text(encoding="utf-8"), filename=str(worker_path)
    )
    for node in module.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "WORKER_CONTRACT"
            for target in node.targets
        ):
            continue
        value = ast.literal_eval(node.value)
        if isinstance(value, str) and value:
            return value
        break
    raise ValueError(
        f"Worker does not declare a literal WORKER_CONTRACT: {worker_path}"
    )


def validate_version(version: str) -> str:
    """Return ``version`` if it is a semantic version, else raise ValueError."""

    if not isinstance(version, str) or not _SEMVER.match(version):
        raise ValueError(
            f"Version {version!r} is not a semantic version (expected X.Y.Z, "
            "optionally with -prerelease or +build)."
        )
    return version


def archive_name(version: str) -> str:
    return f"{ARCHIVE_PREFIX}-v{version}.zip"


def warn_if_tag_not_at_head(version: str, repo_root: Path) -> None:
    """Warn when git tag ``v<version>`` exists but does not point at HEAD."""

    def rev(ref: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--verify", "--quiet", ref],
                cwd=repo_root,
                capture_output=True,
                text=True,
            )
        except OSError:
            return None
        return result.stdout.strip() if result.returncode == 0 else None

    tag = f"v{version}"
    tagged = rev(f"refs/tags/{tag}^{{commit}}")
    if tagged is None:
        return
    head = rev("HEAD")
    if head is not None and tagged != head:
        print(
            f"warning: git tag {tag} exists but does not point at HEAD "
            f"({tagged[:10]} vs {head[:10]}); this archive may not match the tag.",
            file=sys.stderr,
        )


def verify_archive(archive: Path) -> None:
    """Fail unless ``archive`` holds exactly the allowlisted files, once each."""

    with zipfile.ZipFile(archive) as packaged:
        names = packaged.namelist()
        expected = set(ALLOWLISTED_EXPORT_FILES)
        unexpected = sorted(set(names) - expected)
        missing = sorted(expected - set(names))
        duplicated = sorted({n for n in names if names.count(n) > 1})
        if unexpected or missing or duplicated:
            raise RuntimeError(
                f"Archive {Path(archive).name} does not contain exactly the "
                f"allowlisted files (unexpected: {unexpected}; missing: {missing}; "
                f"duplicated: {duplicated})."
            )
        corrupt = packaged.testzip()
        if corrupt is not None:
            raise RuntimeError(
                f"Archive {Path(archive).name} has a corrupt member: {corrupt}"
            )


def package_artifact(
    version: str,
    onnx_dir: str = "artifacts/active_learning/onnx_export",
    worker_script: str = "scripts/trim_batch_service.py",
    output_dir: str = "artifacts",
    force: bool = False,
    repo_root: Path | None = None,
) -> Path:
    """Build ``<output_dir>/edsan-doc-trimmer-v<version>.zip`` and return its path.

    ``onnx_dir`` is read-only: nothing is written into it.
    """

    validate_version(version)
    model_path = Path(onnx_dir).resolve()
    worker_path = Path(worker_script).resolve()
    out_zip_path = (Path(output_dir) / archive_name(version)).resolve()
    if repo_root is None:
        repo_root = Path(__file__).resolve().parent.parent

    if out_zip_path.exists() and not force:
        raise FileExistsError(
            f"{out_zip_path} already exists; releases are immutable. "
            "Pass --force to overwrite it."
        )
    for filename in MODEL_FILES:
        if not (model_path / filename).is_file():
            raise FileNotFoundError(
                f"Missing required artifact file: {model_path / filename}"
            )
    worker_contract = read_worker_contract(worker_path)
    warn_if_tag_not_at_head(version, repo_root)

    # UTC keeps the manifest date independent of the packaging machine's zone.
    exported_at = datetime.fromtimestamp(
        (model_path / "model.onnx").stat().st_mtime, tz=timezone.utc
    ).strftime("%Y-%m-%d")
    manifest = {
        "artifact_name": ARCHIVE_PREFIX,
        "artifact_version": version,
        "worker_contract": worker_contract,
        "model_type": "DrBERT-sequence-classification",
        "exported_at": exported_at,
    }

    out_zip_path.parent.mkdir(parents=True, exist_ok=True)
    partial = out_zip_path.with_name(out_zip_path.name + ".part")
    try:
        with tempfile.TemporaryDirectory(prefix="edsan-trimmer-staging-") as tmp:
            staging = Path(tmp)
            for filename in MODEL_FILES:
                shutil.copy2(model_path / filename, staging / filename)
            shutil.copy2(worker_path, staging / WORKER_FILE)
            (staging / MANIFEST_FILE).write_text(
                json.dumps(manifest, indent=2), encoding="utf-8"
            )

            print(f"Creating release archive: {out_zip_path}...")
            with zipfile.ZipFile(
                partial, "w", compression=zipfile.ZIP_DEFLATED
            ) as zf:
                for filename in ALLOWLISTED_EXPORT_FILES:
                    zf.write(staging / filename, arcname=filename)

        verify_archive(partial)
        os.replace(partial, out_zip_path)
    finally:
        partial.unlink(missing_ok=True)

    print(
        f"Archive packaged successfully: {out_zip_path} "
        f"({out_zip_path.stat().st_size:,} bytes)"
    )
    return out_zip_path


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(
        description="Package an immutable trimmer release archive"
    )
    parser.add_argument(
        "--version", required=True, help="Semantic version of the release, e.g. 1.3.0"
    )
    parser.add_argument("--onnx_dir", default="artifacts/active_learning/onnx_export")
    parser.add_argument("--worker", default="scripts/trim_batch_service.py")
    parser.add_argument(
        "--output_dir",
        default="artifacts",
        help="Folder receiving edsan-doc-trimmer-v<version>.zip",
    )
    parser.add_argument(
        "--force", action="store_true", help="Overwrite an existing archive"
    )
    args = parser.parse_args(argv)

    try:
        archive = package_artifact(
            version=args.version,
            onnx_dir=args.onnx_dir,
            worker_script=args.worker,
            output_dir=args.output_dir,
            force=args.force,
        )
    except (ValueError, FileExistsError, FileNotFoundError, RuntimeError) as exc:
        parser.error(str(exc))

    print("Install it with:")
    print(f'  redsan::edsan_install_trimmer("{archive.as_posix()}")')
    return archive


if __name__ == "__main__":
    main()

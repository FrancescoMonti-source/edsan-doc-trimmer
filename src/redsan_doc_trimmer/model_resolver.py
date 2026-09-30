"""Model resolution utilities for finding DrBERT ONNX trimmer artifacts.

Discovery mirrors the versioned cache of ``redsan`` (redsan#58). Everything that
assumes how that cache is laid out lives in the "cache contract" section below,
so a change on the R side is a change in one place here.

Order: explicit directory > ``EDSAN_TRIMMER_PATH`` (legacy
``REDSAN_TRIMMER_PATH``) > ``EDSAN_TRIMMER_VERSION`` pin > highest valid
``<cache root>/<version>/`` > legacy ``<cache root>/v1`` > the folder of the
invoked script / the working directory (an extracted release run in place).
The training export folder of a development checkout is never searched.
"""

from __future__ import annotations

import json
import os
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

# --------------------------------------------------------------------------- #
# Cache contract (mirrors redsan#58). Adjust here if redsan's layout changes.
# --------------------------------------------------------------------------- #

#: Package name redsan passes to ``tools::R_user_dir(<name>, "cache")``.
CACHE_PACKAGE_NAME = "edsan_doc_trimmer"
#: Pre-versioning install slot: ``<cache root>/v1`` (a cache layout, not a model version).
LEGACY_SLOT_NAME = "v1"
#: Explicit folder (or direct ``model.onnx`` path). ``REDSAN_*`` is the legacy alias.
PATH_ENV_VARS = ("EDSAN_TRIMMER_PATH", "REDSAN_TRIMMER_PATH")
#: Pins one installed version (exact folder name).
VERSION_ENV_VAR = "EDSAN_TRIMMER_VERSION"
#: Files an installed artifact must carry (redsan's artifact validation).
REQUIRED_ARTIFACT_FILES = (
    "model.onnx",
    "tokenizer.json",
    "trim_batch_service.py",
    "artifact.json",
)
MANIFEST_NAME = "artifact.json"
#: Manifest requirements redsan enforces on an artifact.
MIN_ARTIFACT_VERSION = (1, 2, 0)
REQUIRED_WORKER_CONTRACT = "model-only-v1"

# [0-9] and fullmatch, as in redsan's R pattern: \d would accept non-ASCII
# digits and $ a trailing newline.
_VERSION_RE = re.compile(r"[0-9]+(?:\.[0-9]+)*")


def get_trimmer_cache_root() -> Path:
    """Returns the cache root holding one folder per installed artifact version.

    Replicates ``tools::R_user_dir("edsan_doc_trimmer", "cache")``: the first of
    ``R_USER_CACHE_DIR``, ``XDG_CACHE_HOME`` or the per-OS default, followed by
    ``R/edsan_doc_trimmer``.
    """
    base = os.environ.get("R_USER_CACHE_DIR", "").strip() or os.environ.get(
        "XDG_CACHE_HOME", ""
    ).strip()
    if base:
        root = Path(base)
    elif sys.platform == "win32":
        local_appdata = os.environ.get("LOCALAPPDATA", "").strip()
        local = Path(local_appdata) if local_appdata else Path.home() / "AppData" / "Local"
        root = local / "R" / "cache"
    elif sys.platform == "darwin":
        root = Path.home() / "Library" / "Caches" / "org.R-project.R"
    else:
        root = Path.home() / ".cache"
    return root.expanduser() / "R" / CACHE_PACKAGE_NAME


def parse_artifact_version(text: object) -> tuple[int, ...] | None:
    """Parses a dotted numeric version (``1.3.0``); returns None when it isn't one."""
    if not isinstance(text, str) or not _VERSION_RE.fullmatch(text):
        return None
    return tuple(int(part) for part in text.split("."))


def _read_manifest(artifact_dir: Path) -> dict | None:
    try:
        manifest = json.loads((artifact_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return manifest if isinstance(manifest, dict) else None


def artifact_version_if_valid(artifact_dir: Path) -> str | None:
    """Returns the manifest's ``artifact_version`` if the folder is a valid artifact."""
    if not all((artifact_dir / name).is_file() for name in REQUIRED_ARTIFACT_FILES):
        return None
    manifest = _read_manifest(artifact_dir)
    if manifest is None:
        return None
    version = manifest.get("artifact_version")
    parsed = parse_artifact_version(version)
    if parsed is None or parsed < MIN_ARTIFACT_VERSION:
        return None
    if manifest.get("worker_contract") != REQUIRED_WORKER_CONTRACT:
        return None
    return version


@dataclass(frozen=True)
class InstalledArtifact:
    version: str
    path: Path


def list_installed_artifacts(cache_root: Path | None = None) -> list[InstalledArtifact]:
    """Installed artifacts under the cache root, highest version first.

    A folder counts only when its name equals its manifest's ``artifact_version``
    and it passes validation; anything else (``v1.3``, ``v1``, stray folders) is ignored.
    """
    root = get_trimmer_cache_root() if cache_root is None else cache_root
    try:
        children = [c for c in root.iterdir() if c.is_dir()]
    except OSError:
        return []
    found = []
    for child in children:
        if parse_artifact_version(child.name) is None:
            continue
        if artifact_version_if_valid(child) == child.name:
            found.append(InstalledArtifact(child.name, child))
    found.sort(key=lambda a: (parse_artifact_version(a.version), a.version), reverse=True)
    return found


def _legacy_slot(cache_root: Path) -> Path | None:
    slot = cache_root / LEGACY_SLOT_NAME
    return slot if artifact_version_if_valid(slot) is not None else None


# --------------------------------------------------------------------------- #
# Resolution
# --------------------------------------------------------------------------- #


def format_model_not_found_message(
    attempted_paths: list[Path], cache_root: Path | None = None
) -> str:
    """Formats a friendly error message explaining how to fix the missing model."""
    sep = "=" * 80
    attempted_str = "\n".join(f"  - {p}" for p in attempted_paths)
    root = get_trimmer_cache_root() if cache_root is None else cache_root
    return f"""
{sep}
[ERROR] edsan-doc-trimmer model not found!
{sep}
The DrBERT ONNX model file ('model.onnx') could not be located in any of the
following search locations:

{attempted_str}

WHY THIS HAPPENS:
  Large binary model files (~442 MB) are gitignored and not included in 'git clone'.
  Only a released, versioned archive is used at runtime; the training export
  folder of a development checkout is never searched.

HOW TO RESOLVE (Offline / Air-Gapped Hospital HDW Setup):

1. Obtain the release archive 'edsan-doc-trimmer-<version>.zip':
   - From hospital internal GitLab: Project Overview or Deploy > Releases
   - Or from your internal HDW shared model storage (e.g. /data/shared/models/)
   - Or copy from your internet-connected workstation via USB/SFTP.

2. Install it, preferably from R (redsan), which unpacks it into the versioned cache:
     library(redsan)
     edsan_install_trimmer("path/to/edsan-doc-trimmer-<version>.zip")
   Installed versions live in '{root}/<version>/'; the highest one is used
   automatically. Pin one with the {VERSION_ENV_VAR} environment variable.
   If Python is in a custom environment:
     Sys.setenv(REDSAN_PYTHON_PATH = "path/to/python")

3. Or point at an extracted archive explicitly:
     export EDSAN_TRIMMER_PATH="/path/to/extracted_folder"
     # Or in Windows PowerShell:
     $env:EDSAN_TRIMMER_PATH = "C:\\path\\to\\extracted_folder"
   Or pass --onnx_dir to the script.
{sep}
""".strip()


def _normalize_model_path(value: str | Path) -> Path:
    path = Path(value).expanduser().resolve()
    if path.is_file() and path.name.lower() == "model.onnx":
        path = path.parent
    return path


def _installed_versions_text(installed: list[InstalledArtifact]) -> str:
    return ", ".join(a.version for a in installed) if installed else "none"


def resolve_model_dir(candidate_dir: str | Path | None = None) -> Path:
    """Auto-resolves the directory containing model.onnx and tokenizer assets.

    Resolution precedence:
    1. An explicit `candidate_dir` (a directory holding `model.onnx`, or the
       `model.onnx` file itself). Wins over everything; if invalid, raises.
    2. `EDSAN_TRIMMER_PATH` (or, only when that is unset, the legacy
       `REDSAN_TRIMMER_PATH`). If it does not hold a model, raises
       FileNotFoundError; it never falls through.
    3. `EDSAN_TRIMMER_VERSION` pin: `<cache root>/<version>/`. A pin that is not
       installed raises FileNotFoundError listing the installed versions.
    4. The highest valid installed version under the cache root.
    5. The legacy `<cache root>/v1` slot (with a warning asking to reinstall).
    6. The folder of the invoked script, then the working directory (an
       extracted release run in place).

    The training export folder of a development checkout is never searched.

    Raises:
        FileNotFoundError: if no model can be found, or the pinned version is not installed.
    """
    # 1. Explicit candidate_dir
    if candidate_dir is not None and str(candidate_dir).strip():
        cand = _normalize_model_path(candidate_dir)
        if (cand / "model.onnx").is_file():
            return cand
        raise FileNotFoundError(format_model_not_found_message([cand]))

    attempted: list[Path] = []
    cache_root = get_trimmer_cache_root()

    # 2. Explicit path from the environment
    env_var, env_value = next(
        ((v, os.environ[v].strip()) for v in PATH_ENV_VARS if os.environ.get(v, "").strip()),
        (None, ""),
    )
    if env_var:
        p = _normalize_model_path(env_value)
        attempted.append(p)
        if (p / "model.onnx").is_file():
            return p
        # Falling through would silently run a different artifact than the one named.
        raise FileNotFoundError(
            f"{env_var} is set to '{env_value}', but 'model.onnx' was not found in that "
            f"folder. Fix the path, or unset {env_var} to use the installed trimmer versions."
        )

    # 3. Version pin
    installed = list_installed_artifacts(cache_root)
    pin = os.environ.get(VERSION_ENV_VAR, "").strip()
    if pin:
        for artifact in installed:
            if artifact.version == pin:
                return artifact.path.resolve()
        raise FileNotFoundError(
            f"{VERSION_ENV_VAR} is set to '{pin}', but that edsan-doc-trimmer version is "
            f"not installed in {cache_root}. "
            f"Installed versions: {_installed_versions_text(installed)}."
        )

    # 4. Highest installed version
    if installed:
        return installed[0].path.resolve()
    attempted.append(cache_root / "<version>")

    # 5. Legacy v1 slot
    legacy = _legacy_slot(cache_root)
    if legacy is not None:
        warnings.warn(
            f"Using the legacy trimmer install at {legacy}; reinstall the release with "
            "redsan::edsan_install_trimmer() to place it in a versioned folder.",
            stacklevel=2,
        )
        return legacy.resolve()
    attempted.append(cache_root / LEGACY_SLOT_NAME)

    # 6. Extracted release run in place: invoked script's folder, then cwd
    in_place: list[Path] = []
    if getattr(sys, "argv", None) and sys.argv and sys.argv[0]:
        try:
            script_path = Path(sys.argv[0]).expanduser().resolve()
            if script_path.is_file():
                in_place.append(script_path.parent)
        except (OSError, ValueError, TypeError):
            pass  # sys.argv[0] might not be a valid filesystem path
    in_place.append(Path.cwd().resolve())
    for folder in in_place:
        if folder not in attempted:
            attempted.append(folder)
        if (folder / "model.onnx").is_file():
            return folder

    raise FileNotFoundError(format_model_not_found_message(attempted, cache_root))

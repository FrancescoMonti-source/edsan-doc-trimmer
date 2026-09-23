"""Model resolution utilities for finding DrBERT ONNX trimmer artifacts."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def get_user_cache_dirs() -> list[Path]:
    """Returns candidate user cache directories where models may be installed.

    Includes the standard R user cache directory used by
    `redsan::edsan_install_trimmer()` as well as standard Python cache locations.
    """
    candidates: list[Path] = []
    home = Path.home()

    # Windows user cache locations
    if os.name == "nt":
        local_appdata = os.environ.get("LOCALAPPDATA")
        if local_appdata:
            lad = Path(local_appdata)
            # R cache directory: tools::R_user_dir("edsan_doc_trimmer", "cache")/v1
            candidates.append(lad / "R" / "cache" / "R" / "edsan_doc_trimmer" / "v1")
            candidates.append(lad / "edsan_doc_trimmer" / "v1")
        candidates.append(home / "AppData" / "Local" / "R" / "cache" / "R" / "edsan_doc_trimmer" / "v1")
        candidates.append(home / "AppData" / "Local" / "edsan_doc_trimmer" / "v1")

    # Linux / Unix user cache locations (XDG)
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache:
        xc = Path(xdg_cache)
        candidates.append(xc / "R" / "edsan_doc_trimmer" / "v1")
        candidates.append(xc / "edsan_doc_trimmer" / "v1")
    candidates.append(home / ".cache" / "R" / "edsan_doc_trimmer" / "v1")
    candidates.append(home / ".cache" / "edsan_doc_trimmer" / "v1")

    # macOS user cache locations
    candidates.append(home / "Library" / "Caches" / "org.R-project.R" / "R" / "edsan_doc_trimmer" / "v1")
    candidates.append(home / "Library" / "Caches" / "edsan_doc_trimmer" / "v1")

    # Deduplicate while preserving order
    seen: set[str] = set()
    deduped: list[Path] = []
    for c in candidates:
        norm = str(c.resolve()) if c.exists() else str(c)
        if norm not in seen:
            seen.add(norm)
            deduped.append(c)
    return deduped


def format_model_not_found_message(attempted_paths: list[Path]) -> str:
    """Formats an idiot-proof, friendly error message explaining how to fix the missing model."""
    sep = "=" * 80
    attempted_str = "\n".join(f"  - {p}" for p in attempted_paths)
    return f"""
{sep}
[ERROR] edsan-doc-trimmer model not found!
{sep}
The DrBERT ONNX model file ('model.onnx') could not be located in any of the
following search locations:

{attempted_str}

WHY THIS HAPPENS:
  Large binary model files (~442 MB) are gitignored and not included in 'git clone'
  to keep repository operations fast and lightweight.

HOW TO RESOLVE (Offline / Air-Gapped Hospital HDW Setup):

1. Obtain 'edsan-doc-trimmer-v1.3.0.zip':
   - From hospital internal GitLab: Project Overview or Deploy > Releases (tag v1.3.0)
   - Or from your internal HDW shared model storage (e.g. /data/shared/models/)
   - Or copy from your internet-connected workstation via USB/SFTP.

2. Set up the model for Python:
   Option A (Inside this repository):
     Extract directly into the artifacts folder:
       # Linux / macOS:
       unzip edsan-doc-trimmer-v1.3.0.zip -d artifacts/active_learning/onnx_export
       # Windows PowerShell:
       Expand-Archive -Path edsan-doc-trimmer-v1.3.0.zip -DestinationPath artifacts/active_learning/onnx_export -Force

   Option B (Shared HDW folder or custom directory):
     Extract anywhere (e.g. /data/shared/models/edsan-doc-trimmer/v1.3.0)
     and set the EDSAN_TRIMMER_PATH environment variable:
       export EDSAN_TRIMMER_PATH="/data/shared/models/edsan-doc-trimmer/v1.3.0"
       # Or in Windows PowerShell:
       $env:EDSAN_TRIMMER_PATH = "C:\\path\\to\\extracted_folder"

   Option C (Standard user cache):
     Extract into your user cache directory:
       # Linux:   ~/.cache/R/edsan_doc_trimmer/v1/
       # Windows: %LOCALAPPDATA%\\R\\cache\\R\\edsan_doc_trimmer\\v1\\

3. If running from R (redsan):
     library(redsan)
     edsan_install_trimmer("path/to/edsan-doc-trimmer-v1.3.0.zip")
     # If Python executable is in a custom environment:
     Sys.setenv(REDSAN_PYTHON_PATH = "path/to/python")
{sep}
""".strip()


def resolve_model_dir(candidate_dir: str | Path | None = None) -> Path:
    """Auto-resolves the directory containing model.onnx and tokenizer assets.

    Resolution precedence:
    - If `candidate_dir` is explicitly provided:
      Validates that `candidate_dir` contains `model.onnx` (or if pointing directly
      to `model.onnx`, uses its parent). If not, raises FileNotFoundError.
    - If `candidate_dir` is None or empty:
      1. EDSAN_TRIMMER_PATH or REDSAN_TRIMMER_PATH environment variable
      2. Script directory (if running via `python path/to/script.py` in an unzipped release)
      3. Standard user cache directories (e.g. populated by redsan::edsan_install_trimmer())
      4. Repository artifacts relative to current working directory (`artifacts/active_learning/onnx_export`)
      5. Repository artifacts relative to this module
      6. Current working directory

    Raises:
        FileNotFoundError: with an informative, idiot-proof troubleshooting message
        if model.onnx cannot be found.
    """
    # 1. Explicit candidate_dir if provided and non-empty
    if candidate_dir is not None and str(candidate_dir).strip():
        cand = Path(candidate_dir).expanduser().resolve()
        if cand.is_file() and cand.name.lower() == "model.onnx":
            cand = cand.parent
        if (cand / "model.onnx").is_file():
            return cand
        raise FileNotFoundError(format_model_not_found_message([cand]))

    attempted: list[Path] = []

    # 2. Environment variables: EDSAN_TRIMMER_PATH / REDSAN_TRIMMER_PATH
    for env_var in ("EDSAN_TRIMMER_PATH", "REDSAN_TRIMMER_PATH"):
        val = os.environ.get(env_var, "").strip()
        if val:
            p = Path(val).expanduser().resolve()
            if p.is_file() and p.name.lower() == "model.onnx":
                p = p.parent
            if p not in attempted:
                attempted.append(p)
            if (p / "model.onnx").is_file():
                return p

    # 3. Invoked script directory (for standalone extracted zip with trim_batch_service.py)
    if getattr(sys, "argv", None) and sys.argv and sys.argv[0]:
        try:
            script_path = Path(sys.argv[0]).expanduser().resolve()
            if script_path.is_file():
                script_dir = script_path.parent
                if script_dir not in attempted:
                    attempted.append(script_dir)
                if (script_dir / "model.onnx").is_file():
                    return script_dir
        except (OSError, ValueError, TypeError):
            pass  # sys.argv[0] might not be a valid filesystem path


    # 4. Standard user cache directories (from R or Python)
    for cache_p in get_user_cache_dirs():
        resolved_c = cache_p.expanduser().resolve()
        if resolved_c not in attempted:
            attempted.append(resolved_c)
        if (resolved_c / "model.onnx").is_file():
            return resolved_c

    # 5. Repository artifacts relative to current working directory
    cwd_candidate = (Path.cwd() / "artifacts" / "active_learning" / "onnx_export").resolve()
    if cwd_candidate not in attempted:
        attempted.append(cwd_candidate)
    if (cwd_candidate / "model.onnx").is_file():
        return cwd_candidate

    # 6. Relative to this module
    here = Path(__file__).resolve().parent
    repo_cand = (here.parent.parent / "artifacts" / "active_learning" / "onnx_export").resolve()
    if repo_cand not in attempted:
        attempted.append(repo_cand)
    if (repo_cand / "model.onnx").is_file():
        return repo_cand

    # 7. Current directory (for unzipped release run directly from directory)
    cur_dir = Path.cwd().resolve()
    if cur_dir not in attempted:
        attempted.append(cur_dir)
    if (cur_dir / "model.onnx").is_file():
        return cur_dir

    raise FileNotFoundError(format_model_not_found_message(attempted))


#!/usr/bin/env python3
"""Batch document trimming service for redsan R integration.

Accepts JSON input from a file or stdin containing a list of {id, text},
runs batched DrBERT ONNX inference,
and outputs JSON with trimmed text and exact [start_char, end_char] intervals.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

# Ensure repo src/ is in sys.path if running directly from script
_REPO_SRC = Path(__file__).resolve().parent.parent / "src"
if _REPO_SRC.exists() and str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

from dataclasses import dataclass

import numpy as np
import onnxruntime as ort

WORKER_CONTRACT = "model-only-v1"
DEVICE_ENV_VAR = "EDSAN_TRIMMER_DEVICE"
DEVICE_PROVIDERS = {
    "cpu": "CPUExecutionProvider",
    "cuda": "CUDAExecutionProvider",
    "openvino": "OpenVINOExecutionProvider",
    "dml": "DmlExecutionProvider",
    "migraphx": "MIGraphXExecutionProvider",
}
DEVICE_MODES_BY_PROVIDER = {
    provider: mode for mode, provider in DEVICE_PROVIDERS.items()
}
PROVIDER_INSTALL_HINTS = {
    "cuda": "Install onnxruntime-gpu in the Python environment used by redsan.",
    "openvino": (
        "Install onnxruntime-openvino and the OpenVINO runtime in the Python "
        "environment used by redsan."
    ),
    "dml": "Install an ONNX Runtime build that provides DmlExecutionProvider.",
    "migraphx": (
        "Install an ONNX Runtime build with MIGraphXExecutionProvider and its "
        "ROCm/MIGraphX runtime."
    ),
}
RUNTIME_NOTICE_PREFIX = "EDSAN_TRIMMER_NOTICE:"

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None

try:
    from tokenizers import Tokenizer
    USE_RUST_TOKENIZER = True
except ImportError:
    Tokenizer = None
    USE_RUST_TOKENIZER = False

# Resilient imports: use redsan_doc_trimmer package if installed/in path,
# or fallback to self-contained definitions for standalone unzipped deployment
try:
    from redsan_doc_trimmer.dataset import (
        DocumentSpan,
        WindowedLineSample,
        build_windowed_samples,
        extract_lines_with_offsets,
    )
except ImportError:
    @dataclass(frozen=True)
    class DocumentSpan:
        start_char: int
        end_char: int
        text: str
        is_boilerplate: bool = False
        label_source: str = "unlabeled"

    @dataclass(frozen=True)
    class WindowedLineSample:
        line_index: int
        start_char: int
        end_char: int
        target_text: str
        context_before: str
        context_after: str
        is_boilerplate: bool = False
        doc_id: str | None = None

        def to_input_pair(self) -> tuple[str, str]:
            context = f"{self.context_before} \n {self.context_after}".strip()
            return self.target_text, context

    def extract_lines_with_offsets(rectxt: str) -> list[DocumentSpan]:
        lines: list[DocumentSpan] = []
        current_offset = 0
        raw_lines = rectxt.splitlines(keepends=True)
        for raw_line in raw_lines:
            start = current_offset
            end = current_offset + len(raw_line)
            current_offset = end
            clean_text = raw_line.rstrip("\r\n")
            lines.append(
                DocumentSpan(
                    start_char=start,
                    end_char=start + len(clean_text),
                    text=clean_text,
                )
            )
        return lines

    def build_windowed_samples(
        spans: list[DocumentSpan],
        window_size: int = 2,
        doc_id: str | None = None,
    ) -> list[WindowedLineSample]:
        samples: list[WindowedLineSample] = []
        n = len(spans)
        for i, span in enumerate(spans):
            prev_lines = [
                spans[j].text
                for j in range(max(0, i - window_size), i)
                if spans[j].text.strip()
            ]
            context_before = " \n ".join(prev_lines)
            next_lines = [
                spans[j].text
                for j in range(i + 1, min(n, i + 1 + window_size))
                if spans[j].text.strip()
            ]
            context_after = " \n ".join(next_lines)
            samples.append(
                WindowedLineSample(
                    line_index=i,
                    start_char=span.start_char,
                    end_char=span.end_char,
                    target_text=span.text,
                    context_before=context_before,
                    context_after=context_after,
                    is_boilerplate=span.is_boilerplate,
                    doc_id=doc_id,
                )
            )
        return samples

try:
    from redsan_doc_trimmer.model_resolver import resolve_model_dir
except ImportError:
    def resolve_model_dir(candidate_dir: str | Path | None = None) -> Path:
        if candidate_dir is not None and str(candidate_dir).strip():
            cand = Path(candidate_dir).expanduser().resolve()
            if cand.is_file() and cand.name.lower() == "model.onnx":
                cand = cand.parent
            if (cand / "model.onnx").is_file():
                return cand
            raise FileNotFoundError(f"[ERROR] model.onnx not found in: {cand}")

        for env_var in ("EDSAN_TRIMMER_PATH", "REDSAN_TRIMMER_PATH"):
            val = os.environ.get(env_var, "").strip()
            if val:
                p = Path(val).expanduser().resolve()
                if p.is_file() and p.name.lower() == "model.onnx":
                    p = p.parent
                if (p / "model.onnx").is_file():
                    return p

        # Check script parent directory
        script_p = Path(__file__).resolve().parent
        if (script_p / "model.onnx").is_file():
            return script_p

        # Check cwd
        cur = Path.cwd().resolve()
        if (cur / "model.onnx").is_file():
            return cur

        repo_cand = cur / "artifacts" / "active_learning" / "onnx_export"
        if (repo_cand / "model.onnx").is_file():
            return repo_cand

        # Check standard user cache directories (populated by redsan::edsan_install_trimmer)
        home = Path.home()
        cache_cands: list[Path] = []
        if os.name == "nt":
            lad = os.environ.get("LOCALAPPDATA")
            if lad:
                cache_cands.append(Path(lad) / "R" / "cache" / "R" / "edsan_doc_trimmer" / "v1")
                cache_cands.append(Path(lad) / "edsan_doc_trimmer" / "v1")
            cache_cands.append(home / "AppData" / "Local" / "R" / "cache" / "R" / "edsan_doc_trimmer" / "v1")
            cache_cands.append(home / "AppData" / "Local" / "edsan_doc_trimmer" / "v1")
        else:
            xdg = os.environ.get("XDG_CACHE_HOME")
            if xdg:
                cache_cands.append(Path(xdg) / "R" / "edsan_doc_trimmer" / "v1")
                cache_cands.append(Path(xdg) / "edsan_doc_trimmer" / "v1")
            cache_cands.append(home / ".cache" / "R" / "edsan_doc_trimmer" / "v1")
            cache_cands.append(home / ".cache" / "edsan_doc_trimmer" / "v1")

        for c in cache_cands:
            if (c / "model.onnx").is_file():
                return c

        raise FileNotFoundError(
            "================================================================================\n"
            "[ERROR] edsan-doc-trimmer model not found!\n"
            "================================================================================\n"
            "The model file 'model.onnx' could not be located.\n"
            "HOW TO FIX:\n"
            "  1. Set the EDSAN_TRIMMER_PATH environment variable to the model folder:\n"
            "     export EDSAN_TRIMMER_PATH=/path/to/extracted_model\n"
            "  2. Or specify --onnx_dir /path/to/extracted_model\n"
            "================================================================================"
        )


def _detect_accelerators() -> set[str]:
    """Detect GPU vendors without consulting ONNX Runtime provider availability."""
    vendor_ids = {
        "10DE": "nvidia",
        "8086": "intel",
    }
    detected: set[str] = set()

    if os.name == "nt":
        powershell = (
            shutil.which("powershell.exe")
            or shutil.which("powershell")
            or shutil.which("pwsh.exe")
            or shutil.which("pwsh")
        )
        if not powershell:
            return detected
        command = (
            "Get-CimInstance Win32_VideoController | "
            "ForEach-Object { $_.PNPDeviceID }"
        )
        try:
            result = subprocess.run(
                [powershell, "-NoProfile", "-NonInteractive", "-Command", command],
                capture_output=True,
                check=False,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.TimeoutExpired):
            return detected
        for vendor_id in re.findall(r"VEN_([0-9A-F]{4})", result.stdout.upper()):
            vendor = vendor_ids.get(vendor_id)
            if vendor:
                detected.add(vendor)
        return detected

    if sys.platform.startswith("linux"):
        pci_devices = Path("/sys/bus/pci/devices")
        if not pci_devices.is_dir():
            return detected
        for device in pci_devices.iterdir():
            try:
                device_class = int((device / "class").read_text().strip(), 16)
                vendor_id = (device / "vendor").read_text().strip().upper()
            except (OSError, ValueError):
                continue
            if device_class >> 16 != 0x03:
                continue
            vendor = vendor_ids.get(vendor_id.removeprefix("0X"))
            if vendor:
                detected.add(vendor)

    return detected


def _select_execution_provider(
    device_mode: str | None = None,
) -> tuple[str, list[tuple[str, str]]]:
    mode = device_mode
    if mode is None:
        mode = os.environ.get(DEVICE_ENV_VAR, "auto")
    mode = mode.strip().lower()
    allowed_modes = ("auto", *DEVICE_PROVIDERS)
    if mode not in allowed_modes:
        raise ValueError(
            f"Invalid {DEVICE_ENV_VAR} value {mode!r}; choose one of: "
            + "|".join(allowed_modes)
        )

    cpu_provider = DEVICE_PROVIDERS["cpu"]
    if mode == "cpu":
        return cpu_provider, []

    available = set(ort.get_available_providers())
    if mode != "auto":
        provider = DEVICE_PROVIDERS[mode]
        if provider in available:
            return provider, []
        hint = PROVIDER_INSTALL_HINTS[mode]
        return cpu_provider, [
            (
                "WARNING",
                (
                    f"Requested {mode} mode requires {provider}, which is unavailable; "
                    f"falling back to {cpu_provider}. {hint}"
                ),
            )
        ]

    detected = _detect_accelerators()
    priority = (
        ("nvidia", "cuda", "NVIDIA"),
        ("intel", "openvino", "Intel"),
    )
    missing: list[tuple[str, str, str]] = []
    for vendor, mode_name, label in priority:
        if vendor not in detected:
            continue
        provider = DEVICE_PROVIDERS[mode_name]
        if provider in available:
            notices = [
                (
                    "WARNING",
                    (
                        f"{missing_label} GPU detected, but {missing_provider} is "
                        f"unavailable; using {provider}. "
                        f"{PROVIDER_INSTALL_HINTS[missing_mode]}"
                    ),
                )
                for missing_label, missing_mode, missing_provider in missing
            ]
            return provider, notices
        missing.append((label, mode_name, provider))

    notices = [
        (
            "WARNING",
            (
                f"{label} GPU detected, but {provider} is unavailable; falling back "
                f"to {cpu_provider}. {PROVIDER_INSTALL_HINTS[mode_name]}"
            ),
        )
        for label, mode_name, provider in missing
    ]
    return cpu_provider, notices


def _emit_runtime_notices(notices: list[tuple[str, str]]) -> None:
    for level, message in notices:
        single_line = " ".join(message.splitlines())
        sys.stderr.write(f"{RUNTIME_NOTICE_PREFIX}{level}:{single_line}\n")
    if notices:
        sys.stderr.flush()


def _session_providers(
    execution_provider: str,
) -> list[str | tuple[str, dict[str, str]]]:
    if execution_provider == DEVICE_PROVIDERS["openvino"]:
        return [
            (execution_provider, {"device_type": "GPU"}),
            DEVICE_PROVIDERS["cpu"],
        ]
    if execution_provider == DEVICE_PROVIDERS["cpu"]:
        return [execution_provider]
    return [execution_provider, DEVICE_PROVIDERS["cpu"]]


def _empty_result(doc_id: object, raw_text: str) -> dict:
    return {
        "id": doc_id,
        "trimmed_text": "",
        "raw_chars": len(raw_text),
        "trimmed_chars": 0,
        "reduction_pct": 100.0 if raw_text else 0.0,
        "preserved_intervals": [],
        "removed_intervals": [],
        "execution_provider": None,
    }

class _ProviderExecutionError(RuntimeError):
    def __init__(self, provider: str, cause: Exception):
        super().__init__(str(cause))
        self.provider = provider
        self.cause = cause


def _trim_batch_with_provider(
    documents: list[dict],
    onnx_dir: str | Path | None = None,
    threshold: float = 0.50,
    batch_size: int = 128,
    *,
    device_mode: str | None = None,
    extra_notices: list[tuple[str, str]] | None = None,
) -> list[dict]:
    precomputed_results: dict[int, dict] = {}
    for index, doc in enumerate(documents):
        doc_id = doc.get("id", "doc")
        raw_text = doc.get("text", "")

        if not raw_text or not raw_text.strip():
            precomputed_results[index] = _empty_result(doc_id, raw_text)

    # An entirely empty batch does not need to initialize the model runtime.
    # Every non-empty document follows the same Student v3 inference path.
    if len(precomputed_results) == len(documents):
        return [precomputed_results[index] for index in range(len(documents))]

    dir_path = resolve_model_dir(onnx_dir)
    onnx_path = str(dir_path / "model.onnx")
    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = os.cpu_count() or 8
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    execution_provider, notices = _select_execution_provider(device_mode)
    if extra_notices:
        notices.extend(extra_notices)
    session_providers = _session_providers(execution_provider)
    try:
        session = ort.InferenceSession(
            onnx_path,
            sess_opts,
            providers=session_providers,
        )
    except Exception as err:
        if execution_provider == DEVICE_PROVIDERS["cpu"]:
            raise
        failed_provider = execution_provider
        execution_provider = DEVICE_PROVIDERS["cpu"]
        session = ort.InferenceSession(
            onnx_path,
            sess_opts,
            providers=[execution_provider],
        )
        failed_mode = DEVICE_MODES_BY_PROVIDER[failed_provider]
        notices.append(
            (
                "WARNING",
                (
                    f"Could not initialize {failed_provider} ({err}); falling back to "
                    f"{execution_provider}. {PROVIDER_INSTALL_HINTS[failed_mode]}"
                ),
            )
        )

    active_providers = session.get_providers()
    if (
        execution_provider != DEVICE_PROVIDERS["cpu"]
        and execution_provider not in active_providers
    ):
        failed_provider = execution_provider
        execution_provider = DEVICE_PROVIDERS["cpu"]
        if execution_provider not in active_providers:
            session = ort.InferenceSession(
                onnx_path,
                sess_opts,
                providers=[execution_provider],
            )
            active_providers = session.get_providers()
        failed_mode = DEVICE_MODES_BY_PROVIDER[failed_provider]
        notices.append(
            (
                "WARNING",
                (
                    f"ONNX Runtime initialized {failed_provider} with "
                    f"{active_providers}; falling back to {execution_provider}. "
                    f"{PROVIDER_INSTALL_HINTS[failed_mode]}"
                ),
            )
        )

    session.disable_fallback()

    _emit_runtime_notices(notices)
    for result in precomputed_results.values():
        result["execution_provider"] = execution_provider

    tok_json = dir_path / "tokenizer.json"
    if USE_RUST_TOKENIZER and Tokenizer is not None and tok_json.exists():
        tokenizer = Tokenizer.from_file(str(tok_json))
        tokenizer.enable_padding(length=128, pad_id=1, pad_token="<pad>")
        tokenizer.enable_truncation(max_length=128)
        rust_tok = True
    elif AutoTokenizer is not None:
        tokenizer = AutoTokenizer.from_pretrained(str(dir_path))
        rust_tok = False
    else:
        raise RuntimeError(
            "Neither 'tokenizers' nor 'transformers' is installed. "
            "Please run: pip install tokenizers"
        )

    results = []

    for index, doc in enumerate(documents):
        if index in precomputed_results:
            results.append(precomputed_results[index])
            continue

        doc_id = doc.get("id", "doc")
        raw_text = doc.get("text", "")

        spans = extract_lines_with_offsets(raw_text)
        samples = build_windowed_samples(spans, window_size=2)

        line_is_bp = [False] * len(spans)

        if samples:
            for i in range(0, len(samples), batch_size):
                batch = samples[i : i + batch_size]
                targets = [s.target_text for s in batch]
                contexts = [f"{s.context_before} \n {s.context_after}".strip() for s in batch]

                if rust_tok:
                    pairs = list(zip(targets, contexts, strict=False))
                    encs = tokenizer.encode_batch(pairs)
                    input_ids = np.array([e.ids for e in encs], dtype=np.int64)
                    attention_mask = np.array([e.attention_mask for e in encs], dtype=np.int64)
                    ort_inputs = {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                    }
                else:
                    encodings = tokenizer(
                        targets,
                        contexts,
                        padding=True,
                        truncation=True,
                        max_length=128,
                        return_tensors="np",
                    )
                    ort_inputs = {
                        "input_ids": encodings["input_ids"],
                        "attention_mask": encodings["attention_mask"],
                    }
                try:
                    logits = session.run(["logits"], ort_inputs)[0]
                except ort.capi.onnxruntime_pybind11_state.EPFail as err:
                    if execution_provider == DEVICE_PROVIDERS["cpu"]:
                        raise
                    raise _ProviderExecutionError(execution_provider, err) from err
                exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
                probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

                for s, p in zip(batch, probs, strict=False):
                    p_bp = float(p[1])
                    if p_bp > threshold:
                        line_is_bp[s.line_index] = True

        # Reconstruct preserved and removed intervals (1-indexed for R!)
        preserved_intervals = []
        removed_intervals = []
        clinical_chunks = []

        for i, span in enumerate(spans):
            if not span.text.strip():
                continue

            # Convert 0-indexed [start_char, end_char) to 1-indexed inclusive [start, end] for R
            r_start = span.start_char + 1
            r_end = span.end_char

            if line_is_bp[i]:
                removed_intervals.append({
                    "start": r_start,
                    "end": r_end,
                    "family": "boilerplate",
                    "text": span.text,
                })
            else:
                clinical_chunks.append(span.text)
                preserved_intervals.append({
                    "start": r_start,
                    "end": r_end,
                    "family": "clinical",
                    "text": span.text,
                })

        trimmed_text = "\n".join(c for c in clinical_chunks if c.strip())
        raw_len = len(raw_text)
        trim_len = len(trimmed_text)
        reduction = ((raw_len - trim_len) / max(1, raw_len)) * 100.0

        results.append({
            "id": doc_id,
            "trimmed_text": trimmed_text,
            "raw_chars": raw_len,
            "trimmed_chars": trim_len,
            "reduction_pct": round(reduction, 2),
            "preserved_intervals": preserved_intervals,
            "removed_intervals": removed_intervals,
            "execution_provider": execution_provider,
        })

    return results


def trim_batch(
    documents: list[dict],
    onnx_dir: str | Path | None = None,
    threshold: float = 0.50,
    batch_size: int = 128,
) -> list[dict]:
    try:
        return _trim_batch_with_provider(
            documents,
            onnx_dir=onnx_dir,
            threshold=threshold,
            batch_size=batch_size,
        )
    except _ProviderExecutionError as err:
        failed_mode = DEVICE_MODES_BY_PROVIDER[err.provider]
        notice = (
            "WARNING",
            (
                f"Inference with {err.provider} failed ({err.cause}); rerunning the "
                f"entire batch with {DEVICE_PROVIDERS['cpu']}. "
                f"{PROVIDER_INSTALL_HINTS[failed_mode]}"
            ),
        )
        return _trim_batch_with_provider(
            documents,
            onnx_dir=onnx_dir,
            threshold=threshold,
            batch_size=batch_size,
            device_mode="cpu",
            extra_notices=[notice],
        )


def main():
    parser = argparse.ArgumentParser(description="Batch document trimming service")
    parser.add_argument("--input", "-i", type=str, help="Input JSON file path (or '-' for stdin)")
    parser.add_argument("--output", "-o", type=str, help="Output JSON file path (or '-' for stdout)")
    parser.add_argument(
        "--onnx_dir",
        type=str,
        default=None,
        help="Directory containing model.onnx (default: auto-resolved from EDSAN_TRIMMER_PATH, user cache, or repo artifacts)",
    )
    parser.add_argument("--threshold", type=float, default=0.50)
    args = parser.parse_args()

    # Pre-flight resolution of model directory
    try:
        resolved_dir = resolve_model_dir(args.onnx_dir)
    except FileNotFoundError as err:
        sys.stderr.write(str(err) + "\n")
        sys.exit(1)

    # Read input
    if not args.input or args.input == "-":
        payload = json.load(sys.stdin)
    else:
        with open(args.input, "r", encoding="utf-8") as f:
            payload = json.load(f)

    if not isinstance(payload, list):
        payload = [payload]

    # Process
    results = trim_batch(
        documents=payload,
        onnx_dir=resolved_dir,
        threshold=args.threshold,
    )

    # Output
    out_json = json.dumps(results, ensure_ascii=False, indent=2)
    if not args.output or args.output == "-":
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stdout.write(out_json)
    else:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(out_json)


if __name__ == "__main__":
    main()

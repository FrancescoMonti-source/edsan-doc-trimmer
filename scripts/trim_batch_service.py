#!/usr/bin/env python3
"""Batch document trimming service for redsan R integration.

Accepts JSON input from a file or stdin containing a list of {id, text, rectype},
runs batched DrBERT ONNX inference,
and outputs JSON with trimmed text and exact [start_char, end_char] intervals.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Ensure repo src/ is in sys.path if running directly from script
_REPO_SRC = Path(__file__).resolve().parent.parent / "src"
if _REPO_SRC.exists() and str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

from dataclasses import dataclass

import numpy as np
import onnxruntime as ort

WORKER_CONTRACT = "rectype-aware-v1"

try:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    HAS_TORCH_TRANSFORMERS = True
except ImportError:
    torch = None
    AutoModelForSequenceClassification = None
    AutoTokenizer = None
    HAS_TORCH_TRANSFORMERS = False

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


def _empty_result(doc_id: object, raw_text: str) -> dict:
    return {
        "id": doc_id,
        "trimmed_text": "",
        "is_bt": False,
        "raw_chars": len(raw_text),
        "trimmed_chars": 0,
        "reduction_pct": 100.0 if raw_text else 0.0,
        "preserved_intervals": [],
        "removed_intervals": [],
    }

def trim_batch(
    documents: list[dict],
    onnx_dir: str | Path | None = None,
    threshold: float = 0.50,
    apply_hybrid_rules: bool = False,
    batch_size: int = 128,
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
    safetensors_path = dir_path / "model.safetensors"
    use_cuda = (
        HAS_TORCH_TRANSFORMERS
        and torch is not None
        and torch.cuda.is_available()
        and safetensors_path.exists()
    )

    if use_cuda:
        device = torch.device("cuda")
        tokenizer = AutoTokenizer.from_pretrained(str(dir_path))
        model = AutoModelForSequenceClassification.from_pretrained(str(dir_path)).to(device)
        model.eval()
        rust_tok = False
    else:
        device = None
        model = None
        onnx_path = str(dir_path / "model.onnx")
        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = os.cpu_count() or 8
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(onnx_path, sess_opts, providers=["CPUExecutionProvider"])

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

                if use_cuda:
                    encodings = tokenizer(
                        targets,
                        contexts,
                        padding=True,
                        truncation=True,
                        max_length=128,
                        return_tensors="pt",
                    ).to(device)
                    with torch.inference_mode():
                        logits = model(**encodings).logits
                        probs = torch.softmax(logits, dim=-1).cpu().numpy()
                else:
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
                    logits = session.run(["logits"], ort_inputs)[0]
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
            "is_bt": False,
            "raw_chars": raw_len,
            "trimmed_chars": trim_len,
            "reduction_pct": round(reduction, 2),
            "preserved_intervals": preserved_intervals,
            "removed_intervals": removed_intervals,
        })

    return results


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
    parser.add_argument("--no_hybrid", action="store_true", help="Disable hybrid regex pre/post rules")
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
        apply_hybrid_rules=not args.no_hybrid,
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

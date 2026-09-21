#!/usr/bin/env python3
"""Run in-process ONNX inference on a hospital document to trim boilerplate."""

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
from transformers import AutoTokenizer

try:
    from redsan_doc_trimmer.dataset import (
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

        script_p = Path(__file__).resolve().parent
        if (script_p / "model.onnx").is_file():
            return script_p

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
            "Please set the EDSAN_TRIMMER_PATH environment variable or specify --onnx_dir.\n"
            "================================================================================"
        )



def trim_document(
    raw_text: str,
    onnx_dir: str | Path | None = None,
    threshold: float = 0.50,
    batch_size: int = 32,
    window_size: int = 2,
    max_length: int = 128,
    session: ort.InferenceSession | None = None,
    tokenizer: AutoTokenizer | None = None,
) -> dict:
    """Trims boilerplate from raw RECTXT document using in-process ONNX runtime.

    Returns the trimmed text along with exact [start_char, end_char] grounding coordinates.
    """
    resolved_dir = resolve_model_dir(onnx_dir)
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(str(resolved_dir))
    if session is None:
        onnx_path = str(resolved_dir / "model.onnx")
        session = ort.InferenceSession(onnx_path)

    spans = extract_lines_with_offsets(raw_text)
    samples = build_windowed_samples(spans, window_size=window_size)

    if not samples:
        return {
            "trimmed_text": "",
            "clinical_spans": [],
            "boilerplate_lines": [],
            "raw_char_count": len(raw_text),
            "trimmed_char_count": 0,
            "reduction_pct": 100.0,
        }

    line_is_bp = [False] * len(spans)

    for i in range(0, len(samples), batch_size):
        batch = samples[i : i + batch_size]
        target_texts = [s.target_text for s in batch]
        contexts = [f"{s.context_before} \n {s.context_after}".strip() for s in batch]

        encodings = tokenizer(
            target_texts,
            contexts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="np",
        )

        ort_inputs = {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
        }
        logits = session.run(["logits"], ort_inputs)[0]

        # Softmax
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

        for s, p in zip(batch, probs, strict=False):
            p_boilerplate = float(p[1])
            if p_boilerplate > threshold:
                line_is_bp[s.line_index] = True

    # Reconstruct preserved clinical spans with exact character coordinates
    clinical_spans = []
    clinical_chunks = []
    bp_line_indices = []

    for i, span in enumerate(spans):
        if line_is_bp[i]:
            bp_line_indices.append(i)
        else:
            clinical_spans.append({
                "line_index": i,
                "start_char": span.start_char,
                "end_char": span.end_char,
                "text": span.text,
            })
            clinical_chunks.append(span.text)

    trimmed_text = "\n".join(c for c in clinical_chunks if c.strip())
    raw_chars = len(raw_text)
    trimmed_chars = len(trimmed_text)
    reduction = ((raw_chars - trimmed_chars) / max(1, raw_chars)) * 100.0

    return {
        "trimmed_text": trimmed_text,
        "clinical_spans": clinical_spans,
        "boilerplate_line_count": len(bp_line_indices),
        "clinical_line_count": len(clinical_spans),
        "raw_char_count": raw_chars,
        "trimmed_char_count": trimmed_chars,
        "reduction_pct": reduction,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test ONNX document trimmer")
    parser.add_argument(
        "--onnx_dir",
        type=str,
        default=None,
        help="Directory containing model.onnx (default: auto-resolved from EDSAN_TRIMMER_PATH, user cache, or repo artifacts)",
    )
    parser.add_argument("--corpus", type=str, default="data/raw/corpus_sample.jsonl")
    parser.add_argument("--doc_id", type=str, default=None)
    args = parser.parse_args()

    # Pre-flight resolution of model directory
    try:
        resolved_onnx_dir = resolve_model_dir(args.onnx_dir)
    except FileNotFoundError as err:
        sys.stderr.write(str(err) + "\n")
        sys.exit(1)

    # Load a test document
    target_doc = None
    with open(args.corpus, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if args.doc_id is None or d.get("doc_id") == args.doc_id:
                target_doc = d
                break

    if not target_doc:
        print("No document found.")
        sys.exit(1)

    print("=" * 80)
    print(f"TESTING ONNX MODEL: {resolved_onnx_dir / 'model.onnx'}")
    print(f"DOCUMENT ID: {target_doc.get('doc_id')} (Type: {target_doc.get('rectype')})")
    print("=" * 80)

    result = trim_document(target_doc["rectxt"], onnx_dir=resolved_onnx_dir)

    print("\nRESULTS:")
    print(f"  Raw character length:     {result['raw_char_count']} chars")
    print(f"  Trimmed character length: {result['trimmed_char_count']} chars")
    print(f"  Prompt token reduction:   {result['reduction_pct']:.1f}%")
    print(f"  Boilerplate lines stripped: {result['boilerplate_line_count']}")
    print(f"  Clinical lines preserved:   {result['clinical_line_count']}")
    print("-" * 80)
    print("PREVIEW OF TRIMMED CLINICAL NARRATIVE:")
    print("-" * 80)
    print(result["trimmed_text"][:800] + ("..." if len(result["trimmed_text"]) > 800 else ""))
    print("=" * 80)


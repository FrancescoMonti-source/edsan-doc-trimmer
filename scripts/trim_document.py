#!/usr/bin/env python3
"""Run in-process ONNX inference on a hospital document to trim boilerplate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

from redsan_doc_trimmer.dataset import build_windowed_samples, extract_lines_with_offsets


def trim_document(
    raw_text: str,
    onnx_dir: str = "artifacts/active_learning/onnx_export",
    threshold: float = 0.50,
    batch_size: int = 32,
    window_size: int = 2,
    max_length: int = 256,
    session: ort.InferenceSession | None = None,
    tokenizer: AutoTokenizer | None = None,
) -> dict:
    """Trims boilerplate from raw RECTXT document using in-process ONNX runtime.

    Returns the trimmed text along with exact [start_char, end_char] grounding coordinates.
    """
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(onnx_dir)
    if session is None:
        onnx_path = str(Path(onnx_dir) / "model.onnx")
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
        "--onnx_dir", type=str, default="artifacts/active_learning/onnx_export"
    )
    parser.add_argument("--corpus", type=str, default="data/raw/corpus_sample.jsonl")
    parser.add_argument("--doc_id", type=str, default=None)
    args = parser.parse_args()

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
        exit(1)

    print("=" * 80)
    print(f"TESTING ONNX MODEL: {args.onnx_dir}/model.onnx")
    print(f"DOCUMENT ID: {target_doc.get('doc_id')} (Type: {target_doc.get('rectype')})")
    print("=" * 80)

    result = trim_document(target_doc["rectxt"], onnx_dir=args.onnx_dir)

    print(f"\nRESULTS:")
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


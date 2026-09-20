#!/usr/bin/env python3
"""Batch document trimming service for redsan R integration.

Accepts JSON input from a file or stdin containing a list of {id, text},
runs batched DrBERT ONNX inference (with hybrid pre/post rules),
and outputs JSON with trimmed text and exact [start_char, end_char] intervals.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

try:
    from tokenizers import Tokenizer
    USE_RUST_TOKENIZER = True
except ImportError:
    USE_RUST_TOKENIZER = False

from redsan_doc_trimmer.dataset import build_windowed_samples, extract_lines_with_offsets

# Hybrid Regex post-cleaners for known stubborn trailing metadata
POST_BP_PATTERNS = [
    re.compile(r"^\s*Copie\s+(adressée|transmise)\s+à\b.*$", re.IGNORECASE),
    re.compile(r"^\s*[A-Z]{2,3}\s*/\s*[A-Z]{2,3}\s*$"),  # Typist initials like MB /SR
    re.compile(r"^\s*\d+\s*/\s*\d+\s*$"),  # Lone page numbers like 3/8
]


def is_transport_voucher(text: str) -> bool:
    """Detects if a document is a transport voucher (BT)."""
    return bool(re.search(r"^\s*BT\b", text, re.IGNORECASE) or "FORMCHECKBOX" in text)


def trim_batch(
    documents: list[dict],
    onnx_dir: str = "artifacts/active_learning/onnx_export",
    threshold: float = 0.50,
    apply_hybrid_rules: bool = True,
    batch_size: int = 128,
) -> list[dict]:
    dir_path = Path(onnx_dir)
    safetensors_path = dir_path / "model.safetensors"
    use_cuda = torch.cuda.is_available() and safetensors_path.exists()

    if use_cuda:
        device = torch.device("cuda")
        tokenizer = AutoTokenizer.from_pretrained(onnx_dir)
        model = AutoModelForSequenceClassification.from_pretrained(onnx_dir).to(device)
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
        if USE_RUST_TOKENIZER and tok_json.exists():
            tokenizer = Tokenizer.from_file(str(tok_json))
            tokenizer.enable_padding(length=128, pad_id=1, pad_token="<pad>")
            tokenizer.enable_truncation(max_length=128)
            rust_tok = True
        else:
            tokenizer = AutoTokenizer.from_pretrained(onnx_dir)
            rust_tok = False

    results = []

    for doc in documents:
        doc_id = doc.get("id", "doc")
        raw_text = doc.get("text", "")

        # Check for empty or transport voucher
        if not raw_text or not raw_text.strip():
            results.append({
                "id": doc_id,
                "trimmed_text": "",
                "is_bt": False,
                "raw_chars": len(raw_text),
                "trimmed_chars": 0,
                "reduction_pct": 100.0 if raw_text else 0.0,
                "preserved_intervals": [],
                "removed_intervals": [],
            })
            continue

        if apply_hybrid_rules and is_transport_voucher(raw_text):
            # Flagged as BT: Entire document is administrative transport form
            results.append({
                "id": doc_id,
                "trimmed_text": "",
                "is_bt": True,
                "raw_chars": len(raw_text),
                "trimmed_chars": 0,
                "reduction_pct": 100.0,
                "preserved_intervals": [],
                "removed_intervals": [{"start": 1, "end": len(raw_text), "family": "transport_voucher"}],
            })
            continue

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

        # Hybrid post-cleaner for trailing stubborn lines
        if apply_hybrid_rules:
            for i, span in enumerate(spans):
                if not line_is_bp[i]:
                    for pat in POST_BP_PATTERNS:
                        if pat.search(span.text):
                            line_is_bp[i] = True
                            break

        # Reconstruct preserved and removed intervals (1-indexed for R!)
        preserved_intervals = []
        removed_intervals = []
        clinical_chunks = []

        for i, span in enumerate(spans):
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
    parser.add_argument("--onnx_dir", type=str, default="artifacts/active_learning/onnx_export")
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--no_hybrid", action="store_true", help="Disable hybrid regex pre/post rules")
    args = parser.parse_args()

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
        onnx_dir=args.onnx_dir,
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


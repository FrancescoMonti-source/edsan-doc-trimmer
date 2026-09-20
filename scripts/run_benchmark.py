#!/usr/bin/env python3
"""Run large-scale benchmark on unseen datasets (D0840 and denut.rds) including ORDON*."""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from redsan_doc_trimmer.dataset import build_windowed_samples, extract_lines_with_offsets


def run_benchmark(
    sample_file: str = "data/processed/benchmark_sample.jsonl",
    model_dir: str = "artifacts/active_learning/student_v2/best_model",
    max_docs: int = 500,
    batch_size: int = 64,
    threshold: float = 0.50,
    dataset_filter: str | None = None,
    output_json: str = "artifacts/benchmark_results.json",
):
    print("=" * 80)
    print("RED-SAN DOC-TRIMMER: LARGE-SCALE BENCHMARK ON UNSEEN DATASETS")
    print(f"Sample file: {sample_file}")
    print(f"Model path:  {model_dir}")
    print(f"Max docs:    {max_docs}")
    print(f"Batch size:  {batch_size}")
    print(f"Threshold:   {threshold}")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running inference on: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    print("\nLoading model & tokenizer...")
    t_load = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()
    print(f"Model loaded in {time.time() - t_load:.2f}s")

    # Load documents
    docs = []
    with open(sample_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            if dataset_filter and d.get("dataset") != dataset_filter:
                continue
            docs.append(d)
            if len(docs) >= max_docs:
                break

    print(f"Loaded {len(docs)} documents for evaluation (Filter: {dataset_filter or 'All'}).")

    results = []
    type_stats = defaultdict(list)
    dataset_stats = defaultdict(list)
    ordon_examples = []

    t_start = time.time()
    total_raw_chars = 0
    total_trimmed_chars = 0
    total_lines = 0
    total_bp_lines = 0

    with torch.no_grad():
        for doc_idx, doc in enumerate(docs):
            doc_id = doc.get("doc_id", f"doc_{doc_idx}")
            rectype = doc.get("rectype", "UNKNOWN")
            dataset = doc.get("dataset", "UNKNOWN")
            raw_text = doc.get("rectxt", "")

            spans = extract_lines_with_offsets(raw_text)
            samples = build_windowed_samples(spans, window_size=2)

            if not samples:
                continue

            line_is_bp = [False] * len(spans)

            # Batched inference over lines
            for i in range(0, len(samples), batch_size):
                batch = samples[i : i + batch_size]
                targets = [s.target_text for s in batch]
                contexts = [f"{s.context_before} \n {s.context_after}".strip() for s in batch]

                enc = tokenizer(
                    targets,
                    contexts,
                    padding=True,
                    truncation=True,
                    max_length=256,
                    return_tensors="pt",
                ).to(device)

                logits = model(**enc).logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()

                for s, p in zip(batch, probs, strict=False):
                    p_bp = float(p[1])
                    if p_bp > threshold:
                        line_is_bp[s.line_index] = True

            # Reconstruct
            clinical_lines = []
            bp_lines = []
            for i, span in enumerate(spans):
                if line_is_bp[i]:
                    bp_lines.append(span.text)
                else:
                    clinical_lines.append(span.text)

            trimmed_text = "\n".join(c for c in clinical_lines if c.strip())
            raw_chars = len(raw_text)
            trimmed_chars = len(trimmed_text)
            reduction_pct = ((raw_chars - trimmed_chars) / max(1, raw_chars)) * 100.0

            total_raw_chars += raw_chars
            total_trimmed_chars += trimmed_chars
            total_lines += len(spans)
            total_bp_lines += len(bp_lines)

            doc_res = {
                "doc_id": doc_id,
                "dataset": dataset,
                "rectype": rectype,
                "raw_chars": raw_chars,
                "trimmed_chars": trimmed_chars,
                "reduction_pct": reduction_pct,
                "total_lines": len(spans),
                "clinical_lines": len(clinical_lines),
                "bp_lines": len(bp_lines),
            }
            results.append(doc_res)
            type_stats[rectype].append(reduction_pct)
            dataset_stats[dataset].append(reduction_pct)

            # Keep sample of ORDON* for qualitative inspection
            if "ORDON" in rectype.upper() and len(ordon_examples) < 5:
                ordon_examples.append({
                    "doc_id": doc_id,
                    "rectype": rectype,
                    "reduction_pct": reduction_pct,
                    "raw_preview": raw_text[:500],
                    "trimmed_preview": trimmed_text[:500],
                    "stripped_header": "\n".join(bp_lines[:6]),
                    "stripped_footer": "\n".join(bp_lines[-6:]),
                })

            if (doc_idx + 1) % 100 == 0 or (doc_idx + 1) == len(docs):
                elapsed = time.time() - t_start
                rate = (doc_idx + 1) / elapsed
                print(f"[{doc_idx + 1:4d}/{len(docs)}] ({rate:4.1f} docs/s) - Total reduction so far: {((total_raw_chars - total_trimmed_chars)/max(1, total_raw_chars))*100.1:.1f}%")

    total_time = time.time() - t_start
    overall_reduction = ((total_raw_chars - total_trimmed_chars) / max(1, total_raw_chars)) * 100.0

    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"Documents evaluated:        {len(results)}")
    print(f"Total processing time:      {total_time:.2f} seconds ({len(results)/total_time:.1f} docs/sec)")
    print(f"Total raw volume:           {total_raw_chars:,} characters (~{total_raw_chars//4:,} prompt tokens)")
    print(f"Total trimmed volume:       {total_trimmed_chars:,} characters (~{total_trimmed_chars//4:,} prompt tokens)")
    print(f"Overall token reduction:    {overall_reduction:.2f}%")
    print(f"Lines stripped (BP):        {total_bp_lines:,} / {total_lines:,} ({total_bp_lines/max(1, total_lines)*100:.1f}%)")

    print("\n" + "-" * 80)
    print("PER-DATASET BREAKDOWN")
    print("-" * 80)
    for ds, pcts in dataset_stats.items():
        print(f"  {ds:10s} (N={len(pcts):4d}) -> Mean reduction: {np.mean(pcts):5.1f}% | Median: {np.median(pcts):5.1f}% | Min: {np.min(pcts):4.1f}% | Max: {np.max(pcts):4.1f}%")

    print("\n" + "-" * 80)
    print("TOP DOCUMENT TYPES BREAKDOWN (by frequency)")
    print("-" * 80)
    sorted_types = sorted(type_stats.items(), key=lambda x: len(x[1]), reverse=True)
    print(f"{'RECTYPE':<14} {'Count':<7} {'Mean Reduc':<12} {'Median':<10} {'Min':<8} {'Max':<8}")
    print("-" * 65)
    for rtype, pcts in sorted_types[:20]:
        print(f"{rtype:<14} {len(pcts):<7d} {np.mean(pcts):<11.1f}% {np.median(pcts):<9.1f}% {np.min(pcts):<7.1f}% {np.max(pcts):<7.1f}%")

    # Save results
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    summary_out = {
        "total_docs": len(results),
        "total_time_sec": total_time,
        "docs_per_sec": len(results) / total_time,
        "total_raw_chars": total_raw_chars,
        "total_trimmed_chars": total_trimmed_chars,
        "overall_reduction_pct": overall_reduction,
        "dataset_stats": {
            ds: {"count": len(p), "mean": float(np.mean(p)), "median": float(np.median(p))}
            for ds, p in dataset_stats.items()
        },
        "type_stats": {
            t: {"count": len(p), "mean": float(np.mean(p)), "median": float(np.median(p))}
            for t, p in sorted_types
        },
        "ordon_examples": ordon_examples,
        "extreme_highest_trim": sorted(results, key=lambda x: x["reduction_pct"], reverse=True)[:5],
        "extreme_lowest_trim": sorted(results, key=lambda x: x["reduction_pct"])[:5],
    }
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(summary_out, f, indent=2)

    print(f"\nDetailed benchmark artifact written to: {output_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_file", default="data/processed/benchmark_sample.jsonl")
    parser.add_argument("--model_dir", default="artifacts/active_learning/student_v2/best_model")
    parser.add_argument("--max_docs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--dataset_filter", type=str, default=None)
    parser.add_argument("--output_json", default="artifacts/benchmark_results.json")
    args = parser.parse_args()

    run_benchmark(
        sample_file=args.sample_file,
        model_dir=args.model_dir,
        max_docs=args.max_docs,
        batch_size=args.batch_size,
        threshold=args.threshold,
        dataset_filter=args.dataset_filter,
        output_json=args.output_json,
    )


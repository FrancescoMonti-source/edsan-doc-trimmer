#!/usr/bin/env python3
"""End-to-end Active Learning and Hard-Negative Mining pipeline for redsan-doc-trimmer."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from redsan_doc_trimmer.annotate import DEFAULT_TEACHER_MODEL, annotate_corpus_file
from redsan_doc_trimmer.dataset import load_raw_documents
from redsan_doc_trimmer.export_onnx import export_to_onnx
from redsan_doc_trimmer.mining import HardCaseResult, scan_mining_pool
from redsan_doc_trimmer.train import train_from_annotated_jsonl

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("active_learning_pipeline")


def run_pipeline(
    corpus_jsonl: str = "data/raw/corpus_sample.jsonl",
    seed_size: int = 400,
    mine_top_k: int = 200,
    teacher_model: str = DEFAULT_TEACHER_MODEL,
    output_dir: str = "./artifacts/active_learning",
    epochs_v1: int = 3,
    epochs_v2: int = 2,
    lr_v1: float = 2e-5,
    lr_v2: float = 5e-6,
    use_mock: bool = False,
):
    """Executes the complete active learning cycle."""
    out_base = Path(output_dir)
    out_base.mkdir(parents=True, exist_ok=True)
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load full raw corpus sample
    logger.info("Loading documents from %s...", corpus_jsonl)
    all_docs = load_raw_documents(corpus_jsonl)
    total_docs = len(all_docs)
    logger.info("Loaded %d documents.", total_docs)

    if total_docs < seed_size:
        raise ValueError(
            f"Corpus only has {total_docs} docs, less than requested seed size {seed_size}"
        )

    seed_docs = all_docs[:seed_size]
    pool_docs = all_docs[seed_size:]

    seed_raw_path = processed_dir / "seed_pool_raw.jsonl"
    with open(seed_raw_path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(doc, ensure_ascii=False) + "\n" for doc in seed_docs)

    # 2. Phase 1: Annotate Seed Documents
    seed_annotated_path = processed_dir / f"seed_annotated_{seed_size}.jsonl"
    logger.info("=== Phase 1: Annotating Seed Dataset (%d docs) ===", len(seed_docs))
    annotate_corpus_file(
        input_jsonl_path=str(seed_raw_path),
        output_jsonl_path=str(seed_annotated_path),
        model=teacher_model,
        use_mock=use_mock,
        resume=True,
    )
    logger.info("Seed annotation complete: %s", seed_annotated_path)

    # 3. Phase 2: Train Student v1
    v1_ckpt = out_base / "student_v1"
    logger.info("=== Phase 2: Training Student v1 on Seed Set ===")
    train_from_annotated_jsonl(
        jsonl_path=str(seed_annotated_path),
        output_dir=str(v1_ckpt),
        num_train_epochs=epochs_v1,
        learning_rate=lr_v1,
        clinical_penalty_weight=10.0,
    )
    logger.info("Student v1 trained and saved at: %s", v1_ckpt / "best_model")

    # 4. Phase 3: Scan Mining Pool with Student v1
    logger.info(
        "=== Phase 3: Scanning Mining Pool (%d docs) with Student v1 ===",
        len(pool_docs),
    )
    v1_model_path = str(v1_ckpt / "best_model")
    tokenizer = AutoTokenizer.from_pretrained(v1_model_path)
    model = AutoModelForSequenceClassification.from_pretrained(v1_model_path)

    hard_cases: list[HardCaseResult] = scan_mining_pool(
        model=model,
        tokenizer=tokenizer,
        candidate_documents=pool_docs,
        top_k=mine_top_k,
    )
    logger.info(
        "Mined top %d hard cases (ranked by uncertainty + anchor safety violations).",
        len(hard_cases),
    )

    # Save mined hard docs
    hard_ids = {h.doc_id for h in hard_cases}
    mined_raw_path = processed_dir / f"mined_hard_cases_raw_{len(hard_cases)}.jsonl"
    with open(mined_raw_path, "w", encoding="utf-8") as f:
        for doc in pool_docs:
            if doc["doc_id"] in hard_ids:
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    # 5. Phase 4: Targeted Teacher Adjudication
    mined_annotated_path = (
        processed_dir / f"mined_hard_cases_annotated_{len(hard_cases)}.jsonl"
    )
    logger.info("=== Phase 4: Teacher Adjudication on Mined Hard Cases ===")
    annotate_corpus_file(
        input_jsonl_path=str(mined_raw_path),
        output_jsonl_path=str(mined_annotated_path),
        model=teacher_model,
        use_mock=use_mock,
        resume=True,
    )
    logger.info("Teacher adjudication complete: %s", mined_annotated_path)

    # 6. Phase 5: Merge & Fine-Tune Student v2
    augmented_path = processed_dir / "augmented_dataset_v2.jsonl"
    logger.info("=== Phase 5: Combining Datasets & Training Student v2 ===")
    with open(augmented_path, "w", encoding="utf-8") as out_f:
        for p in [seed_annotated_path, mined_annotated_path]:
            with open(p, "r", encoding="utf-8") as in_f:
                out_f.writelines(in_f)

    v2_ckpt = out_base / "student_v2"
    train_from_annotated_jsonl(
        jsonl_path=str(augmented_path),
        output_dir=str(v2_ckpt),
        model_name=v1_model_path,  # Continue from Student v1 weights!
        num_train_epochs=epochs_v2,
        learning_rate=lr_v2,  # Lower LR to refine edge cases
        clinical_penalty_weight=12.0,  # Even higher clinical safety weight
    )
    logger.info("Student v2 trained and saved at: %s", v2_ckpt / "best_model")

    # 7. Phase 6: ONNX Export
    onnx_dir = out_base / "onnx_export"
    logger.info("=== Phase 6: Exporting Final Student Model to ONNX ===")
    export_to_onnx(
        model_path=str(v2_ckpt / "best_model"),
        output_dir=str(onnx_dir),
        onnx_filename="model.onnx",
    )
    logger.info("Pipeline Complete! Deployable ONNX package available at: %s", onnx_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Active learning self-correction pipeline"
    )
    parser.add_argument("--corpus", type=str, default="data/raw/corpus_sample.jsonl")
    parser.add_argument("--seed_size", type=int, default=400)
    parser.add_argument("--mine_top_k", type=int, default=200)
    parser.add_argument("--teacher_model", type=str, default=DEFAULT_TEACHER_MODEL)
    parser.add_argument("--output_dir", type=str, default="./artifacts/active_learning")
    parser.add_argument(
        "--mock", action="store_true", help="Use mock heuristic teacher for testing"
    )
    args = parser.parse_args()

    # Automatically enable mock if OPENAI_API_KEY is missing
    is_mock = args.mock or ("OPENAI_API_KEY" not in os.environ)
    if is_mock and not args.mock:
        logger.info(
            "OPENAI_API_KEY not found in environment; using mock teacher for dry-run."
        )

    run_pipeline(
        corpus_jsonl=args.corpus,
        seed_size=args.seed_size,
        mine_top_k=args.mine_top_k,
        teacher_model=args.teacher_model,
        output_dir=args.output_dir,
        use_mock=is_mock,
    )

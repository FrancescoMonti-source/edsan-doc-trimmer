#!/usr/bin/env python3
"""Dedicated runner to train Student v2 from curated annotations and export ONNX."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from redsan_doc_trimmer.export_onnx import export_to_onnx
from redsan_doc_trimmer.train import train_from_annotated_jsonl

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("train_student_v2")


def train_v2_from_curated(
    seed_jsonl: str = "data/processed/seed_annotated_1000.jsonl",
    mined_jsonl: str = "data/processed/mined_hard_cases_annotated_500.jsonl",
    v1_model_dir: str = "artifacts/active_learning/student_v1/best_model",
    output_dir: str = "./artifacts/active_learning",
    epochs: int = 2,
    lr: float = 5e-6,
    clinical_penalty: float = 12.0,
):
    out_base = Path(output_dir)
    out_base.mkdir(parents=True, exist_ok=True)
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    # 1. Merge seed and mined datasets
    augmented_path = processed_dir / "augmented_dataset_v2.jsonl"
    logger.info("Combining seed (%s) and mined cases (%s)...", seed_jsonl, mined_jsonl)

    with open(augmented_path, "w", encoding="utf-8") as out_f:
        for p in [seed_jsonl, mined_jsonl]:
            p_path = Path(p)
            if not p_path.exists():
                raise FileNotFoundError(f"Required dataset file not found: {p}")
            with open(p_path, "r", encoding="utf-8") as in_f:
                out_f.writelines(in_f)

    logger.info("Augmented dataset ready at: %s", augmented_path)

    # 2. Fine-tune Student v2 starting from Student v1 weights
    v2_ckpt = out_base / "student_v2"
    logger.info("=== Fine-Tuning Student v2 (Continuing from Student v1) ===")
    train_from_annotated_jsonl(
        jsonl_path=str(augmented_path),
        output_dir=str(v2_ckpt),
        model_name=v1_model_dir,
        num_train_epochs=epochs,
        learning_rate=lr,
        clinical_penalty_weight=clinical_penalty,
    )
    logger.info("Student v2 successfully trained: %s", v2_ckpt / "best_model")

    # 3. Export to ONNX
    onnx_dir = out_base / "onnx_export"
    logger.info("=== Exporting Student v2 to ONNX ===")
    export_to_onnx(
        model_path=str(v2_ckpt / "best_model"),
        output_dir=str(onnx_dir),
        onnx_filename="model.onnx",
    )
    logger.info("ONNX artifact exported successfully to: %s", onnx_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Student v2 from curated human/teacher datasets"
    )
    parser.add_argument(
        "--seed",
        type=str,
        default="data/processed/seed_annotated_1000.jsonl",
        help="Path to seed annotated dataset",
    )
    parser.add_argument(
        "--mined",
        type=str,
        default="data/processed/mined_hard_cases_annotated_500.jsonl",
        help="Path to mined/curated edge cases dataset",
    )
    parser.add_argument(
        "--v1_model",
        type=str,
        default="artifacts/active_learning/student_v1/best_model",
        help="Path to Student v1 checkpoint directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./artifacts/active_learning",
        help="Output base directory",
    )
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--clinical_penalty", type=float, default=12.0)
    args = parser.parse_args()

    train_v2_from_curated(
        seed_jsonl=args.seed,
        mined_jsonl=args.mined,
        v1_model_dir=args.v1_model,
        output_dir=args.output_dir,
        epochs=args.epochs,
        lr=args.lr,
        clinical_penalty=args.clinical_penalty,
    )


#!/usr/bin/env python3
"""Dedicated runner to train Student v3 from enriched dataset and export ONNX."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from redsan_doc_trimmer.export_onnx import export_to_onnx
from redsan_doc_trimmer.train import train_from_annotated_jsonl

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("train_student_v3")


def train_v3(
    dataset_jsonl: str = "data/processed/augmented_dataset_v3.jsonl",
    v2_model_dir: str = "artifacts/active_learning/student_v2/best_model",
    output_dir: str = "./artifacts/active_learning",
    epochs: int = 2,
    batch_size: int = 32,
    lr: float = 5e-6,
    clinical_penalty: float = 12.0,
):
    out_base = Path(output_dir)
    out_base.mkdir(parents=True, exist_ok=True)

    dataset_path = Path(dataset_jsonl)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    # 1. Fine-tune Student v3 starting from Student v2 weights
    v3_ckpt = out_base / "student_v3"
    logger.info("=== Fine-Tuning Student v3 (Continuing from Student v2) ===")
    logger.info("Dataset: %s", dataset_path)
    logger.info("Base Model: %s", v2_model_dir)
    logger.info("Epochs: %d, Batch Size: %d, LR: %g, Clinical Penalty: %g", epochs, batch_size, lr, clinical_penalty)

    train_from_annotated_jsonl(
        jsonl_path=str(dataset_path),
        output_dir=str(v3_ckpt),
        model_name=v2_model_dir,
        num_train_epochs=epochs,
        batch_size=batch_size,
        learning_rate=lr,
        clinical_penalty_weight=clinical_penalty,
    )
    best_model_path = v3_ckpt / "best_model"
    logger.info("Student v3 successfully trained: %s", best_model_path)

    # 2. Export to ONNX
    onnx_dir = out_base / "onnx_export"
    logger.info("=== Exporting Student v3 to ONNX ===")
    export_to_onnx(
        model_path=str(best_model_path),
        output_dir=str(onnx_dir),
        onnx_filename="model.onnx",
    )
    logger.info("ONNX artifact exported successfully to: %s", onnx_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Student v3 DrBERT with transport voucher and metadata learning"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/processed/augmented_dataset_v3.jsonl",
        help="Path to augmented v3 dataset",
    )
    parser.add_argument(
        "--v2_model",
        type=str,
        default="artifacts/active_learning/student_v2/best_model",
        help="Path to Student v2 checkpoint directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./artifacts/active_learning",
        help="Output base directory",
    )
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--clinical_penalty", type=float, default=12.0)
    args = parser.parse_args()

    train_v3(
        dataset_jsonl=args.dataset,
        v2_model_dir=args.v2_model,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        clinical_penalty=args.clinical_penalty,
    )

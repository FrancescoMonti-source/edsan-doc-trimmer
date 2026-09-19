"""Training pipeline for fine-tuning DrBERT with asymmetric clinical loss."""

from __future__ import annotations

import argparse
import os
import random

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
)
from torch import nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from redsan_doc_trimmer.dataset import WindowedLineSample, load_annotated_samples

# Default pre-trained French biomedical encoder
DEFAULT_MODEL_NAME = "DrBERT/DrBERT-7GB"


class AsymmetricTrainer(Trainer):
    """Custom Trainer implementing asymmetric class-weighted loss to protect clinical content.

    A false positive (predicting boilerplate on clinical text) has a much higher penalty
    than a false negative (leaving boilerplate in the prompt).
    """

    def __init__(self, *args, clinical_weight: float = 10.0, **kwargs):
        super().__init__(*args, **kwargs)
        # Class 0: Clinical, Class 1: Boilerplate
        # Weight for class 0 is increased relative to class 1
        weights = torch.tensor([clinical_weight, 1.0])
        self.loss_fct = nn.CrossEntropyLoss(weight=weights)

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        if self.loss_fct.weight.device != logits.device:
            self.loss_fct.weight = self.loss_fct.weight.to(logits.device)
        loss = self.loss_fct(
            logits.view(-1, self.model.config.num_labels), labels.view(-1)
        )
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred) -> dict[str, float]:
    """Computes evaluation metrics focusing on clinical recall and safety."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    # Class 0: Clinical, Class 1: Boilerplate
    clinical_recall = recall_score(labels, preds, pos_label=0, zero_division=0)
    clinical_prec = precision_score(labels, preds, pos_label=0, zero_division=0)
    bp_recall = recall_score(labels, preds, pos_label=1, zero_division=0)
    bp_prec = precision_score(labels, preds, pos_label=1, zero_division=0)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)

    return {
        "clinical_recall": float(clinical_recall),
        "clinical_precision": float(clinical_prec),
        "boilerplate_recall": float(bp_recall),
        "boilerplate_precision": float(bp_prec),
        "macro_f1": float(macro_f1),
    }


def prepare_hf_dataset(
    samples: list[WindowedLineSample], tokenizer, max_length: int = 256
) -> Dataset:
    """Prepares Hugging Face Dataset with target line + context encoding."""
    target_texts = [s.target_text for s in samples]
    contexts = [f"{s.context_before} \n {s.context_after}".strip() for s in samples]
    labels = [1 if s.is_boilerplate else 0 for s in samples]

    encodings = tokenizer(
        target_texts,
        contexts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )
    encodings["labels"] = labels
    return Dataset.from_dict(encodings)


def run_training(
    train_samples: list[WindowedLineSample],
    eval_samples: list[WindowedLineSample],
    output_dir: str = "./checkpoints/drbert-trimmer",
    model_name: str = DEFAULT_MODEL_NAME,
    num_train_epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    clinical_penalty_weight: float = 10.0,
):
    """Executes DrBERT fine-tuning."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    train_dataset = prepare_hf_dataset(train_samples, tokenizer)
    eval_dataset = prepare_hf_dataset(eval_samples, tokenizer)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir="./runs",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="clinical_recall",
        greater_is_better=True,
    )

    trainer = AsymmetricTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        clinical_weight=clinical_penalty_weight,
    )

    trainer.train()
    trainer.save_model(os.path.join(output_dir, "best_model"))
    tokenizer.save_pretrained(os.path.join(output_dir, "best_model"))


def train_from_annotated_jsonl(
    jsonl_path: str,
    output_dir: str = "./checkpoints/drbert-trimmer",
    model_name: str = DEFAULT_MODEL_NAME,
    eval_split: float = 0.2,
    seed: int = 42,
    **training_kwargs,
):
    """Loads annotated samples from JSONL and trains DrBERT."""
    all_samples = load_annotated_samples(jsonl_path)
    random.seed(seed)
    random.shuffle(all_samples)

    split_idx = int(len(all_samples) * (1 - eval_split))
    train_samples = all_samples[:split_idx]
    eval_samples = all_samples[split_idx:]

    print(f"Loaded {len(all_samples)} total line samples.")
    print(f"Train samples: {len(train_samples)} | Eval samples: {len(eval_samples)}")

    run_training(
        train_samples=train_samples,
        eval_samples=eval_samples,
        output_dir=output_dir,
        model_name=model_name,
        **training_kwargs,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train DrBERT model with asymmetric clinical loss"
    )
    parser.add_argument(
        "--data", type=str, required=True, help="Path to annotated JSONL dataset"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./checkpoints/drbert-trimmer"
    )
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--clinical_weight", type=float, default=10.0)
    args = parser.parse_args()

    train_from_annotated_jsonl(
        jsonl_path=args.data,
        output_dir=args.output_dir,
        model_name=args.model_name,
        num_train_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        clinical_penalty_weight=args.clinical_weight,
    )

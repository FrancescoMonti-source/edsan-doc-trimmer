"""Training pipeline for fine-tuning DrBERT with asymmetric clinical loss."""

from __future__ import annotations

import os
from typing import Dict, Any, List
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset

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

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        if self.loss_fct.weight.device != logits.device:
            self.loss_fct.weight = self.loss_fct.weight.to(logits.device)
        loss = self.loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def prepare_hf_dataset(samples: List[Any], tokenizer, max_length: int = 256) -> Dataset:
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
    train_samples: List[Any],
    eval_samples: List[Any],
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
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir="./runs",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
    )

    trainer = AsymmetricTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        clinical_weight=clinical_penalty_weight,
    )

    trainer.train()
    trainer.save_model(os.path.join(output_dir, "best_model"))
    tokenizer.save_pretrained(os.path.join(output_dir, "best_model"))

"""Active learning and hard-negative mining module for redsan-doc-trimmer."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F

from redsan_doc_trimmer.dataset import (
    DocumentSpan,
    build_windowed_samples,
    extract_lines_with_offsets,
)

logger = logging.getLogger(__name__)


@dataclass
class LinePrediction:
    """Prediction result for a single document line."""

    line_index: int
    text: str
    prob_clinical: float
    prob_boilerplate: float
    is_uncertain: bool
    is_anchor_violation: bool = False
    violation_details: str | None = None


@dataclass
class HardCaseResult:
    """Summary of mined hardness metrics for an individual document."""

    doc_id: str
    rectype: str | None
    total_lines: int
    uncertain_line_count: int
    anchor_violation_count: int
    hardness_score: float
    line_predictions: list[LinePrediction] = field(default_factory=list)
    anchor_violations: list[str] = field(default_factory=list)


def compute_uncertainty_score(
    prob_boilerplate: float,
    lower_bound: float = 0.35,
    upper_bound: float = 0.65,
) -> tuple[bool, float]:
    """Calculates whether a prediction falls into the borderline uncertainty window.

    Returns (is_uncertain, margin_to_center), where 0.0 is maximally uncertain (p=0.50).
    """
    is_uncertain = lower_bound <= prob_boilerplate <= upper_bound
    margin = abs(prob_boilerplate - 0.50)
    return is_uncertain, margin


def predict_document_lines(
    model: Any,
    tokenizer: Any,
    rectxt: str,
    doc_id: str = "doc",
    window_size: int = 2,
    batch_size: int = 32,
    device: torch.device | None = None,
    max_length: int = 256,
) -> tuple[list[DocumentSpan], list[LinePrediction]]:
    """Runs inference on a document's lines with surrounding context."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    spans = extract_lines_with_offsets(rectxt)
    samples = build_windowed_samples(spans, window_size=window_size, doc_id=doc_id)

    line_predictions: list[LinePrediction] = []

    for i in range(0, len(samples), batch_size):
        batch_samples = samples[i : i + batch_size]
        target_texts = [s.target_text for s in batch_samples]
        contexts = [
            f"{s.context_before} \n {s.context_after}".strip() for s in batch_samples
        ]

        encodings = tokenizer(
            target_texts,
            contexts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encodings = {k: v.to(device) for k, v in encodings.items()}

        with torch.no_grad():
            outputs = model(**encodings)
            probs = F.softmax(outputs.logits, dim=-1).cpu().numpy()

        for sample, prob in zip(batch_samples, probs, strict=False):
            p_clinical = float(prob[0])
            p_boilerplate = float(prob[1])
            is_uncertain, _ = compute_uncertainty_score(p_boilerplate)

            line_predictions.append(
                LinePrediction(
                    line_index=sample.line_index,
                    text=sample.target_text,
                    prob_clinical=p_clinical,
                    prob_boilerplate=p_boilerplate,
                    is_uncertain=is_uncertain,
                )
            )

    return spans, line_predictions


def check_anchor_violations(
    predictions: list[LinePrediction],
    clinical_anchors: dict[str, str],
    boilerplate_threshold: float = 0.50,
) -> list[str]:
    """Identifies false positive predictions where the model flagged verified clinical text as boilerplate."""
    violations: list[str] = []
    if not clinical_anchors:
        return violations

    for pred in predictions:
        if pred.prob_boilerplate > boilerplate_threshold:
            line_str = pred.text.strip()
            if len(line_str) < 10:
                continue
            for anchor_name, anchor_content in clinical_anchors.items():
                if isinstance(anchor_content, str) and line_str in anchor_content:
                    pred.is_anchor_violation = True
                    msg = (
                        f"Line [{pred.line_index}] flagged as boilerplate (p={pred.prob_boilerplate:.2f}) "
                        f"but belongs to clinical section '{anchor_name}': '{line_str[:50]}...'"
                    )
                    pred.violation_details = msg
                    violations.append(msg)
                    break

    return violations


def score_document_hardness(
    doc_id: str,
    rectype: str | None,
    predictions: list[LinePrediction],
    clinical_anchors: dict[str, str] | None = None,
    uncertainty_weight: float = 1.0,
    violation_weight: float = 5.0,
) -> HardCaseResult:
    """Computes an overall hardness score for a document.

    Anchor violations are heavily weighted (5x) because false positives on clinical content
    represent catastrophic failures under the Cardinal Rule.
    """
    violations = check_anchor_violations(predictions, clinical_anchors or {})

    uncertain_count = sum(1 for p in predictions if p.is_uncertain)
    violation_count = len(violations)

    # Score: combined weighted measure of uncertainty and safety violations
    # Normalized by document line count to avoid biasing purely towards ultra-long documents
    norm_factor = max(10, len(predictions))
    hardness_score = (
        (uncertain_count * uncertainty_weight) + (violation_count * violation_weight)
    ) / (norm_factor**0.5)

    return HardCaseResult(
        doc_id=doc_id,
        rectype=rectype,
        total_lines=len(predictions),
        uncertain_line_count=uncertain_count,
        anchor_violation_count=violation_count,
        hardness_score=float(hardness_score),
        line_predictions=predictions,
        anchor_violations=violations,
    )


def scan_mining_pool(
    model: Any,
    tokenizer: Any,
    candidate_documents: list[dict[str, Any]],
    top_k: int = 200,
    batch_size: int = 32,
    device: torch.device | None = None,
) -> list[HardCaseResult]:
    """Scans an unlabeled pool of documents, ranking them by hardness score."""
    results: list[HardCaseResult] = []

    for doc in candidate_documents:
        doc_id = doc.get("doc_id", "unknown")
        rectxt = doc.get("rectxt", "")
        rectype = doc.get("rectype")
        clinical_anchors = doc.get("clinical_anchors", {})

        if not rectxt.strip():
            continue

        _, predictions = predict_document_lines(
            model=model,
            tokenizer=tokenizer,
            rectxt=rectxt,
            doc_id=doc_id,
            batch_size=batch_size,
            device=device,
        )

        hard_case = score_document_hardness(
            doc_id=doc_id,
            rectype=rectype,
            predictions=predictions,
            clinical_anchors=clinical_anchors,
        )
        results.append(hard_case)

    # Sort descending by hardness score
    results.sort(key=lambda r: r.hardness_score, reverse=True)
    return results[:top_k]

"""Tests for transport vouchers and discharge letters under pure model trimming.

These cases assert that:
1. Every non-empty document continues to ordinary model inference.
2. The worker has no deterministic whole-document transport shortcut.
3. Coordinates and grounding guarantees are maintained for model output.
"""

from __future__ import annotations

import inspect
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("onnxruntime")
np = pytest.importorskip("numpy")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import trim_batch_service as worker
from trim_batch_service import trim_batch


def test_worker_exposes_only_model_inference_controls():
    assert "apply_hybrid_rules" not in inspect.signature(trim_batch).parameters

    worker_script = Path(worker.__file__).resolve()
    help_result = subprocess.run(
        [sys.executable, str(worker_script), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--no_hybrid" not in help_result.stdout


# A real bon de transport. All 495 in the corpus are RECTYPE ORDON7.
VOUCHER = """BT - BON DE TRANSPORT

Patient : [FIRSTNAME] [LASTNAME]
 FORMCHECKBOX  Ambulance
 FORMCHECKBOX  VSL
 FORMCHECKBOX  Taxi conventionne
Motif : retour a domicile
"""

# A lettre de liaison. Every one carries the provisoire/definitive tick-boxes,
# and 240 of them were deleted for it under the old deterministic rule.
# Its RECTYPE is CRH2AB. The weight and the conclusion are what redsan-coding reads.
DISCHARGE_LETTER = """Centre Hospitalier Universitaire de Rouen
Service de Medecine Geriatrique

LETTRE DE LIAISON A LA SORTIE DU PATIENT :

 FORMCHECKBOX  Lettre de Liaison definitive, remise en main propre au patient.

Date d'entree : 07/05/2026
Date de sortie : 15/05/2026
Poids : 72.5 Kg
Taille : 1.68 m

Motif d'hospitalisation : sepsis sur pneumopathie communautaire.
Conclusion : denutrition severe dans un contexte d'agression aigue.
"""


@pytest.fixture(autouse=True)
def ordinary_model_runtime(monkeypatch, tmp_path):
    """Use an all-clinical model so routing tests remain artifact-independent."""

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "model.onnx").write_bytes(b"test model")

    class FakeTokenizer:
        @classmethod
        def from_pretrained(cls, _model_dir):
            return cls()

        def __call__(self, targets, _contexts, **_kwargs):
            count = len(targets)
            return {
                "input_ids": np.zeros((count, 2), dtype=np.int64),
                "attention_mask": np.ones((count, 2), dtype=np.int64),
            }

    class FakeSession:
        def __init__(self, *_args, **_kwargs):
            pass

        def run(self, _outputs, inputs):
            count = inputs["input_ids"].shape[0]
            return [np.tile(np.array([[10.0, 0.0]]), (count, 1))]

    monkeypatch.setattr(worker, "resolve_model_dir", lambda _candidate=None: model_dir)
    monkeypatch.setattr(worker, "USE_RUST_TOKENIZER", False)
    monkeypatch.setattr(worker, "AutoTokenizer", FakeTokenizer)
    monkeypatch.setattr(worker.ort, "InferenceSession", FakeSession)


def test_discharge_letter_preserves_clinical_facts():
    """Discharge letters must never be blanked and must keep all clinical facts verbatim."""
    docs = [{"id": "letter_1", "text": DISCHARGE_LETTER}]
    results = trim_batch(docs)
    assert len(results) == 1
    res = results[0]

    # Must preserve essential clinical anchors
    trimmed = res["trimmed_text"]
    assert "Poids : 72.5 Kg" in trimmed
    assert "Taille : 1.68 m" in trimmed
    assert "Motif d'hospitalisation : sepsis sur pneumopathie communautaire." in trimmed
    assert (
        "Conclusion : denutrition severe dans un contexte d'agression aigue." in trimmed
    )

    # Document must not be zeroed out
    assert res["trimmed_chars"] > 0
    assert res["reduction_pct"] < 50.0


def test_discharge_letter_batch_preserves_clinical_facts():
    """Batched discharge letters retain their clinical facts."""
    docs = [
        {"id": "first", "text": DISCHARGE_LETTER},
        {"id": "second", "text": DISCHARGE_LETTER},
    ]
    results = trim_batch(docs)
    for res in results:
        assert "Poids : 72.5 Kg" in res["trimmed_text"]
        assert "Conclusion : denutrition severe" in res["trimmed_text"]
        assert res["trimmed_chars"] > 0


@pytest.mark.parametrize(
    "document_id",
    [
        "first",
        "second",
    ],
)
def test_every_nonempty_document_uses_model_inference(document_id):
    document = {"id": document_id, "text": VOUCHER}
    result = trim_batch([document])[0]

    assert "is_bt" not in result
    assert result["trimmed_text"] == "\n".join(
        line for line in VOUCHER.splitlines() if line.strip()
    )
    assert result["preserved_intervals"]


def test_empty_document():
    """An empty document returns safely without error."""
    docs = [{"id": "empty", "text": ""}]
    results = trim_batch(docs)
    assert len(results) == 1
    assert results[0]["trimmed_text"] == ""
    assert "is_bt" not in results[0]
    assert results[0]["trimmed_chars"] == 0
    assert results[0]["preserved_intervals"] == []


def test_coordinates_and_grounding_guarantee():
    """Every preserved interval in 1-indexed [start, end] must exactly match the source text slice."""
    docs = [{"id": "letter_1", "text": DISCHARGE_LETTER}]
    results = trim_batch(docs)
    intervals = results[0]["preserved_intervals"]
    removed = results[0]["removed_intervals"]
    assert len(intervals) > 0

    for iv in intervals:
        start_1 = iv["start"]
        end_1 = iv["end"]
        assert start_1 <= end_1, f"Invalid interval: start {start_1} > end {end_1}"
        assert len(iv["text"].strip()) > 0, (
            "Preserved interval must not be empty or whitespace"
        )
        # In R 1-indexed inclusive [start, end] corresponds to Python [start-1 : end]
        sliced = DISCHARGE_LETTER[start_1 - 1 : end_1]
        assert sliced == iv["text"]

    # Preserved intervals must be strictly monotonically increasing without overlap
    starts = [iv["start"] for iv in intervals]
    ends = [iv["end"] for iv in intervals]
    for i in range(1, len(starts)):
        assert starts[i] > ends[i - 1], (
            f"Intervals must not overlap: start {starts[i]} <= prev end {ends[i - 1]}"
        )

    # Removed intervals must also have valid start <= end and non-empty text
    for iv in removed:
        assert iv["start"] <= iv["end"], (
            f"Invalid removed interval: start {iv['start']} > end {iv['end']}"
        )
        assert len(iv["text"].strip()) > 0, (
            "Removed interval must not be empty or whitespace"
        )
        sliced = DISCHARGE_LETTER[iv["start"] - 1 : iv["end"]]
        assert sliced == iv["text"]


def test_intervals_with_empty_and_whitespace_lines():
    """Documents with multiple consecutive empty and whitespace lines must produce strictly valid intervals."""
    text_with_empty_lines = (
        "\n\n\nPremier paragraphe clinique.\n"
        "   \n\t\n"
        "Deuxieme paragraphe clinique avec constantes : Poids 70 kg.\n\n\n"
    )
    docs = [{"id": "multi_empty", "text": text_with_empty_lines}]
    results = trim_batch(docs)
    res = results[0]

    for iv in res["preserved_intervals"]:
        assert iv["start"] <= iv["end"]
        assert len(iv["text"].strip()) > 0
        sliced = text_with_empty_lines[iv["start"] - 1 : iv["end"]]
        assert sliced == iv["text"]

    for iv in res["removed_intervals"]:
        assert iv["start"] <= iv["end"]
        assert len(iv["text"].strip()) > 0
        sliced = text_with_empty_lines[iv["start"] - 1 : iv["end"]]
        assert sliced == iv["text"]

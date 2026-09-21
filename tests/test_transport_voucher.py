"""Tests for the worker's rectype-aware-v1 transport-voucher contract.

These cases assert that:
1. Whole-document removal requires literal `FORMCHECKBOX` and a transport
   RECTYPE (`BT` or an `ORDON` prefix).
2. All other documents continue to ordinary model inference.
3. Coordinates and grounding guarantees are maintained for model output.
"""

from __future__ import annotations

import pytest

pytest.importorskip("onnxruntime")
np = pytest.importorskip("numpy")

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import trim_batch_service as worker
from trim_batch_service import is_transport_voucher, trim_batch

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
    docs = [{"id": "letter_1", "text": DISCHARGE_LETTER, "rectype": "CRH2AB"}]
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


def test_discharge_letter_with_different_rectypes():
    """Discharge letters under any clinical rectype (CRH2AB, LDL2024) retain clinical facts."""
    docs = [
        {"id": "crh", "text": DISCHARGE_LETTER, "rectype": "CRH2AB"},
        {"id": "ldl", "text": DISCHARGE_LETTER, "rectype": "LDL2024"},
    ]
    results = trim_batch(docs)
    for res in results:
        assert "Poids : 72.5 Kg" in res["trimmed_text"]
        assert "Conclusion : denutrition severe" in res["trimmed_text"]
        assert res["trimmed_chars"] > 0


@pytest.mark.parametrize("rectype", ["BT", "ORDON", "ORDON7", "ORDONACTE2"])
def test_transport_rectype_with_literal_form_marker_is_a_voucher(rectype):
    assert is_transport_voucher(VOUCHER, rectype)


@pytest.mark.parametrize("rectype", [None, "", "   ", "CRH2AB", "BT7", "ordon7", 7])
def test_missing_blank_or_unrelated_rectype_is_not_a_voucher(rectype):
    assert not is_transport_voucher(VOUCHER, rectype)


@pytest.mark.parametrize(
    "document",
    [
        {"id": "missing", "text": VOUCHER},
        {"id": "blank", "text": VOUCHER, "rectype": ""},
        {"id": "clinical", "text": VOUCHER, "rectype": "CRH2AB"},
    ],
)
def test_unconfirmed_voucher_continues_to_model_inference(document):
    result = trim_batch([document])[0]

    assert result["is_bt"] is False
    assert result["trimmed_text"] == "\n".join(
        line for line in VOUCHER.splitlines() if line.strip()
    )
    assert result["preserved_intervals"]


def test_form_marker_inside_clinical_letter_uses_model_inference():
    assert not is_transport_voucher(DISCHARGE_LETTER, "CRH2AB")


def test_transport_rectype_without_literal_form_marker_uses_model_inference():
    assert not is_transport_voucher("Clinical narrative", "BT")
    assert not is_transport_voucher("formcheckbox", "ORDON7")


@pytest.mark.parametrize("rectype", ["BT", "ORDON7"])
def test_confirmed_voucher_is_removed_without_loading_a_model(rectype):
    results = trim_batch([{"id": "v_1", "text": VOUCHER, "rectype": rectype}])
    assert results == [
        {
            "id": "v_1",
            "trimmed_text": "",
            "is_bt": True,
            "raw_chars": len(VOUCHER),
            "trimmed_chars": 0,
            "reduction_pct": 100.0,
            "preserved_intervals": [],
            "removed_intervals": [],
        }
    ]


def test_empty_document():
    """An empty document returns safely without error."""
    docs = [{"id": "empty", "text": "", "rectype": "ORDON7"}]
    results = trim_batch(docs)
    assert len(results) == 1
    assert results[0]["trimmed_text"] == ""
    assert results[0]["is_bt"] is False
    assert results[0]["trimmed_chars"] == 0
    assert results[0]["preserved_intervals"] == []


def test_coordinates_and_grounding_guarantee():
    """Every preserved interval in 1-indexed [start, end] must exactly match the source text slice."""
    docs = [{"id": "letter_1", "text": DISCHARGE_LETTER, "rectype": "CRH2AB"}]
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
    docs = [{"id": "multi_empty", "text": text_with_empty_lines, "rectype": "CRH2AB"}]
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

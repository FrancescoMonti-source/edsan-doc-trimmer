"""Tests for transport vouchers and discharge letters under pure model trimming.

`is_transport_voucher()` was removed in favor of Student v3 DrBERT retraining.
There is no longer a deterministic veto or shadow pipeline.

These cases assert that:
1. Discharge letters (`CRH2AB`, `LDL2024`) containing `FORMCHECKBOX` preserve
   all clinical facts verbatim (Poids, Motif, Conclusion).
2. Transport vouchers are properly trimmed down by the trained model.
3. Coordinates and grounding guarantees are maintained.
"""

from __future__ import annotations

import pytest

pytest.importorskip("onnxruntime")
pytest.importorskip("numpy")

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from trim_batch_service import trim_batch  # noqa: E402


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
    assert "Conclusion : denutrition severe dans un contexte d'agression aigue." in trimmed

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


def test_voucher_is_trimmed_by_model():
    """Transport vouchers must be significantly trimmed by the model."""
    docs = [{"id": "v_1", "text": VOUCHER, "rectype": "ORDON7"}]
    results = trim_batch(docs)
    assert len(results) == 1
    res = results[0]
    # The header and administrative lines should be trimmed
    assert res["reduction_pct"] > 30.0


def test_empty_document():
    """An empty document returns safely without error."""
    docs = [{"id": "empty", "text": "", "rectype": "ORDON7"}]
    results = trim_batch(docs)
    assert len(results) == 1
    assert results[0]["trimmed_text"] == ""
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
        assert len(iv["text"].strip()) > 0, "Preserved interval must not be empty or whitespace"
        # In R 1-indexed inclusive [start, end] corresponds to Python [start-1 : end]
        sliced = DISCHARGE_LETTER[start_1 - 1 : end_1]
        assert sliced == iv["text"]

    # Preserved intervals must be strictly monotonically increasing without overlap
    starts = [iv["start"] for iv in intervals]
    ends = [iv["end"] for iv in intervals]
    for i in range(1, len(starts)):
        assert starts[i] > ends[i - 1], f"Intervals must not overlap: start {starts[i]} <= prev end {ends[i-1]}"

    # Removed intervals must also have valid start <= end and non-empty text
    for iv in removed:
        assert iv["start"] <= iv["end"], f"Invalid removed interval: start {iv['start']} > end {iv['end']}"
        assert len(iv["text"].strip()) > 0, "Removed interval must not be empty or whitespace"
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


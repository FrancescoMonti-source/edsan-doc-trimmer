"""Unit tests for active learning, uncertainty calculation, and hard-negative mining."""

from redsan_doc_trimmer.mining import (
    LinePrediction,
    check_anchor_violations,
    compute_uncertainty_score,
    score_document_hardness,
)


def test_compute_uncertainty_score():
    is_unc, margin = compute_uncertainty_score(0.50)
    assert is_unc is True
    assert margin == 0.0

    is_unc, margin = compute_uncertainty_score(0.40)
    assert is_unc is True
    assert round(margin, 2) == 0.10

    is_unc, margin = compute_uncertainty_score(0.95)
    assert is_unc is False

    is_unc, margin = compute_uncertainty_score(0.05)
    assert is_unc is False


def test_check_anchor_violations():
    predictions = [
        LinePrediction(
            line_index=0,
            text="Centre Hospitalier Universitaire",
            prob_clinical=0.05,
            prob_boilerplate=0.95,
            is_uncertain=False,
        ),
        LinePrediction(
            line_index=1,
            text="Conclusion : Pas de signe d'ischémie myocardique aiguë.",
            prob_clinical=0.20,
            prob_boilerplate=0.80,  # Erroneous false positive!
            is_uncertain=False,
        ),
        LinePrediction(
            line_index=2,
            text="Patient stable sur le plan hémodynamique.",
            prob_clinical=0.95,
            prob_boilerplate=0.05,
            is_uncertain=False,
        ),
    ]

    anchors = {
        "CONCLUSIONDIAGNOSTIC": "Conclusion : Pas de signe d'ischémie myocardique aiguë."
    }

    violations = check_anchor_violations(predictions, anchors)
    assert len(violations) == 1
    assert "Line [1]" in violations[0]
    assert "CONCLUSIONDIAGNOSTIC" in violations[0]
    assert predictions[1].is_anchor_violation is True


def test_score_document_hardness():
    preds = [
        LinePrediction(0, "L0", 0.5, 0.5, is_uncertain=True),
        LinePrediction(1, "L1", 0.52, 0.48, is_uncertain=True),
        LinePrediction(2, "L2", 0.9, 0.1, is_uncertain=False),
        LinePrediction(
            3,
            "L3 Diagnostic confirmé",
            0.1,
            0.9,
            is_uncertain=False,
            is_anchor_violation=True,
        ),
    ]
    anchors = {"CONCLUSION": "L3 Diagnostic confirmé"}

    result = score_document_hardness("doc_test", "CRH2AB", preds, anchors)
    assert result.doc_id == "doc_test"
    assert result.uncertain_line_count == 2
    assert result.anchor_violation_count == 1
    assert result.hardness_score > 0

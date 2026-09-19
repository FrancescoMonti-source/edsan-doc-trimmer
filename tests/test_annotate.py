"""Tests for teacher annotation logic, span conversions, and clinical anchor safeguards."""

import json

from redsan_doc_trimmer.annotate import (
    BoilerplateSpan,
    DocBoilerplateAnnotation,
    annotate_corpus_file,
    validate_against_clinical_anchors,
)
from redsan_doc_trimmer.dataset import extract_lines_with_offsets


def test_boilerplate_span_mapping():
    doc_id = "test_doc_1"
    spans = [
        BoilerplateSpan(
            start_line=0, end_line=2, category="hospital_header", reason="CHU header"
        ),
        BoilerplateSpan(
            start_line=5, end_line=5, category="signature_block", reason="signature"
        ),
    ]
    annotation = DocBoilerplateAnnotation(doc_id=doc_id, boilerplate_spans=spans)

    indices = annotation.get_boilerplate_line_indices(total_lines=6)
    assert indices == {0, 1, 2, 5}

    text = "CHU Rouen\nService de Nutrition\nTel: 02.32.88.88.88\nPatient vu pour suivi.\nExamen normal.\nDr Martin"
    doc_spans = extract_lines_with_offsets(text)
    annotated = annotation.apply_to_spans(doc_spans)

    assert len(annotated) == 6
    assert annotated[0].is_boilerplate is True
    assert annotated[1].is_boilerplate is True
    assert annotated[2].is_boilerplate is True
    assert annotated[3].is_boilerplate is False
    assert annotated[4].is_boilerplate is False
    assert annotated[5].is_boilerplate is True


def test_clinical_anchor_safeguard():
    # If a line mistakenly matches a clinical anchor, it must be rescued
    text = "Hôpital de Rouen\nConclusion : Absence de lésion suspecte à ce jour.\nDr Martin"
    spans = extract_lines_with_offsets(text)

    # Erroneous annotation that marked line 1 (conclusion) as boilerplate
    bad_annotation = DocBoilerplateAnnotation(
        doc_id="doc_err",
        boilerplate_spans=[
            BoilerplateSpan(
                start_line=0, end_line=1, category="header", reason="error"
            ),
            BoilerplateSpan(
                start_line=2, end_line=2, category="sig", reason="signature"
            ),
        ],
    )

    anchors = {
        "CONCLUSIONDIAGNOSTIC": "Conclusion : Absence de lésion suspecte à ce jour."
    }

    validated_annot, rescued = validate_against_clinical_anchors(
        bad_annotation, spans, anchors
    )
    rescued_indices = validated_annot.get_boilerplate_line_indices(len(spans))

    assert 1 not in rescued_indices  # Line 1 was successfully rescued!
    assert 0 in rescued_indices  # Line 0 remains boilerplate
    assert 2 in rescued_indices  # Line 2 remains boilerplate
    assert len(rescued) == 1


def test_mock_annotation_and_corpus_export(tmp_path):
    # Test batch runner with temporary files
    sample_input = tmp_path / "sample_input.jsonl"
    sample_output = tmp_path / "sample_output.jsonl"

    rec1 = {
        "doc_id": "test_1",
        "rectype": "CRH2AB",
        "rectxt": "Centre Hospitalier Universitaire de Rouen\nService Digestif\nPatient admis pour rectorragies.\nDr Dupont",
        "clinical_anchors": {},
    }
    rec2 = {
        "doc_id": "test_2",
        "rectype": "DICT",
        "rectxt": "Hôpitaux de Rouen\n\nPas d'anomalie décelée.\n\nSecrétariat médical",
        "clinical_anchors": {},
    }

    with open(sample_input, "w", encoding="utf-8") as f:
        f.write(json.dumps(rec1) + "\n")
        f.write(json.dumps(rec2) + "\n")

    count = annotate_corpus_file(
        str(sample_input),
        str(sample_output),
        use_mock=True,
        resume=False,
    )
    assert count == 2

    # Verify output contents
    with open(sample_output, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f]

    assert len(lines) == 2
    assert lines[0]["doc_id"] == "test_1"
    assert lines[1]["doc_id"] == "test_2"
    assert "boilerplate_spans" in lines[0]
    assert "lines" in lines[0]

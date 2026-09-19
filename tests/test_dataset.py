"""Tests for dataset extraction and exact character offset preservation."""

from pathlib import Path

from redsan_doc_trimmer.dataset import (
    DocumentSpan,
    build_windowed_samples,
    extract_lines_with_offsets,
    load_raw_documents,
    reconstruct_trimmed_text,
    verify_coordinates_and_grounding,
)


def test_exact_character_offsets():
    sample_text = (
        "Hôpital Universitaire\nService de Médecine Interne\nPatient de 65 ans."
    )
    spans = extract_lines_with_offsets(sample_text)

    assert len(spans) == 3
    assert verify_coordinates_and_grounding(sample_text, spans)

    assert spans[0].text == "Hôpital Universitaire"
    assert spans[1].text == "Service de Médecine Interne"
    assert spans[2].text == "Patient de 65 ans."


def test_windowed_samples_context():
    sample_text = "L1\nL2\nL3\nL4"
    spans = extract_lines_with_offsets(sample_text)
    samples = build_windowed_samples(spans, window_size=1)

    assert len(samples) == 4
    assert samples[1].target_text == "L2"
    assert samples[1].context_before == "L1"
    assert samples[1].context_after == "L3"

    assert samples[0].context_before == ""
    assert samples[0].context_after == "L2"


def test_reconstruct_trimmed_text():
    sample_text = "En-tête CHU\nPatient hospitalisé pour dyspnée.\nSignature Dr Martin"
    spans = extract_lines_with_offsets(sample_text)
    spans = [
        DocumentSpan(
            spans[0].start_char, spans[0].end_char, spans[0].text, is_boilerplate=True
        ),
        DocumentSpan(
            spans[1].start_char, spans[1].end_char, spans[1].text, is_boilerplate=False
        ),
        DocumentSpan(
            spans[2].start_char, spans[2].end_char, spans[2].text, is_boilerplate=True
        ),
    ]

    trimmed = reconstruct_trimmed_text(sample_text, spans, remove_boilerplate=True)
    assert trimmed == "Patient hospitalisé pour dyspnée."


def test_real_corpus_sample_coordinates():
    corpus_file = Path("data/raw/corpus_sample.jsonl")
    if not corpus_file.exists():
        return

    docs = load_raw_documents(str(corpus_file))[:20]
    assert len(docs) > 0

    for doc in docs:
        raw_text = doc["rectxt"]
        spans = extract_lines_with_offsets(raw_text)
        assert len(spans) > 0
        assert verify_coordinates_and_grounding(raw_text, spans), (
            f"Grounding guarantee failed for doc {doc['doc_id']}"
        )
        # Ensure no ORDON or BT was leaked into the sample
        assert not doc["rectype"].startswith("ORDON")
        assert not doc["rectype"].startswith("BT")

"""Tests for dataset extraction and exact character offset preservation."""

from redsan_doc_trimmer.dataset import extract_lines_with_offsets, build_windowed_samples


def test_exact_character_offsets():
    sample_text = "Hôpital Universitaire\nService de Médecine Interne\nPatient de 65 ans."
    spans = extract_lines_with_offsets(sample_text)

    assert len(spans) == 3
    # Check text reconstruction via slice
    for span in spans:
        extracted = sample_text[span.start_char : span.end_char]
        assert extracted == span.text

    assert spans[0].text == "Hôpital Universitaire"
    assert spans[1].text == "Service de Médecine Interne"
    assert spans[2].text == "Patient de 65 ans."


def test_windowed_samples_context():
    sample_text = "L1\nL2\nL3\nL4"
    spans = extract_lines_with_offsets(sample_text)
    samples = build_windowed_samples(spans, window_size=1)

    assert len(samples) == 4
    # Middle line has both prev and next
    assert samples[1].target_text == "L2"
    assert samples[1].context_before == "L1"
    assert samples[1].context_after == "L3"

    # First line has empty context_before
    assert samples[0].context_before == ""
    assert samples[0].context_after == "L2"

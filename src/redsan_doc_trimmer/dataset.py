"""Dataset preparation with exact character offset tracking and surrounding context."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DocumentSpan:
    """Represents a span in a raw RECTXT document with exact character coordinates."""

    start_char: int
    end_char: int
    text: str
    is_boilerplate: bool = False
    label_source: str = "unlabeled"


@dataclass(frozen=True)
class WindowedLineSample:
    """A target line paired with its surrounding context for classification."""

    line_index: int
    start_char: int
    end_char: int
    target_text: str
    context_before: str
    context_after: str
    is_boilerplate: bool = False
    doc_id: str | None = None

    def to_input_pair(self) -> tuple[str, str]:
        """Returns (target_text, full_context) for two-sequence encoder models."""
        context = f"{self.context_before} \n {self.context_after}".strip()
        return self.target_text, context


def extract_lines_with_offsets(rectxt: str) -> list[DocumentSpan]:
    """Splits a raw RECTXT document into lines while preserving exact character offsets."""
    lines: list[DocumentSpan] = []
    current_offset = 0

    # Handle various newline conventions without losing character index parity
    raw_lines = rectxt.splitlines(keepends=True)
    for raw_line in raw_lines:
        start = current_offset
        end = current_offset + len(raw_line)
        current_offset = end

        # Strip trailing newlines for the text representation
        clean_text = raw_line.rstrip("\r\n")
        lines.append(
            DocumentSpan(
                start_char=start,
                end_char=start + len(clean_text),
                text=clean_text,
            )
        )

    return lines


def verify_coordinates_and_grounding(rectxt: str, spans: list[DocumentSpan]) -> bool:
    """Verifies that every span coordinates exactly match the slice of raw RECTXT.

    This enforces the Coordinates & Grounding Guarantee required for downstream citation
    and quote verification in redsan and redsan-coding.
    """
    for span in spans:
        extracted = rectxt[span.start_char : span.end_char]
        if extracted != span.text:
            return False
    return True


def build_windowed_samples(
    spans: list[DocumentSpan],
    window_size: int = 2,
    doc_id: str | None = None,
) -> list[WindowedLineSample]:
    """Builds samples with surrounding context lines to disambiguate broken lines."""
    samples: list[WindowedLineSample] = []
    n = len(spans)

    for i, span in enumerate(spans):
        # Gather previous non-empty lines
        prev_lines = [
            spans[j].text
            for j in range(max(0, i - window_size), i)
            if spans[j].text.strip()
        ]
        context_before = " \n ".join(prev_lines)

        # Gather subsequent non-empty lines
        next_lines = [
            spans[j].text
            for j in range(i + 1, min(n, i + 1 + window_size))
            if spans[j].text.strip()
        ]
        context_after = " \n ".join(next_lines)

        samples.append(
            WindowedLineSample(
                line_index=i,
                start_char=span.start_char,
                end_char=span.end_char,
                target_text=span.text,
                context_before=context_before,
                context_after=context_after,
                is_boilerplate=span.is_boilerplate,
                doc_id=doc_id,
            )
        )

    return samples


def reconstruct_trimmed_text(
    rectxt: str,
    spans: list[DocumentSpan],
    remove_boilerplate: bool = True,
) -> str:
    """Reconstructs the trimmed document text from spans.

    Only lines with is_boilerplate == False are retained if remove_boilerplate is True.
    """
    kept_spans = [s for s in spans if not (remove_boilerplate and s.is_boilerplate)]
    return "\n".join(s.text for s in kept_spans if s.text.strip())


def load_raw_documents(jsonl_path: str) -> list[dict[str, Any]]:
    """Loads raw documents from an extracted JSONL file."""
    records: list[dict[str, Any]] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_annotated_samples(
    jsonl_path: str,
    window_size: int = 2,
    filter_empty_target: bool = True,
) -> list[WindowedLineSample]:
    """Loads annotated records from JSONL and converts them into WindowedLineSample instances."""
    samples: list[WindowedLineSample] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            doc_id = doc.get("doc_id")
            line_data = doc.get("lines", [])

            spans = [
                DocumentSpan(
                    start_char=item["start_char"],
                    end_char=item["end_char"],
                    text=item["text"],
                    is_boilerplate=item.get("is_boilerplate", False),
                    label_source="annotated",
                )
                for item in line_data
            ]

            doc_samples = build_windowed_samples(
                spans, window_size=window_size, doc_id=doc_id
            )
            if filter_empty_target:
                doc_samples = [s for s in doc_samples if s.target_text.strip()]
            samples.extend(doc_samples)

    return samples

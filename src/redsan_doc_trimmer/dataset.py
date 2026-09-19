"""Dataset preparation with exact character offset tracking and surrounding context."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


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

    def to_input_pair(self) -> Tuple[str, str]:
        """Returns (target_text, full_context) for two-sequence encoder models."""
        context = f"{self.context_before} \n {self.context_after}".strip()
        return self.target_text, context


def extract_lines_with_offsets(rectxt: str) -> List[DocumentSpan]:
    """Splits a raw RECTXT document into lines while preserving exact character offsets."""
    lines: List[DocumentSpan] = []
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


def build_windowed_samples(
    spans: List[DocumentSpan],
    window_size: int = 2,
) -> List[WindowedLineSample]:
    """Builds samples with surrounding context lines to disambiguate broken lines."""
    samples: List[WindowedLineSample] = []
    n = len(spans)

    for i, span in enumerate(spans):
        # Gather previous non-empty lines
        prev_lines = [
            spans[j].text for j in range(max(0, i - window_size), i) if spans[j].text.strip()
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
            )
        )

    return samples

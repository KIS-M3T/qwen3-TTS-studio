"""Offset tracking and span mapping for Italian Text Normalization (Milestone M2, WP1).

Maintains exact coordinate mapping between original raw_text and intermediate/normalized
strings across multiple transformation stages (S3, P1, P4).
Ensures all TNAnnotation objects point to half-open intervals [char_start, char_end)
on the original raw_text.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

from src.linguistics.types import TNAnnotation


@dataclass
class _DeltaRecord:
    """Records a single length mutation in the text."""
    curr_pos: int       # Position in the intermediate text where replacement began
    old_len: int        # Original length of the replaced span
    new_len: int        # New length after replacement
    raw_start: int      # Exact start offset in original raw_text
    raw_end: int        # Exact end offset in original raw_text


class OffsetTracker:
    """Tracks length delta shifts to map intermediate text spans back to original raw_text."""

    def __init__(self, raw_text: str):
        """Initialize tracker with original raw input text.

        Args:
            raw_text: Original raw input text.
        """
        self.raw_text = raw_text
        self.current_text = raw_text
        self.annotations: List[TNAnnotation] = []
        # List of delta mutations: (curr_pos, old_len, new_len, raw_start, raw_end)
        self._mutations: List[_DeltaRecord] = []

    def get_raw_span(self, intermediate_start: int, intermediate_end: int) -> Tuple[int, int]:
        """Convert intermediate string coordinates [start, end) back to original raw_text [start, end).

        Args:
            intermediate_start: Start index in current intermediate text.
            intermediate_end: End index in current intermediate text.

        Returns:
            Tuple of (raw_start, raw_end) in original raw_text.
        """
        # Map a single point from intermediate text back to raw_text
        raw_start = self._map_point_to_raw(intermediate_start)
        raw_end = self._map_point_to_raw(intermediate_end)
        return raw_start, raw_end

    def _map_point_to_raw(self, intermediate_pos: int) -> int:
        """Trace an intermediate position back to original raw_text index."""
        pos = intermediate_pos
        # Trace backwards through mutations in reverse chronological order
        for mut in reversed(self._mutations):
            if pos < mut.curr_pos:
                continue
            elif pos <= mut.curr_pos + mut.new_len:
                # Point falls inside the mutated region: map proportionally or to boundary
                if mut.new_len == 0:
                    pos = mut.raw_start
                else:
                    ratio = min(1.0, max(0.0, (pos - mut.curr_pos) / mut.new_len))
                    pos = mut.raw_start + int(ratio * mut.old_len)
            else:
                # Point is after the mutated region: adjust by delta (new_len - old_len)
                pos = pos - (mut.new_len - mut.old_len)
        return max(0, min(len(self.raw_text), pos))

    def record_replacement(
        self,
        intermediate_start: int,
        intermediate_end: int,
        replacement: str,
        category: str,
        raw_span: Optional[str] = None,
    ) -> TNAnnotation:
        """Record an in-place text replacement and generate a calibrated TNAnnotation.

        Args:
            intermediate_start: Start index in current intermediate text.
            intermediate_end: End index in current intermediate text.
            replacement: Expanded replacement string.
            category: NSW category string (e.g. "cardinal", "date", "currency").
            raw_span: Optional original span string. If None, sliced from raw_text.

        Returns:
            TNAnnotation with coordinates strictly mapped to original raw_text.
        """
        raw_start, raw_end = self.get_raw_span(intermediate_start, intermediate_end)
        actual_raw_span = raw_span if raw_span is not None else self.raw_text[raw_start:raw_end]

        old_len = intermediate_end - intermediate_start
        new_len = len(replacement)

        mutation = _DeltaRecord(
            curr_pos=intermediate_start,
            old_len=old_len,
            new_len=new_len,
            raw_start=raw_start,
            raw_end=raw_end,
        )
        self._mutations.append(mutation)

        # Update current intermediate text
        self.current_text = (
            self.current_text[:intermediate_start]
            + replacement
            + self.current_text[intermediate_end:]
        )

        annotation = TNAnnotation(
            raw_span=actual_raw_span,
            expanded_span=replacement,
            category=category,
            char_start=raw_start,
            char_end=raw_end,
        )
        self.annotations.append(annotation)
        return annotation

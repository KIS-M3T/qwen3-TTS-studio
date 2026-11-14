"""Offset tracking and span mapping for Italian Text Normalization (Milestone M2, WP1).

Maintains exact coordinate mapping between original raw_text and intermediate/normalized
strings across multiple transformation stages (S3, P1, P4, WP1-B1).

Employs a piecewise chunk-based model that prevents double re-mapping and preserves
strict alignment between the tracker and the intermediate shielded text.
Ensures all TNAnnotation objects point to half-open intervals [char_start, char_end)
on the original raw_text.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

from src.linguistics.shielding import ShieldedEntity
from src.linguistics.types import TNAnnotation


@dataclass
class _Chunk:
    """Represents a continuous segment of text mapped to an original raw_text interval."""
    text: str
    raw_start: int
    raw_end: int


class OffsetTracker:
    """Tracks length delta shifts to map intermediate text spans back to original raw_text."""

    def __init__(self, raw_text: str):
        """Initialize tracker with original raw input text.

        Args:
            raw_text: Original raw input text.
        """
        self.raw_text = raw_text
        self._chunks: List[_Chunk] = (
            [_Chunk(text=raw_text, raw_start=0, raw_end=len(raw_text))]
            if raw_text
            else []
        )
        self.annotations: List[TNAnnotation] = []

    @property
    def current_text(self) -> str:
        """Return the current intermediate reconstructed text."""
        return "".join(c.text for c in self._chunks)

    def _split_at(self, pos: int) -> None:
        """Split chunks at the specified intermediate character index."""
        if pos <= 0 or not self._chunks:
            return

        curr = 0
        new_chunks = []
        for c in self._chunks:
            c_len = len(c.text)
            if curr < pos < curr + c_len:
                offset = pos - curr
                raw_span = c.raw_end - c.raw_start
                if c_len == raw_span:
                    raw_mid = c.raw_start + offset
                else:
                    raw_mid = c.raw_start + int(round(offset * raw_span / c_len))

                left = _Chunk(c.text[:offset], c.raw_start, raw_mid)
                right = _Chunk(c.text[offset:], raw_mid, c.raw_end)
                new_chunks.extend([left, right])
            else:
                new_chunks.append(c)
            curr += c_len

        self._chunks = new_chunks

    def get_raw_span(self, intermediate_start: int, intermediate_end: int) -> Tuple[int, int]:
        """Convert intermediate string coordinates [start, end) back to original raw_text [start, end).

        Args:
            intermediate_start: Start index in current intermediate text.
            intermediate_end: End index in current intermediate text.

        Returns:
            Tuple of (raw_start, raw_end) in original raw_text.
        """
        if not self._chunks:
            return 0, 0

        self._split_at(intermediate_start)
        self._split_at(intermediate_end)

        curr = 0
        idx_start = None
        idx_end = None
        for i, c in enumerate(self._chunks):
            c_len = len(c.text)
            if curr == intermediate_start and idx_start is None:
                idx_start = i
            if curr + c_len == intermediate_end and idx_end is None:
                idx_end = i + 1
                break
            curr += c_len

        if idx_start is None or idx_end is None:
            raise ValueError(
                f"Cannot map intermediate slice [{intermediate_start}:{intermediate_end}] "
                f"in text of length {len(self.current_text)}"
            )

        middle = self._chunks[idx_start:idx_end]
        return middle[0].raw_start, middle[-1].raw_end

    def record_replacement(
        self,
        intermediate_start: int,
        intermediate_end: int,
        replacement: str,
        category: str,
        raw_span: Optional[str] = None,
        track_annotation: bool = True,
    ) -> Optional[TNAnnotation]:
        """Record an in-place text replacement and generate a calibrated TNAnnotation.

        Args:
            intermediate_start: Start index in current intermediate text.
            intermediate_end: End index in current intermediate text.
            replacement: Expanded replacement string.
            category: NSW category string (e.g. "cardinal", "date", "currency").
            raw_span: Optional original span string. If None, sliced from raw_text.
            track_annotation: Whether to append a TNAnnotation to self.annotations.

        Returns:
            TNAnnotation with coordinates strictly mapped to original raw_text, or None.
        """
        if intermediate_start < 0 or intermediate_end > len(self.current_text):
            raise IndexError(
                f"Indices [{intermediate_start}:{intermediate_end}] out of bounds for text length {len(self.current_text)}"
            )

        self._split_at(intermediate_start)
        self._split_at(intermediate_end)

        curr = 0
        idx_start = None
        idx_end = None
        for i, c in enumerate(self._chunks):
            c_len = len(c.text)
            if curr == intermediate_start and idx_start is None:
                idx_start = i
            if curr + c_len == intermediate_end and idx_end is None:
                idx_end = i + 1
                break
            curr += c_len

        if idx_start is None or idx_end is None:
            raise ValueError(
                f"Cannot find chunk slice for [{intermediate_start}:{intermediate_end}] "
                f"in current text of length {len(self.current_text)}"
            )

        middle = self._chunks[idx_start:idx_end]
        raw_start = middle[0].raw_start
        raw_end = middle[-1].raw_end

        # Replace middle chunks with a single chunk covering the same raw_text interval
        new_chunk = _Chunk(text=replacement, raw_start=raw_start, raw_end=raw_end)
        self._chunks = self._chunks[:idx_start] + [new_chunk] + self._chunks[idx_end:]

        if not track_annotation:
            return None

        actual_raw_span = raw_span if raw_span is not None else self.raw_text[raw_start:raw_end]
        annotation = TNAnnotation(
            raw_span=actual_raw_span,
            expanded_span=replacement,
            category=category,
            char_start=raw_start,
            char_end=raw_end,
        )
        self.annotations.append(annotation)
        return annotation

    def apply_shielding(self, shielded_entities: List[ShieldedEntity]) -> None:
        """Align the tracker with the Stage 1 shielded text before intermediate modules run.

        Args:
            shielded_entities: List of ShieldedEntity from Stage 1.
        """
        for entity in shielded_entities:
            # In raw_text, each entity starts at entity.char_start and ends at entity.char_end.
            # Find its intermediate position in current_text:
            pos = self.current_text.find(entity.original_text)
            if pos != -1:
                self.record_replacement(
                    intermediate_start=pos,
                    intermediate_end=pos + len(entity.original_text),
                    replacement=entity.placeholder,
                    category="shield",
                    track_annotation=False,
                )

    def deshield_entity(self, placeholder: str, original_text: str) -> None:
        """Restore a shielded placeholder in the tracker chunks during Stage 8.

        Args:
            placeholder: Private placeholder string to replace.
            original_text: Original entity text to restore.
        """
        pos = self.current_text.find(placeholder)
        if pos != -1:
            self.record_replacement(
                intermediate_start=pos,
                intermediate_end=pos + len(placeholder),
                replacement=original_text,
                category="deshield",
                track_annotation=False,
            )

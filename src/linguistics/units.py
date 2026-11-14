"""Units of Measure and Percentages Normalization (Milestone M2, WP4).

Implements Stage 4 of the Italian Linguistic Normalization Pipeline (Spec Rev 2.1, § 2.4):
- Percentages ('25%' -> 'venticinque per cento', '0,5%' -> 'zero virgola cinque per cento')
- Metric and physical units ('10 km' -> 'dieci chilometri', '1 kg' -> 'un chilogrammo',
  '80 km/h' -> 'ottanta chilometri orari', '85 m²' -> 'ottantacinque metri quadrati')
- Respects Stage 1 shielded tokens and protected numeric clusters.
"""

import re
from typing import List, Tuple

from src.linguistics.numbers import (
    METRIC_UNIT_ALT,
    METRIC_UNIT_SPECS,
    collect_protected_spans,
    format_decimal_italian,
    is_leading_zero_integer,
    is_unit_singular_one,
    overlaps_any,
)
from src.linguistics.tracker import OffsetTracker

_PERCENTAGE_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_.,/€$£–—−:-])"
    r"(?P<sign>-)?"
    r"(?P<int>\d{1,3}(?:\.\d{3})+|\d+)"
    r"(?:,(?P<frac>\d+))?"
    r"\s*%"
    r"(?![A-Za-z0-9_%]|[.,]\d|[.,][.,])"
)

_UNIT_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_.,/€$£–—−:-])"
    r"(?P<sign>-)?"
    r"(?P<int>\d{1,3}(?:\.\d{3})+|\d+)"
    r"(?:,(?P<frac>\d+))?"
    r"\s*"
    rf"(?P<unit>{METRIC_UNIT_ALT})"
    r"(?![A-Za-z0-9_/²³°%]|\.[A-Za-z0-9_])"
)

_PRECEDING_RANGE_OR_RATIO = re.compile(r"\d\s*[-–—−/]\s*$")


class UnitsNormalizer:
    """Normalizes percentages and metric/physical units (Milestone M2, WP4)."""

    @classmethod
    def normalize_units(cls, text: str, tracker: OffsetTracker) -> str:
        """Find and normalize percentages and metric/physical units in Stage 4 text."""
        if not text:
            return text

        protected_spans = collect_protected_spans(text)
        # Tuple: (start, end, replacement, category)
        candidates: List[Tuple[int, int, str, str]] = []

        # 1. Percentages ('25%', '0,5%')
        for m in _PERCENTAGE_PATTERN.finditer(text):
            start, end = m.span()
            if overlaps_any(start, end, protected_spans) or _PRECEDING_RANGE_OR_RATIO.search(text[:start]):
                continue

            int_token = m.group("int")
            if is_leading_zero_integer(int_token):
                continue

            num_words = format_decimal_italian(
                int_token,
                m.group("frac"),
                bool(m.group("sign")),
            )
            if num_words is not None:
                candidates.append((start, end, f"{num_words} per cento", "percentage"))

        # 2. Metric and physical units ('10 km', '1 kg', '80 km/h', '85 m²')
        for m in _UNIT_PATTERN.finditer(text):
            start, end = m.span()
            if overlaps_any(start, end, protected_spans) or _PRECEDING_RANGE_OR_RATIO.search(text[:start]):
                continue
            if any(c_start < end and start < c_end for c_start, c_end, _, _ in candidates):
                continue

            unit_key = m.group("unit")
            singular, plural = METRIC_UNIT_SPECS[unit_key]
            int_token = m.group("int")
            if is_leading_zero_integer(int_token):
                continue

            frac_token = m.group("frac")
            is_negative = bool(m.group("sign"))

            if is_unit_singular_one(int_token, frac_token):
                prefix = "meno " if is_negative else ""
                replacement = f"{prefix}un {singular}"
            else:
                num_words = format_decimal_italian(int_token, frac_token, is_negative)
                if num_words is None:
                    continue
                replacement = f"{num_words} {plural}"

            candidates.append((start, end, replacement, "unit_measure"))

        if not candidates:
            return text

        # Apply replacements right-to-left to preserve earlier intermediate indices
        candidates.sort(key=lambda c: c[0], reverse=True)
        for start, end, replacement, category in candidates:
            tracker.record_replacement(
                intermediate_start=start,
                intermediate_end=end,
                replacement=replacement,
                category=category,
            )

        tracker.annotations.sort(key=lambda a: a.char_start)
        return tracker.current_text

"""Currency and monetary amount normalization for Italian Text Normalization (Milestone M2, WP3).

Implements formal monetary register (Ratified Decision D5.A):
- Supports prefix and suffix currency symbols (€, $, £) and ISO 4217 codes (EUR, USD, GBP)
- Repositions currency nouns uniformly after the numeric value
- Strict grammatical invariance for 'euro' (never 'euri')
- Gender-accurate singular/plural forms ('un euro', 'una sterlina', 'un dollaro')
- Formal cents declination ('un centesimo', 'cinquanta centesimi')
- Explicit zero amounts ('zero euro') and negative signs ('meno quindici euro')
- Italian thousands separator (dots) and decimal comma parsing
- Non-interference with shielded entities (§§§SHIELD_...) and numeric ranges
"""

import re
from dataclasses import dataclass
from typing import List, Tuple

from src.linguistics.numbers_core import MAX_SUPPORTED_INT, cardinal_to_italian
from src.linguistics.tracker import OffsetTracker


@dataclass(frozen=True)
class _CurrencySpec:
    singular: str
    plural: str
    gender: str  # "m" or "f"
    sub_singular: str = "centesimo"
    sub_plural: str = "centesimi"
    sub_gender: str = "m"


_CURRENCY_SPECS = {
    "€": _CurrencySpec("euro", "euro", "m"),
    "EUR": _CurrencySpec("euro", "euro", "m"),
    "$": _CurrencySpec("dollaro", "dollari", "m"),
    "USD": _CurrencySpec("dollaro", "dollari", "m"),
    "£": _CurrencySpec("sterlina", "sterline", "f"),
    "GBP": _CurrencySpec("sterlina", "sterline", "f"),
}

# Prefix form: optional '-', currency symbol/code, optional '-', integer[.thousands][,cents]
_PREFIX_CURRENCY_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_.,-€$£])"
    r"(?P<sign1>-\s*)?"
    r"(?P<curr>€|\$|£|(?:EUR|USD|GBP)\b)"
    r"\s*"
    r"(?P<sign2>-\s*)?"
    r"(?P<int>\d{1,3}(?:\.\d{3})+|\d+)"
    r"(?:,(?P<cents>\d{1,2}))?"
    r"(?![.,]\d|/\d|\s*-\s*\d|\s*(?:€|\$|£|(?:EUR|USD|GBP)\b)|[\d%A-Za-z€$£])"
)

# Suffix form: optional '-', integer[.thousands][,cents], currency symbol/code
_SUFFIX_CURRENCY_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_.,-€$£])"
    r"(?P<sign>-\s*)?"
    r"(?P<int>\d{1,3}(?:\.\d{3})+|\d+)"
    r"(?:,(?P<cents>\d{1,2}))?"
    r"\s*"
    r"(?P<curr>€|\$|£|(?:EUR|USD|GBP)\b)"
    r"(?![A-Za-z0-9_€$£])"
)

_PRECEDING_CURRENCY_MARKER = re.compile(r"(?:€|\$|£|\b(?:EUR|USD|GBP))\s*$")

_SHIELD_PATTERN = re.compile(r"§§§SHIELD_[a-f0-9]{12}_\d+§§§")


def _format_unit(amount: int, singular: str, plural: str, gender: str) -> str:
    """Format an integer monetary unit with Italian singular/plural agreement."""
    if amount == 1:
        num_word = "una" if gender == "f" else "un"
        return f"{num_word} {singular}"
    if amount >= 1_000_000 and amount % 1_000_000 == 0:
        return f"{cardinal_to_italian(amount)} di {plural}"
    return f"{cardinal_to_italian(amount)} {plural}"


def _format_monetary_amount(
    int_val: int,
    cents_val: int,
    is_negative: bool,
    spec: _CurrencySpec,
) -> str:
    """Build formal spoken Italian monetary phrase (Decision D5.A)."""
    prefix = "meno " if is_negative else ""

    # Pure zero amount: 0 € / 0,00 € -> "zero euro"
    if int_val == 0 and cents_val == 0:
        return f"{prefix}zero {spec.plural}"

    # Pure fractional cents: 0,01 € -> "un centesimo"; 0,50 € -> "cinquanta centesimi"
    if int_val == 0 and cents_val > 0:
        cents_str = _format_unit(
            cents_val, spec.sub_singular, spec.sub_plural, spec.sub_gender
        )
        return f"{prefix}{cents_str}"

    # Integer >= 1, with optional cents
    major_str = _format_unit(int_val, spec.singular, spec.plural, spec.gender)
    if cents_val == 0:
        return f"{prefix}{major_str}"

    cents_str = _format_unit(
        cents_val, spec.sub_singular, spec.sub_plural, spec.sub_gender
    )
    return f"{prefix}{major_str} e {cents_str}"


class CurrencyNormalizer:
    """Normalizes Italian and major foreign currency expressions (Milestone M2, WP3)."""

    @classmethod
    def normalize_currencies(
        cls,
        text: str,
        tracker: OffsetTracker,
        register: str = "formal",
    ) -> str:
        """Find and normalize currency expressions in intermediate text.

        Args:
            text: Current intermediate text (from tracker.current_text).
            tracker: OffsetTracker instance to register replacements.
            register: Monetary register ("formal" ratified as D5.A).

        Returns:
            Updated intermediate text.
        """
        if not text:
            return text

        shield_spans = [m.span() for m in _SHIELD_PATTERN.finditer(text)]

        def is_shielded(start: int, end: int) -> bool:
            return any(s_start < end and s_end > start for s_start, s_end in shield_spans)

        candidates: List[Tuple[int, int, str]] = []

        # 1. Suffix currency forms (e.g. 50 €, 12,50 €, -15,00 €, 200 EUR)
        for m in _SUFFIX_CURRENCY_PATTERN.finditer(text):
            start, end = m.span()
            if is_shielded(start, end):
                continue
            if _PRECEDING_CURRENCY_MARKER.search(text[:start]):
                continue  # Guard against malformed dual-symbol expressions like '€ 50 €'

            curr_key = m.group("curr")
            spec = _CURRENCY_SPECS.get(curr_key)
            if spec is None:
                continue

            int_str = m.group("int").replace(".", "")
            int_val = int(int_str)
            if int_val > MAX_SUPPORTED_INT:
                continue

            cents_raw = m.group("cents")
            if cents_raw is None:
                cents_val = 0
            elif len(cents_raw) == 1:
                cents_val = int(cents_raw) * 10
            else:
                cents_val = int(cents_raw)

            is_negative = bool(m.group("sign"))
            replacement = _format_monetary_amount(int_val, cents_val, is_negative, spec)
            candidates.append((start, end, replacement))

        # 2. Prefix currency forms (e.g. € 2.500, $ 100, £ 50, -€ 15,00)
        for m in _PREFIX_CURRENCY_PATTERN.finditer(text):
            start, end = m.span()
            if is_shielded(start, end):
                continue

            # Skip if overlapping with an already matched suffix expression
            if any(c_start < end and start < c_end for c_start, c_end, _ in candidates):
                continue

            sign1 = m.group("sign1")
            sign2 = m.group("sign2")
            if sign1 and sign2:
                continue  # Double sign is malformed

            curr_key = m.group("curr")
            spec = _CURRENCY_SPECS.get(curr_key)
            if spec is None:
                continue

            int_str = m.group("int").replace(".", "")
            int_val = int(int_str)
            if int_val > MAX_SUPPORTED_INT:
                continue

            cents_raw = m.group("cents")
            if cents_raw is None:
                cents_val = 0
            elif len(cents_raw) == 1:
                cents_val = int(cents_raw) * 10
            else:
                cents_val = int(cents_raw)

            is_negative = bool(sign1 or sign2)
            replacement = _format_monetary_amount(int_val, cents_val, is_negative, spec)
            candidates.append((start, end, replacement))

        if not candidates:
            return text

        # Replace right-to-left to preserve earlier intermediate indices
        candidates.sort(key=lambda c: c[0], reverse=True)

        for start, end, replacement in candidates:
            tracker.record_replacement(
                intermediate_start=start,
                intermediate_end=end,
                replacement=replacement,
                category="currency",
            )

        tracker.annotations.sort(key=lambda a: a.char_start)
        return tracker.current_text

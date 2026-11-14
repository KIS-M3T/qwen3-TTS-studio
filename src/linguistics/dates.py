"""Date normalization for Italian Text Normalization (Milestone M2, WP2).

Handles:
- Full Gregorian dates (DD/MM/YYYY, DD-MM-YYYY, DD.MM.YYYY)
- Abbreviated numeric dates (DD/MM, DD-MM)
- Alphanumeric dates with named months (e.g. '25 aprile 1945')
- Mandatory rule of the first of the month ('1°', '01/', '1/' -> 'primo')
- Conservative No-Op on invalid calendar dates (e.g. '31/02/2026', '29/02/2025')
- Non-interference with shielded entities (§§§SHIELD_...)
"""

import datetime
import re
from typing import List, Optional, Tuple

from src.linguistics.numbers_core import cardinal_to_italian
from src.linguistics.tracker import OffsetTracker

MONTH_NAMES = {
    1: "gennaio",
    2: "febbraio",
    3: "marzo",
    4: "aprile",
    5: "maggio",
    6: "giugno",
    7: "luglio",
    8: "agosto",
    9: "settembre",
    10: "ottobre",
    11: "novembre",
    12: "dicembre",
}

MONTH_NAME_TO_NUM = {name: num for num, name in MONTH_NAMES.items()}
MONTHS_REGEX_STR = "|".join(MONTH_NAMES.values())

# Pattern 1: Alphanumeric dates with day + month name + optional year
# e.g. "1° maggio", "1^ ottobre", "01 maggio", "25 aprile 1945", "15 marzo"
_ALPHA_DATE_PATTERN = re.compile(
    rf"\b(0?1\s*(?:°|\^|o|º)?|[0-2]?[2-9]|1\d|2\d|3[01])\s+({MONTHS_REGEX_STR})(?:\s+(\d{{1,4}}))?\b",
    re.IGNORECASE,
)

# Pattern 2: Numeric dates with 4-digit year (DD/MM/YYYY, DD-MM-YYYY, DD.MM.YYYY)
_NUMERIC_DATE_WITH_YEAR_PATTERN = re.compile(
    r"\b(\d{1,2})([/.-])(\d{1,2})\2(\d{4})\b"
)

# Pattern 3: Numeric dates without year (DD/MM, DD-MM), strictly avoiding trailing year components
_NUMERIC_DATE_NO_YEAR_PATTERN = re.compile(
    r"\b(\d{1,2})([/-])(0[1-9]|1[0-2])(?![/.-]\d)\b"
)

_SHIELD_PATTERN = re.compile(r"§§§SHIELD_[a-f0-9]{12}_\d+§§§")


def is_valid_gregorian_date(day: int, month: int, year: Optional[int] = None) -> bool:
    """Validate day/month/year against the Gregorian calendar.

    Args:
        day: Day of month (1..31).
        month: Month number (1..12).
        year: Optional year (e.g. 2026). If None, checked using leap year 2024 to allow Feb 29.

    Returns:
        True if the date is a valid Gregorian calendar date, False otherwise.
    """
    if month < 1 or month > 12 or day < 1 or day > 31:
        return False
    ref_year = year if year is not None else 2024
    try:
        datetime.date(ref_year, month, day)
        return True
    except ValueError:
        return False


class DateNormalizer:
    """Normalizes Italian date expressions with strict Gregorian calendar validation."""

    @classmethod
    def normalize_dates(cls, text: str, tracker: OffsetTracker) -> str:
        """Find and normalize all valid date expressions in intermediate text.

        Args:
            text: Current intermediate text (from tracker.current_text).
            tracker: OffsetTracker instance to register replacements.

        Returns:
            Updated intermediate text.
        """
        if not text:
            return text

        # Collect shielded entity spans to prevent matching inside them
        shield_spans = [m.span() for m in _SHIELD_PATTERN.finditer(text)]

        def is_shielded(start: int, end: int) -> bool:
            return any(s_start < end and s_end > start for s_start, s_end in shield_spans)

        # Collect all replacement candidates as (start, end, replacement_str)
        candidates: List[Tuple[int, int, str]] = []

        # 1. Numeric dates with year
        for m in _NUMERIC_DATE_WITH_YEAR_PATTERN.finditer(text):
            start, end = m.span()
            if is_shielded(start, end):
                continue
            d = int(m.group(1))
            mo = int(m.group(3))
            y = int(m.group(4))
            if not is_valid_gregorian_date(d, mo, y):
                continue  # Conservative No-Op on impossible dates

            d_str = "primo" if d == 1 else cardinal_to_italian(d)
            mo_str = MONTH_NAMES[mo]
            y_str = cardinal_to_italian(y)
            replacement = f"{d_str} {mo_str} {y_str}"
            candidates.append((start, end, replacement))

        # 2. Numeric dates without year
        for m in _NUMERIC_DATE_NO_YEAR_PATTERN.finditer(text):
            start, end = m.span()
            if is_shielded(start, end):
                continue
            # Avoid matching if part of an already matched numeric date with year
            if any(c_start <= start and end <= c_end for c_start, c_end, _ in candidates):
                continue
            d = int(m.group(1))
            mo = int(m.group(3))
            if not is_valid_gregorian_date(d, mo, None):
                continue  # Conservative No-Op on impossible dates

            d_str = "primo" if d == 1 else cardinal_to_italian(d)
            mo_str = MONTH_NAMES[mo]
            replacement = f"{d_str} {mo_str}"
            candidates.append((start, end, replacement))

        # 3. Alphanumeric dates
        for m in _ALPHA_DATE_PATTERN.finditer(text):
            start, end = m.span()
            if is_shielded(start, end):
                continue
            if any(c_start <= start and end <= c_end for c_start, c_end, _ in candidates):
                continue

            raw_day = m.group(1)
            mo_name = m.group(2).lower()
            y_raw = m.group(3)

            mo = MONTH_NAME_TO_NUM[mo_name]
            d_clean = re.sub(r"[^\d]", "", raw_day)
            d = int(d_clean)
            y = int(y_raw) if y_raw else None

            if not is_valid_gregorian_date(d, mo, y):
                continue  # Conservative No-Op

            d_str = "primo" if d == 1 else cardinal_to_italian(d)
            if y is not None:
                replacement = f"{d_str} {mo_name} {cardinal_to_italian(y)}"
            else:
                replacement = f"{d_str} {mo_name}"
            candidates.append((start, end, replacement))

        if not candidates:
            return text

        # Sort candidates descending by start index to execute right-to-left
        candidates.sort(key=lambda c: c[0], reverse=True)

        for start, end, replacement in candidates:
            tracker.record_replacement(
                intermediate_start=start,
                intermediate_end=end,
                replacement=replacement,
                category="date",
            )

        # Sort tracker annotations chronologically by raw text position
        tracker.annotations.sort(key=lambda a: a.char_start)
        return tracker.current_text

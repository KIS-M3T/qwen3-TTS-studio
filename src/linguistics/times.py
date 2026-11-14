"""Time and duration normalization for Italian Text Normalization (Milestone M2, WP2).

Handles:
- 24-hour clock times (HH:MM) with natural phonotactic contractions ('alle 12:00' -> 'a mezzogiorno', 'alle 08:00' -> 'alle otto')
- Clock times with seconds (HH:MM:SS) in confirmed clock contexts ('dodici, quindici minuti e trenta secondi')
- Chronometric durations in confirmed timing contexts ('un'ora, quindici minuti e trenta secondi')
- Conservative No-Op on ambiguous uncontextualized HH:MM:SS (preserves '12:15:30' intact)
- Conservative No-Op on impossible hours and minutes (preserves '25:70', '12:99' intact)
- Non-interference with shielded entities (§§§SHIELD_...)
"""

import re
from typing import List, Optional, Tuple

from src.linguistics.numbers_core import cardinal_to_italian
from src.linguistics.tracker import OffsetTracker

# 3-component times: HH:MM:SS (only valid 00..23, 00..59, 00..59)
_TIME_3_PATTERN = re.compile(r"\b([01]?\d|2[0-3]):([0-5]\d):([0-5]\d)\b")

# 2-component times: HH:MM with optional preceding preposition "alle "
_TIME_2_PATTERN = re.compile(
    r"\b(?:(alle)\s+)?([01]?\d|2[0-3]):([0-5]\d)\b",
    re.IGNORECASE,
)

_CLOCK_CONTEXT_PATTERN = re.compile(
    r"\b(?:orologio(?:\s+\w+)*\s+segnava\s+le|orario\s+(?:di|del)|alle|ore)\b",
    re.IGNORECASE,
)

_DURATION_CONTEXT_PATTERN = re.compile(
    r"\b(?:tempo(?:\s+\w+)*\s+cronometro\s+è\s+di|cronometro|durata(?:\s+(?:di|del))?|tempo\s+di|corsa\s+in)\b",
    re.IGNORECASE,
)

_SHIELD_PATTERN = re.compile(r"§§§SHIELD_[a-f0-9]{12}_\d+§§§")


class TimeNormalizer:
    """Normalizes Italian time expressions and durations with conservative contextual gating."""

    @classmethod
    def normalize_times(cls, text: str, tracker: OffsetTracker) -> str:
        """Find and normalize all valid time expressions in intermediate text.

        Args:
            text: Current intermediate text (from tracker.current_text).
            tracker: OffsetTracker instance to register replacements.

        Returns:
            Updated intermediate text.
        """
        if not text:
            return text

        shield_spans = [m.span() for m in _SHIELD_PATTERN.finditer(text)]

        def is_shielded(start: int, end: int) -> bool:
            return any(s_start < end and s_end > start for s_start, s_end in shield_spans)

        candidates: List[Tuple[int, int, str]] = []

        # 1. First scan for 3-component times (HH:MM:SS)
        matched_3_spans = set()
        for m in _TIME_3_PATTERN.finditer(text):
            start, end = m.span()
            if is_shielded(start, end):
                continue

            h = int(m.group(1))
            minute = int(m.group(2))
            sec = int(m.group(3))

            pre_ctx = text[max(0, start - 60):start]
            is_clock = bool(_CLOCK_CONTEXT_PATTERN.search(pre_ctx))
            is_duration = bool(_DURATION_CONTEXT_PATTERN.search(pre_ctx))

            if is_clock:
                h_str = cardinal_to_italian(h)
                m_str = cardinal_to_italian(minute)
                s_str = cardinal_to_italian(sec)
                min_word = "minuto" if minute == 1 else "minuti"
                sec_word = "secondo" if sec == 1 else "secondi"
                replacement = f"{h_str}, {m_str} {min_word} e {s_str} {sec_word}"
                candidates.append((start, end, replacement))
                matched_3_spans.add((start, end))
            elif is_duration:
                parts = []
                if h == 1:
                    parts.append("un'ora")
                elif h > 1:
                    parts.append(f"{cardinal_to_italian(h)} ore")

                if minute == 1:
                    parts.append("un minuto")
                elif minute > 1:
                    parts.append(f"{cardinal_to_italian(minute)} minuti")

                if sec == 1:
                    parts.append("un secondo")
                elif sec > 1:
                    parts.append(f"{cardinal_to_italian(sec)} secondi")

                if len(parts) == 3:
                    replacement = f"{parts[0]}, {parts[1]} e {parts[2]}"
                elif len(parts) == 2:
                    replacement = f"{parts[0]} e {parts[1]}"
                elif len(parts) == 1:
                    replacement = parts[0]
                else:
                    replacement = "zero secondi"

                candidates.append((start, end, replacement))
                matched_3_spans.add((start, end))
            else:
                # Ambiguous format in neutral context (P2): Conservative No-Op!
                # Record span so 2-component matcher doesn't match the HH:MM prefix
                matched_3_spans.add((start, end))

        # 2. Second scan for 2-component times (HH:MM)
        for m in _TIME_2_PATTERN.finditer(text):
            start, end = m.span()
            if is_shielded(start, end):
                continue

            # Ensure we do not match inside a 3-component time
            if any(s3 <= start and end <= e3 for s3, e3 in matched_3_spans):
                continue

            prep = m.group(1)
            h = int(m.group(2))
            minute = int(m.group(3))
            has_alle = bool(prep)

            if h == 12 and minute == 0:
                replacement = "a mezzogiorno" if has_alle else "mezzogiorno"
            elif h == 0 and minute == 0:
                replacement = "a mezzanotte" if has_alle else "mezzanotte"
            elif h == 1 and minute == 0:
                replacement = "all'una" if has_alle else "l'una"
            elif h == 1 and minute > 0:
                min_str = cardinal_to_italian(minute)
                replacement = f"all'una e {min_str}" if has_alle else f"l'una e {min_str}"
            elif minute == 0:
                h_str = cardinal_to_italian(h)
                replacement = f"alle {h_str}" if has_alle else h_str
            else:
                h_str = cardinal_to_italian(h)
                m_str = cardinal_to_italian(minute)
                replacement = f"alle {h_str} e {m_str}" if has_alle else f"{h_str} e {m_str}"

            candidates.append((start, end, replacement))

        if not candidates:
            return text

        # Sort descending by start index to replace right-to-left
        candidates.sort(key=lambda c: c[0], reverse=True)

        for start, end, replacement in candidates:
            tracker.record_replacement(
                intermediate_start=start,
                intermediate_end=end,
                replacement=replacement,
                category="time",
            )

        tracker.annotations.sort(key=lambda a: a.char_start)
        return tracker.current_text

"""Time and duration normalization for Italian Text Normalization (Milestone M2, WP2).

Handles:
- Atomic colon-chain tokenization to reject partial matches inside invalid HH:MM:SS (e.g. 12:15:60,
  alle 12:15:99) or multi-segment colon chains (e.g. 12:15:30:40) (WP2-B1)
- 24-hour clock times (HH:MM) with natural phonotactic contractions ('alle 12:00' -> 'a mezzogiorno',
  'alle 08:00' -> 'alle otto', 'alle 01:00' -> 'all'una')
- Grammatically anchored context gating within immediate clause boundaries for HH:MM:SS (WP2-B2):
  * Clock times with seconds in confirmed governing clock contexts ('dodici, quindici minuti e trenta secondi')
  * Chronometric durations in confirmed governing duration contexts ('un'ora, quindici minuti e trenta secondi')
  * Conservative No-Op on neutral, cross-clause, or unanchored occurrences ('12:15:30', '01:15:30')
- Conservative No-Op on impossible hours and minutes ('25:70', '12:99')
- Non-interference with shielded entities (§§§SHIELD_...)
"""

import re
from typing import List, Tuple

from src.linguistics.numbers_core import cardinal_to_italian
from src.linguistics.tracker import OffsetTracker

# Atomic colon-separated numeric chain: ensures HH:MM is never partially matched inside
# invalid HH:MM:SS (e.g. 12:15:60) or longer colon chains (e.g. 12:15:30:40) (WP2-B1).
_COLON_CHAIN_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_:])(\d+(?::\d+)+)(?![A-Za-z0-9_:])"
)

# Extracts only the immediate clause/phrase preceding a temporal token (stopping at punctuation boundaries).
_IMMEDIATE_CLAUSE_PATTERN = re.compile(r"([^.!?;,\n]*)$")

# Immediate preceding preposition "alle" governing 2-component HH:MM times
_PRECEDING_ALLE_PATTERN = re.compile(r"\b(alle)\s+$", re.IGNORECASE)

# Grammatically anchored clock context markers ending right before HH:MM:SS (WP2-B2)
_CLOCK_CONTEXT_PATTERN = re.compile(
    r"(?:"
    r"(?:^|\s)(?:alle(?:\s+ore)?|ore|verso\s+le|fino\s+alle|dalle|per\s+le)\s+"
    r"|\b(?:orologio|orario|ora|sveglia)\b(?:\s+(?:(?:da|di|del(?:l'|la|lo)?|della|sul(?:l'|la|lo)?|al(?:l'|la|lo)?)\s+\w+|esatt[oa]|locale|centrale|intern[oa])){0,2}\s+(?:segna(?:va)?|indica(?:va)?|mostra(?:va)?|batte(?:va)?)\s+(?:le|l')\s+"
    r"|\b(?:orario|ora)\b(?:\s+(?:di|del(?:l'|la|lo)?|della|esatt[ao]))?\s*(?:è\s+alle|è|:)\s+"
    r")$",
    re.IGNORECASE,
)

# Grammatically anchored chronometric duration markers ending right before HH:MM:SS (WP2-B2)
_DURATION_CONTEXT_PATTERN = re.compile(
    r"(?:"
    r"\b(?:tempo|durata|cronometro|intervallo)\b"
    r"(?:\s+(?:registrat[oa]|misurat[oa]|totale|complessiv[oa]|effettiv[oa]|finale|massim[oa]|minim[oa]|stimat[oa]|previst[oa]|del(?:l'|la|lo)?|della|da[l']?|dal|su[l']?)){0,4}"
    r"\s*(?:(?:è|era)\s+di|(?:segna(?:va)?|indica(?:va)?|riporta(?:va)?|fermat[oa]\s+a)|di|in|a|su|pari\s+a|:)\s+"
    r"|\b(?:corsa|gara|maratona|prova|tappa|completat[oa]|percors[oa]|chius[oa]|terminat[oa])\s+in\s+"
    r")$",
    re.IGNORECASE,
)

_SHIELD_PATTERN = re.compile(r"§§§SHIELD_[a-f0-9]{12}_\d+§§§")


class TimeNormalizer:
    """Normalizes Italian time expressions and durations with atomic parsing and grammatical gating."""

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

        for m in _COLON_CHAIN_PATTERN.finditer(text):
            start, end = m.span(1)
            if is_shielded(start, end):
                continue

            parts = m.group(1).split(":")

            # WP2-B1: Only exact 2-part (HH:MM) or 3-part (HH:MM:SS) colon tokens are eligible.
            # Longer chains (e.g. 12:15:30:40) are strictly preserved as No-Op.
            if len(parts) not in (2, 3):
                continue

            # Validate digit formatting: hours 1-2 digits, minutes/seconds strictly 2 digits
            if not (1 <= len(parts[0]) <= 2 and all(len(p) == 2 for p in parts[1:])):
                continue

            h = int(parts[0])
            minute = int(parts[1])
            if not (0 <= h <= 23 and 0 <= minute <= 59):
                continue  # Impossible hour/minute (e.g. 25:70, 12:99) -> No-Op

            if len(parts) == 3:
                sec = int(parts[2])
                if not (0 <= sec <= 59):
                    continue  # Impossible seconds (e.g. 12:15:60, 12:15:99) -> No-Op

                # WP2-B2: Restrict context evaluation to the immediate governing clause
                clause_match = _IMMEDIATE_CLAUSE_PATTERN.search(text[:start])
                clause = clause_match.group(1) if clause_match else text[:start]

                is_clock = bool(_CLOCK_CONTEXT_PATTERN.search(clause))
                is_duration = bool(_DURATION_CONTEXT_PATTERN.search(clause))

                if is_clock:
                    h_str = cardinal_to_italian(h)
                    m_str = cardinal_to_italian(minute)
                    s_str = cardinal_to_italian(sec)
                    min_word = "minuto" if minute == 1 else "minuti"
                    sec_word = "secondo" if sec == 1 else "secondi"
                    replacement = f"{h_str}, {m_str} {min_word} e {s_str} {sec_word}"
                    candidates.append((start, end, replacement))
                elif is_duration:
                    dur_parts = []
                    if h == 1:
                        dur_parts.append("un'ora")
                    elif h > 1:
                        dur_parts.append(f"{cardinal_to_italian(h)} ore")

                    if minute == 1:
                        dur_parts.append("un minuto")
                    elif minute > 1:
                        dur_parts.append(f"{cardinal_to_italian(minute)} minuti")

                    if sec == 1:
                        dur_parts.append("un secondo")
                    elif sec > 1:
                        dur_parts.append(f"{cardinal_to_italian(sec)} secondi")

                    if len(dur_parts) == 3:
                        replacement = f"{dur_parts[0]}, {dur_parts[1]} e {dur_parts[2]}"
                    elif len(dur_parts) == 2:
                        replacement = f"{dur_parts[0]} e {dur_parts[1]}"
                    elif len(dur_parts) == 1:
                        replacement = dur_parts[0]
                    else:
                        replacement = "zero secondi"

                    candidates.append((start, end, replacement))
                else:
                    # Ambiguous or non-governing context -> Conservative No-Op (WP2-B2)
                    continue

            else:
                # Valid 2-component time HH:MM
                alle_match = _PRECEDING_ALLE_PATTERN.search(text[:start])
                has_alle = bool(alle_match)
                replace_start = alle_match.start(1) if alle_match else start

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

                candidates.append((replace_start, end, replacement))

        if not candidates:
            return text

        # Sort descending by start index to replace right-to-left
        candidates.sort(key=lambda c: c[0], reverse=True)

        for replace_start, end, replacement in candidates:
            tracker.record_replacement(
                intermediate_start=replace_start,
                intermediate_end=end,
                replacement=replacement,
                category="time",
            )

        tracker.annotations.sort(key=lambda a: a.char_start)
        return tracker.current_text

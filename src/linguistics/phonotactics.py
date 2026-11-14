"""Phonotactic Refinement, Elisions, and Accent Normalization (Milestone M2, WP5).

Implements Stage 7 of the Italian Linguistic Normalization Pipeline (Spec Rev 2.1, § 2.7):
- Preserves native phonological apostrophes ('l\'albero', 'un\'amica') without spacing
- Preserves apocope 'qual è' and normalizes erroneous 'qual\'è' -> 'qual è'
- Normalizes grave-to-acute accents on canonical compounds ('perchè' -> 'perché', 'affinchè' -> 'affinché')
- Contextually disambiguates 'se' vs 'sé' ('Chi fa da se fa per tre.' -> 'Chi fa da sé fa per tre.')
  and high-certainty proclitic verb frames for 'da' vs 'dà' ('mi dà', 'si dà da fare')
- Enforces Decision D1.B: conservative No-Op on heterophonic homographs ('ancora', 'subito', 'principi').
"""

import re
from typing import Dict, List, Tuple

from src.linguistics.numbers import overlaps_any
from src.linguistics.tracker import OffsetTracker

_SHIELD_TOKEN_PATTERN = re.compile(r"§§§SHIELD_[a-f0-9]{12}_\d+§§§")

_GRAVE_TO_ACUTE_MAP: Dict[str, str] = {
    "perchè": "perché",
    "affinchè": "affinché",
    "poichè": "poiché",
    "benchè": "benché",
    "giacchè": "giacché",
    "sicchè": "sicché",
    "finchè": "finché",
    "cosicchè": "cosicché",
    "purchè": "purché",
    "anzichè": "anziché",
    "fuorchè": "fuorché",
    "nonchè": "nonché",
    "macchè": "macché",
    "pressochè": "pressoché",
    "alcunchè": "alcunché",
    "ventitrè": "ventitré",
    "trentatrè": "trentatré",
    "quarantatrè": "quarantatré",
    "cinquantatrè": "cinquantatré",
    "sessantatrè": "sessantatré",
    "settantatrè": "settantatré",
    "ottantatrè": "ottantatré",
    "novantatrè": "novantatré",
    "nè": "né",
    "sè": "sé",
}

_GRAVE_ALT = "|".join(re.escape(w) for w in sorted(_GRAVE_TO_ACUTE_MAP.keys(), key=len, reverse=True))
_GRAVE_ACCENT_PATTERN = re.compile(
    rf"(?<![A-Za-z0-9_À-ÖØ-öø-ÿ§])(?P<word>{_GRAVE_ALT})(?![A-Za-z0-9_À-ÖØ-öø-ÿ])",
    re.IGNORECASE,
)

_ERRONEOUS_QUAL_E_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_À-ÖØ-öø-ÿ§])(?P<word>qual)['’](?:è|é)(?![A-Za-z0-9_À-ÖØ-öø-ÿ])",
    re.IGNORECASE,
)

# Unambiguous prepositional frames requiring stressed reflexive pronoun 'sé'
_SE_PREP_PATTERN = re.compile(
    r"(?<![A-Za-zÀ-ÖØ-öø-ÿ])(?:da|tra|fra|di\s+per|fuori\s+di|dentro\s+di)[ \t]+(?P<pron>se)(?![A-Za-z0-9_À-ÖØ-öø-ÿ]|[ \t]+stess[oaie]\b|[ \t]+medesim[oaie]\b)",
    re.IGNORECASE,
)
_SE_COORD_PATTERN = re.compile(
    r"(?<![A-Za-zÀ-ÖØ-öø-ÿ])(?:tra|fra)[ \t]+s[eé][ \t]+e[ \t]+(?P<pron>se)(?![A-Za-z0-9_À-ÖØ-öø-ÿ]|[ \t]+stess[oaie]\b|[ \t]+medesim[oaie]\b)",
    re.IGNORECASE,
)
_SE_TERMINAL_PREP_PATTERN = re.compile(
    r"(?<![A-Za-zÀ-ÖØ-öø-ÿ])(?:per|in)[ \t]+(?P<pron>se)(?=[ \t]*(?:[,.;:!?]|$))",
    re.IGNORECASE,
)

# High-certainty proclitic pronoun frames where 'da' is unmistakably verb 'dà'
_DA_VERB_PATTERN = re.compile(
    r"(?<![A-Za-zÀ-ÖØ-öø-ÿ])(?P<pro>mi|ti|gli|ci|vi|si|glielo|gliela|glieli|gliele|gliene|"
    r"melo|telo|celo|velo|me[ \t]+lo|te[ \t]+lo|ce[ \t]+lo|ve[ \t]+lo|"
    r"me[ \t]+la|te[ \t]+la|ce[ \t]+la|ve[ \t]+la|"
    r"me[ \t]+ne|te[ \t]+ne|ce[ \t]+ne|ve[ \t]+ne|se[ \t]+ne)[ \t]+(?P<verb>da)"
    r"(?![A-Za-z0-9_À-ÖØ-öø-ÿ])",
    re.IGNORECASE,
)


def _preserve_case(source: str, target: str) -> str:
    """Preserve all-uppercase or initial-uppercase casing from source onto target."""
    if not source:
        return target
    if source.isupper():
        return target.upper()
    if source[0].isupper():
        return target[0].upper() + target[1:]
    return target


class PhonotacticNormalizer:
    """Performs conservative Italian phonotactic and accent refinements (Stage 7)."""

    @classmethod
    def normalize_phonotactics(
        cls,
        text: str,
        tracker: OffsetTracker,
        *,
        preserve_apostrophes: bool = True,
        normalize_accent_gravis_acutus: bool = True,
        enable_homograph_disambiguation: bool = False,
    ) -> str:
        """Apply Stage 7 accent, elision, and disambiguation rules on intermediate text."""
        if not text:
            return text

        # Decision D1.B baseline keeps heterophonic homographs untouched.
        _ = enable_homograph_disambiguation

        shield_spans = (
            [m.span() for m in _SHIELD_TOKEN_PATTERN.finditer(text)]
            if "§§§SHIELD_" in text
            else []
        )

        # Collect (start, end, replacement) keeping disjoint spans
        candidates: List[Tuple[int, int, str]] = []
        occupied: List[Tuple[int, int]] = []

        def add_candidate(start: int, end: int, replacement: str) -> None:
            if overlaps_any(start, end, shield_spans) or overlaps_any(start, end, occupied):
                return
            candidates.append((start, end, replacement))
            occupied.append((start, end))

        # 1. Erroneous elision typo 'qual\'è' -> 'qual è' (preserving native apostrophes elsewhere)
        if preserve_apostrophes and ("'" in text or "’" in text):
            if "qual" in text.lower():
                for m in _ERRONEOUS_QUAL_E_PATTERN.finditer(text):
                    start, end = m.span()
                    add_candidate(start, end, _preserve_case(m.group(0), "qual è"))

        # 2. Accent grave -> acute ('perchè' -> 'perché') and 'se'/'da' disambiguation
        if normalize_accent_gravis_acutus:
            if "è" in text or "È" in text:
                for m in _GRAVE_ACCENT_PATTERN.finditer(text):
                    start, end = m.span("word")
                    raw_word = m.group("word")
                    fixed = _GRAVE_TO_ACUTE_MAP.get(raw_word.lower())
                    if fixed is not None:
                        add_candidate(start, end, _preserve_case(raw_word, fixed))

            text_lower = text.lower()
            if " se" in text_lower or "\tse" in text_lower:
                for pattern in (_SE_PREP_PATTERN, _SE_COORD_PATTERN, _SE_TERMINAL_PREP_PATTERN):
                    for m in pattern.finditer(text):
                        start, end = m.span("pron")
                        raw_pron = m.group("pron")
                        if raw_pron.islower():
                            add_candidate(start, end, "sé")

            if " da" in text_lower or "\tda" in text_lower:
                for m in _DA_VERB_PATTERN.finditer(text):
                    raw_pro = m.group("pro")
                    raw_verb = m.group("verb")
                    if raw_pro.isupper() or not raw_verb.islower():
                        continue
                    add_candidate(m.span("verb")[0], m.span("verb")[1], "dà")

        if not candidates:
            return text

        # Apply replacements right-to-left to preserve intermediate coordinates
        candidates.sort(key=lambda c: c[0], reverse=True)
        for start, end, replacement in candidates:
            tracker.record_replacement(
                intermediate_start=start,
                intermediate_end=end,
                replacement=replacement,
                category="phonotactics",
            )

        tracker.annotations.sort(key=lambda a: a.char_start)
        return tracker.current_text

"""Titles, Common Abbreviations, and Closed-Dictionary Acronyms Normalization (Milestone M2, WP5).

Implements Stage 6 of the Italian Linguistic Normalization Pipeline (Spec Rev 2.1, § 2.6):
- Professional titles with mandatory apocope for masculine '-ore' before proper names
  ('dott. Rossi' -> 'dottor Rossi', 'il dott.' -> 'il dottore', 'dott.ssa Ferrari' -> 'dottoressa Ferrari',
   'avv. Neri' -> 'avvocato Neri')
- Common Italian abbreviations ('ecc.' -> 'eccetera', 'pag.' -> 'pagina', 'p.za' -> 'piazza')
  and context-aware 'cfr.' ('cfr. capitolo 2' -> 'confronta il capitolo due')
- Closed-dictionary acronym/initialism handling (Policy D4.B):
  lexicalized acronyms ('NATO', 'ISTAT') preserved verbatim;
  curated spelled initialisms ('CGIL' -> 'C G I L', 'INPS' -> 'I N P S', 'BTP' -> 'B T P');
  all unlisted uppercase tokens ('IKEA', 'KWH-TECH') left untouched (conservative No-Op).
"""

import json
from pathlib import Path
import re
from typing import Any, Dict, List, Set, Tuple

from src.linguistics.numbers import overlaps_any
from src.linguistics.tracker import OffsetTracker

_RESOURCES_DIR = Path(__file__).resolve().parent / "resources"


def _load_json_resource(filename: str) -> Dict[str, Any]:
    """Load a UTF-8 JSON linguistic resource bundled under src/linguistics/resources."""
    with (_RESOURCES_DIR / filename).open("r", encoding="utf-8") as fh:
        return json.load(fh)


_TITLES_DATA: Dict[str, Dict[str, Any]] = _load_json_resource("italian_titles.json")
_ABBR_DATA: Dict[str, Any] = _load_json_resource("italian_abbreviations.json")
_ACRONYMS_DATA: Dict[str, Any] = _load_json_resource("italian_acronyms.json")

_SIMPLE_ABBRS: Dict[str, Dict[str, Any]] = _ABBR_DATA["simple"]
_CFR_MASC_IL: Set[str] = set(_ABBR_DATA["cfr"]["masculine_il_targets"])
_CFR_FEM_LA: Set[str] = set(_ABBR_DATA["cfr"]["feminine_la_targets"])
_CFR_L_APOS: Set[str] = set(_ABBR_DATA["cfr"]["l_apostrophe_targets"])
_EXISTING_ARTICLES: Set[str] = {"il", "lo", "la", "l", "i", "gli", "le", "un", "una", "uno"}

_SPELLED_ACRONYMS: Dict[str, str] = dict(_ACRONYMS_DATA["spelled"])

# Longest-match alternation guarantees 'dott.ssa' beats 'dott.', 'p.zza' beats 'p.za', etc.
_TITLE_ALT = "|".join(re.escape(k) for k in sorted(_TITLES_DATA.keys(), key=len, reverse=True))
_ABBR_ALT = "|".join(re.escape(k) for k in sorted(_SIMPLE_ABBRS.keys(), key=len, reverse=True))
_SPELLED_ACRONYM_ALT = "|".join(
    re.escape(k) for k in sorted(_SPELLED_ACRONYMS.keys(), key=len, reverse=True)
)

_SHIELD_TOKEN_PATTERN = re.compile(r"§§§SHIELD_[a-f0-9]{12}_\d+§§§")
_TITLE_PATTERN = re.compile(
    rf"(?<![A-Za-z0-9_À-ÖØ-öø-ÿ./\\§])(?P<tok>{_TITLE_ALT})(?![A-Za-z0-9_À-ÖØ-öø-ÿ/\\])",
    re.IGNORECASE,
)
_SIMPLE_ABBR_PATTERN = re.compile(
    rf"(?<![A-Za-z0-9_À-ÖØ-öø-ÿ./\\§])(?P<tok>{_ABBR_ALT})(?![A-Za-z0-9_À-ÖØ-öø-ÿ/\\])",
    re.IGNORECASE,
)
_CFR_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_À-ÖØ-öø-ÿ./\\§])(?P<tok>cfr\.)(?![A-Za-z0-9_À-ÖØ-öø-ÿ/\\])",
    re.IGNORECASE,
)
_PRECEDING_CONFRONTA_VERB = re.compile(r"\b(?:si\s+veda|vedi|vedere)\s+$", re.IGNORECASE)
_NEXT_PROPER_NAME = re.compile(
    r"^[ \t]+(?!§§§SHIELD_)[A-ZÀ-ÖØ-Þ](?:['’][A-ZÀ-ÖØ-Þ][a-zà-öø-ÿ-]+|[a-zà-öø-ÿ][A-Za-zÀ-ÖØ-öø-ÿ'-]*)\b"
)
_NEXT_WORD_AFTER_SPACE = re.compile(r"^([ \t]+)(?!§§§SHIELD_)([A-Za-zÀ-ÖØ-öø-ÿ]+)")
_NEXT_SENTENCE_START = re.compile(r"^[ \t]+[A-ZÀ-ÖØ-Þ][a-zà-öø-ÿ]")
_SENTENCE_END_WS = re.compile(r"^[ \t]*(?:\r?\n|$)")

_SPELLED_ACRONYM_PATTERN = re.compile(
    rf"(?<![A-Za-z0-9_À-ÖØ-öø-ÿ./\\§-])(?P<acr>{_SPELLED_ACRONYM_ALT})(?![A-Za-z0-9_À-ÖØ-öø-ÿ/\\-]|\.[A-Za-z0-9_])"
)


def _match_case_prefix(matched: str, replacement: str) -> str:
    """Capitalize replacement when the source token begins with an uppercase letter."""
    if matched and matched[0].isupper():
        return replacement[0].upper() + replacement[1:]
    return replacement


def _should_preserve_final_dot(text: str, start: int, end: int, is_ecc_alias: bool = False) -> bool:
    """Return True when an abbreviation period doubles as a sentence-ending full stop."""
    tail = text[end:]
    if _SENTENCE_END_WS.match(tail):
        prefix = text[:start].strip()
        if not prefix:
            return False
        if is_ecc_alias:
            return True
        return prefix[0].isupper() or any(ch in prefix for ch in ".,;!?:")
    if is_ecc_alias and _NEXT_SENTENCE_START.match(tail):
        return True
    return False


def _apply_candidates(candidates: List[Tuple[int, int, str, str]], tracker: OffsetTracker) -> None:
    """Apply non-overlapping replacement candidates right-to-left through OffsetTracker."""
    if not candidates:
        return
    candidates.sort(key=lambda c: c[0], reverse=True)
    for start, end, replacement, category in candidates:
        tracker.record_replacement(
            intermediate_start=start,
            intermediate_end=end,
            replacement=replacement,
            category=category,
        )
    tracker.annotations.sort(key=lambda a: a.char_start)


class TitlesAcronymsNormalizer:
    """Normalizes Italian titles, common abbreviations, and closed-catalog acronyms (Stage 6)."""

    @classmethod
    def normalize_titles_and_acronyms(
        cls,
        text: str,
        tracker: OffsetTracker,
        *,
        enable_titles: bool = True,
        enable_abbreviations: bool = True,
        enable_acronyms: bool = True,
        strict_acronym_dictionary: bool = True,
    ) -> str:
        """Execute Stage 6 passes on intermediate text, updating tracker coordinates."""
        if not text:
            return text

        # Pass 6A: Professional titles and standard abbreviations (all contain '.')
        if (enable_titles or enable_abbreviations) and "." in text:
            cls._apply_titles_and_simple_abbreviations(
                text=text,
                tracker=tracker,
                enable_titles=enable_titles,
                enable_abbreviations=enable_abbreviations,
            )

        # Pass 6B: Contextual 'cfr.' (sees abbreviations expanded in 6A, e.g. 'cfr. cap. 2')
        curr_text = tracker.current_text
        if enable_abbreviations and "cfr." in curr_text.lower():
            cls._apply_cfr_expansion(curr_text, tracker)

        # Pass 6C: Closed-dictionary initialisms (Policy D4.B)
        curr_text = tracker.current_text
        if enable_acronyms and strict_acronym_dictionary and _SPELLED_ACRONYMS:
            if any("A" <= ch <= "Z" for ch in curr_text):
                cls._apply_spelled_acronyms(curr_text, tracker)

        return tracker.current_text

    @classmethod
    def _shield_spans(cls, text: str) -> List[Tuple[int, int]]:
        if "§§§SHIELD_" not in text:
            return []
        return [m.span() for m in _SHIELD_TOKEN_PATTERN.finditer(text)]

    @classmethod
    def _apply_titles_and_simple_abbreviations(
        cls,
        text: str,
        tracker: OffsetTracker,
        enable_titles: bool,
        enable_abbreviations: bool,
    ) -> None:
        shield_spans = cls._shield_spans(text)
        candidates: List[Tuple[int, int, str, str]] = []
        occupied: List[Tuple[int, int]] = []

        if enable_titles:
            for m in _TITLE_PATTERN.finditer(text):
                start, end = m.span()
                if overlaps_any(start, end, shield_spans):
                    continue
                raw_tok = m.group("tok")
                spec = _TITLES_DATA[raw_tok.lower()]
                apocope = spec.get("apocope")
                if apocope and _NEXT_PROPER_NAME.match(text[end:]):
                    base_word = apocope
                else:
                    base_word = spec["full"]

                expansion = _match_case_prefix(raw_tok, base_word)
                if _should_preserve_final_dot(text, start, end):
                    expansion = f"{expansion}."
                candidates.append((start, end, expansion, "title"))
                occupied.append((start, end))

        if enable_abbreviations:
            for m in _SIMPLE_ABBR_PATTERN.finditer(text):
                start, end = m.span()
                if overlaps_any(start, end, shield_spans) or overlaps_any(start, end, occupied):
                    continue
                raw_tok = m.group("tok")
                key = raw_tok.lower()
                spec = _SIMPLE_ABBRS[key]

                # Guard initials like 'N. Rossi' or all-caps codes like 'CAP.'/'ALL.'
                if spec["require_lowercase"] and not raw_tok.islower():
                    continue
                if spec["forbid_all_caps"] and raw_tok.isupper():
                    continue

                expansion = _match_case_prefix(raw_tok, spec["expansion"])
                if _should_preserve_final_dot(text, start, end, is_ecc_alias=(key in ("ecc.", "etc."))):
                    expansion = f"{expansion}."
                candidates.append((start, end, expansion, "abbreviation"))
                occupied.append((start, end))

        _apply_candidates(candidates, tracker)

    @classmethod
    def _apply_cfr_expansion(cls, text: str, tracker: OffsetTracker) -> None:
        shield_spans = cls._shield_spans(text)
        candidates: List[Tuple[int, int, str, str]] = []

        for m in _CFR_PATTERN.finditer(text):
            start, end = m.span()
            if overlaps_any(start, end, shield_spans):
                continue
            if _PRECEDING_CONFRONTA_VERB.search(text[:start]):
                continue

            raw_tok = m.group("tok")
            base_verb = _match_case_prefix(raw_tok, "confronta")
            adj_end = end
            replacement = base_verb

            m_next = _NEXT_WORD_AFTER_SPACE.match(text[end:])
            if m_next:
                space = m_next.group(1)
                next_word = m_next.group(2).lower()
                if next_word not in _EXISTING_ARTICLES:
                    if next_word in _CFR_MASC_IL:
                        replacement = f"{base_verb} il"
                    elif next_word in _CFR_FEM_LA:
                        replacement = f"{base_verb} la"
                    elif next_word in _CFR_L_APOS:
                        replacement = f"{base_verb} l'"
                        adj_end = end + len(space)

            candidates.append((start, adj_end, replacement, "abbreviation"))

        _apply_candidates(candidates, tracker)

    @classmethod
    def _apply_spelled_acronyms(cls, text: str, tracker: OffsetTracker) -> None:
        shield_spans = cls._shield_spans(text)
        candidates: List[Tuple[int, int, str, str]] = []

        for m in _SPELLED_ACRONYM_PATTERN.finditer(text):
            start, end = m.span()
            if overlaps_any(start, end, shield_spans):
                continue
            raw_acr = m.group("acr")
            spelled = _SPELLED_ACRONYMS.get(raw_acr)
            if spelled is not None:
                candidates.append((start, end, spelled, "acronym"))

        _apply_candidates(candidates, tracker)

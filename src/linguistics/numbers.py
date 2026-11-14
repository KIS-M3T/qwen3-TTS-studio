"""Numbers, Decimals, Ordinals, Centuries, Phones, Postal & Code Normalization (Milestone M2, WP4).

Implements Stage 5 of the Italian Linguistic Normalization Pipeline (Spec Rev 2.1, § 2.5):
- Certified telephone numbers with digit-by-digit reading (Decision D2.A)
- Italian 5-digit postal codes (CAP)
- Unhyphenated alphanumeric shipping/reference codes
- Roman centuries ('XXI secolo' -> 'ventunesimo secolo')
- Abbreviated centuries with apostrophe ("'900" -> "Novecento")
- Ordinals ('1°', '2ª' -> 'primo', 'seconda')
- Decimals with explicit leading/trailing zeros ('3,05', '3,50')
- Negative numbers ('-5' -> 'meno cinque')
- Cardinals with unified Treccani orthography and morphological gender agreement for '1' ('1 mela' -> 'una mela')
- Shared structural exclusion helper protecting shielded tokens, impossible dates/times,
  and rejected/malformed currency clusters from earlier stages.
"""

import re
from typing import Dict, List, Optional, Tuple

from src.linguistics.dates import (
    MONTHS_REGEX_STR,
    MONTH_NAME_TO_NUM,
    is_valid_gregorian_date,
)
from src.linguistics.numbers_core import (
    MAX_SUPPORTED_INT,
    cardinal_to_italian,
    ordinal_to_italian,
)
from src.linguistics.tracker import OffsetTracker

_MAX_SUPPORTED_INT_DIGITS = len(str(MAX_SUPPORTED_INT))

_DIGIT_WORDS = [
    "zero",
    "uno",
    "due",
    "tre",
    "quattro",
    "cinque",
    "sei",
    "sette",
    "otto",
    "nove",
]

_CENTURY_APOSTROPHE_MAP: Dict[str, str] = {
    "200": "Duecento",
    "300": "Trecento",
    "400": "Quattrocento",
    "500": "Cinquecento",
    "600": "Seicento",
    "700": "Settecento",
    "800": "Ottocento",
    "900": "Novecento",
}

# Morphological base lexicon for feminine singular agreement of bare cardinal '1' (Spec § 2.5)
_FEMININE_CONSONANT_NOUNS = frozenset({
    "banca", "banconota", "barca", "bambina", "banana", "bici", "bicicletta",
    "bolletta", "borsa", "bottiglia", "camera", "canzone", "carta", "casa",
    "cerimonia", "chiave", "cifra", "ciliegia", "classe", "colonna", "conferenza",
    "copia", "coppa", "cosa", "cucina", "data", "decisione", "delibera",
    "differenza", "domanda", "donna", "dottoressa", "fattura", "fermata", "festa",
    "figlia", "finestra", "firma", "forchetta", "foto", "fotografia", "fragola",
    "frase", "frazione", "gara", "giornata", "gomma", "lampada", "legge",
    "lettera", "lezione", "lingua", "lira", "macchina", "madre", "mamma",
    "marca", "materia", "matita", "medaglia", "mela", "misura", "moglie",
    "moneta", "moto", "multa", "nave", "nonna", "norma", "notizia", "notte",
    "pagina", "panchina", "parola", "parte", "partita", "pellicola", "penna",
    "pera", "percentuale", "persona", "pesca", "piazza", "poesia", "porta",
    "professoressa", "proposta", "prova", "qualità", "quantità", "quota",
    "ragazza", "rata", "regola", "relazione", "ricevuta", "richiesta", "riga",
    "risposta", "riunione", "sala", "scatola", "scelta", "scheda", "sconfitta",
    "scrivania", "scuola", "sede", "sedia", "seduta", "sera", "serie",
    "settimana", "sezione", "signora", "signorina", "somma", "sorella", "spesa",
    "stanza", "stazione", "sterlina", "storia", "strada", "studentessa",
    "tabella", "taglia", "tassa", "tavola", "tazza", "tazzina", "temperatura",
    "testa", "velocità", "via", "visita", "vittoria", "voce", "volta", "zia",
})

_FEMININE_VOWEL_NOUNS = frozenset({
    "agenzia", "ala", "altezza", "amica", "ampiezza", "anima", "apertura",
    "arancia", "area", "arma", "arte", "attesa", "attività", "autorità",
    "avventura", "azienda", "azione", "eccezione", "elezione", "email",
    "emergenza", "energia", "entrata", "epoca", "erba", "esigenza",
    "esperienza", "estate", "età", "idea", "inchiesta", "industria",
    "informazione", "iniziativa", "ipotesi", "iscrizione", "isola", "istruzione",
    "occasione", "offerta", "ombra", "onda", "opera", "opinione",
    "opportunità", "ora", "organizzazione", "udienza", "unità", "università",
    "urgenza", "usanza", "uscita", "uva",
})

# Standalone / prepositional words after '1' where Italian uses cardinal 'uno', not 'un'
_STANDALONE_FOLLOWERS = frozenset({
    "a", "ad", "al", "all", "alle", "anche", "che", "chi", "col", "come", "con",
    "cui", "da", "dai", "dal", "dalle", "degli", "dei", "del", "delle", "di",
    "dove", "e", "ed", "era", "fino", "fra", "ha", "in", "ma", "meno", "nei",
    "nel", "nelle", "non", "o", "od", "per", "però", "più", "quando", "se",
    "sono", "su", "sui", "sul", "sulle", "tra", "è",
})

_IMPURE_MASCULINE_START = re.compile(r"^(?:s[bcdfglmnpqrtvz]|z|gn|ps|pn|x)", re.IGNORECASE)

# Protected structural patterns that Stages 4 and 5 must NEVER alter or partially rewrite.
# Guarded with character triggers in collect_protected_spans() and word boundaries for O(N) scan.
_PROTECTED_SHIELD_PATTERN = re.compile(r"§§§SHIELD_[a-f0-9]{12}_\d+§§§")
_PROTECTED_COLON_PATTERN = re.compile(r"\b\d+(?::\d+)+\b")
_PROTECTED_SLASH_PATTERN = re.compile(r"\b\d+(?:\s*/\s*\d+)+\b")
_PROTECTED_CURR_PREFIX_PATTERN = re.compile(
    r"(?:€|\$|£|\b(?:EUR|USD|GBP)\b)\s*[-]?\s*\d[\d.,]*(?:\s*[-–—−/]\s*\d[\d.,]*)*(?:\s*(?:€|\$|£|\b(?:EUR|USD|GBP)\b|[A-Za-z%]+))?"
)
_PROTECTED_CURR_SUFFIX_PATTERN = re.compile(
    r"\b\d[\d.,]*(?:\s*[-–—−/]\s*\d[\d.,]*)*(?:\s*[-–—−/])?\s*(?:€|\$|£|\b(?:EUR|USD|GBP|EURO)\b)(?:\s*\d[\d.,]*)?[A-Za-z0-9€$£]*"
)
_PROTECTED_HYPHEN_PATTERN = re.compile(r"(?<!\+)\b[A-Za-z0-9]+(?:-[A-Za-z0-9]+)+\b")
_PROTECTED_DOT_PATTERN = re.compile(r"\b\d+(?:\.\d{1,2}|\.\d{4,})(?:\.\d+)*\b")
_PROTECTED_MALFORMED_PATTERN = re.compile(r"\b\d+(?:[.,]{2,}\d+)+\b|\b\d+,\d+(?:,\d+)+\b")
_CURRENCY_SYMBOLS_OR_CODES = ("€", "$", "£", "EUR", "USD", "GBP")

# Curated metric and physical units mapped to (singular, plural) Italian forms.
# Shared with Stage 4 (units.py) as single source of truth so Stage 5 protects
# conservative No-Op leading-zero unit quantities (e.g. '00001 kg') from CAP/cardinal rules.
# Keys are matched case-sensitively to distinguish e.g. '5 g' (grams) from '5G' (telecom).
METRIC_UNIT_SPECS: Dict[str, Tuple[str, str]] = {
    "km/h": ("chilometro orario", "chilometri orari"),
    "m/s": ("metro al secondo", "metri al secondo"),
    "km²": ("chilometro quadrato", "chilometri quadrati"),
    "km2": ("chilometro quadrato", "chilometri quadrati"),
    "m²": ("metro quadrato", "metri quadrati"),
    "m2": ("metro quadrato", "metri quadrati"),
    "cm²": ("centimetro quadrato", "centimetri quadrati"),
    "cm2": ("centimetro quadrato", "centimetri quadrati"),
    "mm²": ("millimetro quadrato", "millimetri quadrati"),
    "mm2": ("millimetro quadrato", "millimetri quadrati"),
    "m³": ("metro cubo", "metri cubi"),
    "m3": ("metro cubo", "metri cubi"),
    "cm³": ("centimetro cubo", "centimetri cubi"),
    "cm3": ("centimetro cubo", "centimetri cubi"),
    "km": ("chilometro", "chilometri"),
    "cm": ("centimetro", "centimetri"),
    "mm": ("millimetro", "millimetri"),
    "kg": ("chilogrammo", "chilogrammi"),
    "mg": ("milligrammo", "milligrammi"),
    "ml": ("millilitro", "millilitri"),
    "cl": ("centilitro", "centilitri"),
    "dl": ("decilitro", "decilitri"),
    "kWh": ("chilowattora", "chilowattora"),
    "kW": ("chilowatt", "chilowatt"),
    "MW": ("megawatt", "megawatt"),
    "GHz": ("gigahertz", "gigahertz"),
    "MHz": ("megahertz", "megahertz"),
    "kHz": ("chilohertz", "chilohertz"),
    "Hz": ("hertz", "hertz"),
    "°C": ("grado centigrado", "gradi centigradi"),
    "m": ("metro", "metri"),
    "g": ("grammo", "grammi"),
    "l": ("litro", "litri"),
    "L": ("litro", "litri"),
}

METRIC_UNIT_ALT = "|".join(re.escape(k) for k in sorted(METRIC_UNIT_SPECS.keys(), key=len, reverse=True))

_PROTECTED_ALPHA_DATE_PATTERN = re.compile(
    rf"(?<![A-Za-z0-9_])(?P<day>\d{{1,2}})\s*(?:°|\^|o|º)?\s+(?P<month>{MONTHS_REGEX_STR})(?:\s+(?P<year>\d{{1,4}}))?(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_PROTECTED_LEADING_ZERO_UNIT_PATTERN = re.compile(
    rf"(?<![A-Za-z0-9_.,/€$£–—−:-])-?\b0\d+(?:,\d+)?\s*(?:%|{METRIC_UNIT_ALT})(?![A-Za-z0-9_/²³°%]|\.[A-Za-z0-9_])"
)

# Certified telephone patterns (Policy D2.A)
_INTL_PHONE_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_+=:])(?P<plus>\+)(?:39|1|33|44|49|41|34|[1-9]\d{0,2})(?:[\s.-]?\d{2,5}){1,5}\b(?![.,]\d)"
)
_DOMESTIC_PHONE_WITH_SEP = re.compile(
    r"(?<![A-Za-z0-9_.,/€$£–—−:-])\b(?:0\d{1,3}[\s.-]\d{5,8}|3\d{2}[\s.-]\d{6,7})\b(?![.,/:-]\d)"
)
_PHONE_CONTEXT_PATTERN = re.compile(
    r"\b(?:tel(?:efono)?|cell(?:ulare)?|fax|numero|chiamare|contattare|recapito)\b[^.!?\n]{0,20}$",
    re.IGNORECASE,
)

# Postal codes (CAP)
_LEADING_ZERO_CAP_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_.,/€$£–—−:-])\b0\d{4}\b(?![A-Za-z0-9_/-]|[.,]\d|[.,][.,])"
)
_EXPLICIT_CAP_PATTERN = re.compile(
    r"(?<=\bCAP\s)\d{5}\b(?![A-Za-z0-9_/-]|[.,]\d|[.,][.,])|(?<=\bC\.A\.P\.\s)\d{5}\b(?![A-Za-z0-9_/-]|[.,]\d|[.,][.,])"
)

# Alphanumeric shipping/reference codes (e.g. AB123CD, AZ1234)
_ALPHANUM_CODE_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_.,/€$£–—−:-])(?:[A-Z]+\d+[A-Z]+[A-Z0-9]*|[A-Z]{2,}\d{2,}[A-Z0-9]*)(?![A-Za-z0-9_/-])"
)

# Roman centuries (I..XXXIX followed by secolo/secoli context)
_ROMAN_CENTURY_PATTERN = re.compile(
    r"\b(?P<rom>X{0,3}(?:IX|IV|V?I{0,3}))\b(?=\s+(?:secol[oi]\b|(?:e|ed|o|a|al|fino\s+al)\s+(?:il\s+)?X{0,3}(?:IX|IV|V?I{0,3})\s+secol[oi]\b))"
)

# Abbreviated centuries ('200..'900)
_APOSTROPHE_CENTURY_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])['’](?P<cent>[2-9]00)\b(?![A-Za-z0-9_])"
)

# Numeric ordinals (1°, 2ª, 3º, 4^)
_ORDINAL_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_.,/€$£–—−:-])(?P<num>\d{1,6})(?P<marker>[°ºª]|\^(?!\d))(?![A-Za-z0-9_])"
)

# Standard cardinals and decimals (with optional attached unary minus)
_NUMBER_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_.,/€$£–—−:-])(?P<sign>-)?(?P<int>\d{1,3}(?:\.\d{3})+|\d+)(?:,(?P<frac>\d+))?(?![A-Za-z0-9_/:°ºª%]|[.,]+\d|[.,][.,]|-(?=[A-Za-z0-9]))"
)

_NEXT_LEXICAL_WORD = re.compile(r"^[ \t]+([A-Za-zÀ-ÖØ-öø-ÿ]+)")


def collect_protected_spans(text: str) -> List[Tuple[int, int]]:
    """Collect character spans that Stage 4 and Stage 5 must leave completely untouched."""
    spans: List[Tuple[int, int]] = []
    if "§§§SHIELD_" in text:
        spans.extend(m.span() for m in _PROTECTED_SHIELD_PATTERN.finditer(text))
    if ":" in text:
        spans.extend(m.span() for m in _PROTECTED_COLON_PATTERN.finditer(text))
    if "/" in text:
        spans.extend(m.span() for m in _PROTECTED_SLASH_PATTERN.finditer(text))
    if any(tok in text for tok in _CURRENCY_SYMBOLS_OR_CODES):
        spans.extend(m.span() for m in _PROTECTED_CURR_PREFIX_PATTERN.finditer(text))
        spans.extend(m.span() for m in _PROTECTED_CURR_SUFFIX_PATTERN.finditer(text))
    if "-" in text:
        spans.extend(m.span() for m in _PROTECTED_HYPHEN_PATTERN.finditer(text))
    if "." in text:
        spans.extend(m.span() for m in _PROTECTED_DOT_PATTERN.finditer(text))
    if "." in text or "," in text:
        spans.extend(m.span() for m in _PROTECTED_MALFORMED_PATTERN.finditer(text))
    if "0" in text:
        spans.extend(m.span() for m in _PROTECTED_LEADING_ZERO_UNIT_PATTERN.finditer(text))
    text_lower = text.lower()
    if any(month in text_lower for month in MONTH_NAME_TO_NUM) and any(ch.isdigit() for ch in text):
        for m in _PROTECTED_ALPHA_DATE_PATTERN.finditer(text):
            day = int(m.group("day"))
            month = MONTH_NAME_TO_NUM[m.group("month").lower()]
            year_str = m.group("year")
            year = int(year_str) if year_str is not None else None
            if not is_valid_gregorian_date(day, month, year):
                spans.append(m.span())
    return spans


def is_leading_zero_integer(int_token: str) -> bool:
    """Return True if an integer token has redundant leading zero padding (e.g. '01', '0001')."""
    raw_digits = int_token.replace(".", "")
    return len(raw_digits) > 1 and raw_digits.startswith("0")


def is_unit_singular_one(int_token: str, frac_token: Optional[str]) -> bool:
    """Return True iff the numeric expression is unambiguously the exact integer 1."""
    if frac_token is not None or is_leading_zero_integer(int_token):
        return False
    raw_digits = int_token.replace(".", "")
    return len(raw_digits) <= _MAX_SUPPORTED_INT_DIGITS and int(raw_digits) == 1


def overlaps_any(start: int, end: int, spans: List[Tuple[int, int]]) -> bool:
    """Return True if [start, end) overlaps any span in the provided list."""
    return any(s_start < end and s_end > start for s_start, s_end in spans)


def format_decimal_italian(
    int_token: str,
    frac_token: Optional[str],
    is_negative: bool = False,
) -> Optional[str]:
    """Format an integer or Italian comma-decimal number into spoken words.

    Guards string digit lengths prior to int() to avoid ValueError on >4300-digit inputs.
    """
    int_str = int_token.replace(".", "")
    if len(int_str) > _MAX_SUPPORTED_INT_DIGITS:
        return None

    int_val = int(int_str)
    if int_val > MAX_SUPPORTED_INT:
        return None

    prefix = "meno " if is_negative else ""
    int_words = cardinal_to_italian(int_val)

    if frac_token is None:
        return f"{prefix}{int_words}"

    if len(frac_token) > _MAX_SUPPORTED_INT_DIGITS:
        return None

    leading_zeros = len(frac_token) - len(frac_token.lstrip("0"))
    if leading_zeros == len(frac_token):
        frac_words = " ".join(["zero"] * len(frac_token))
    else:
        nonzero_val = int(frac_token.lstrip("0"))
        if nonzero_val > MAX_SUPPORTED_INT:
            return None
        zero_prefix = "zero " * leading_zeros
        frac_words = f"{zero_prefix}{cardinal_to_italian(nonzero_val)}"

    return f"{prefix}{int_words} virgola {frac_words}"


def _roman_to_int(roman: str) -> int:
    """Convert a valid Roman numeral (I..XXXIX) into an integer."""
    values = {"I": 1, "V": 5, "X": 10}
    total = 0
    prev = 0
    for ch in reversed(roman):
        val = values[ch]
        if val < prev:
            total -= val
        else:
            total += val
            prev = val
    return total


def _spell_digits(raw: str, include_plus: bool = False) -> str:
    """Read digits one by one in canonical Italian (Decision D2.A)."""
    tokens = ["più"] if include_plus else []
    for ch in raw:
        if ch.isdigit():
            tokens.append(_DIGIT_WORDS[int(ch)])
    return " ".join(tokens)


def _spell_alphanumeric_code(code: str) -> str:
    """Tokenize an alphanumeric code letter-by-letter and digit-by-digit."""
    tokens = []
    for ch in code:
        if ch.isdigit():
            tokens.append(_DIGIT_WORDS[int(ch)])
        else:
            tokens.append(ch)
    return " ".join(tokens)


def _agree_cardinal_one(text: str, end: int) -> Tuple[int, str]:
    """Choose masculine/feminine agreement for standalone cardinal '1' based on following word.

    Returns (adjusted_end, replacement_word). For feminine vowel-initial nouns like '1 ora',
    consumes the intervening whitespace to emit clean elided "un'ora".
    """
    m_next = _NEXT_LEXICAL_WORD.match(text[end:])
    if not m_next:
        return end, "uno"

    next_word = m_next.group(1).lower()
    if next_word in _STANDALONE_FOLLOWERS:
        return end, "uno"
    if next_word in _FEMININE_CONSONANT_NOUNS:
        return end, "una"
    if next_word in _FEMININE_VOWEL_NOUNS:
        space_len = m_next.start(1)
        return end + space_len, "un'"
    if _IMPURE_MASCULINE_START.match(next_word):
        return end, "uno"
    return end, "un"


class NumberNormalizer:
    """Normalizes phones, postal codes, codes, centuries, ordinals, decimals, and cardinals."""

    @classmethod
    def normalize_numbers(
        cls,
        text: str,
        tracker: OffsetTracker,
        phone_mode: str = "digit_by_digit",
    ) -> str:
        """Find and normalize numeric expressions in Stage 5 intermediate text."""
        if not text:
            return text

        shield_spans = (
            [m.span() for m in _PROTECTED_SHIELD_PATTERN.finditer(text)]
            if "§§§SHIELD_" in text
            else []
        )
        protected_spans = collect_protected_spans(text)

        # Tuple: (start, end, replacement, category)
        candidates: List[Tuple[int, int, str, str]] = []
        occupied: List[Tuple[int, int]] = []

        def add_candidate(start: int, end: int, replacement: str, category: str) -> None:
            candidates.append((start, end, replacement, category))
            occupied.append((start, end))

        # 1. Certified Telephone Numbers (D2.A)
        if "+" in text:
            for m in _INTL_PHONE_PATTERN.finditer(text):
                start, end = m.span()
                if overlaps_any(start, end, shield_spans):
                    continue
                digits_only = [ch for ch in m.group(0) if ch.isdigit()]
                if 7 <= len(digits_only) <= 15:
                    add_candidate(
                        start,
                        end,
                        _spell_digits(m.group(0), include_plus=True),
                        "phone_number",
                    )

        for m in _DOMESTIC_PHONE_WITH_SEP.finditer(text):
            start, end = m.span()
            if overlaps_any(start, end, shield_spans) or overlaps_any(start, end, occupied):
                continue
            digits_only = [ch for ch in m.group(0) if ch.isdigit()]
            if 9 <= len(digits_only) <= 11 and _PHONE_CONTEXT_PATTERN.search(text[:start]):
                add_candidate(start, end, _spell_digits(m.group(0)), "phone_number")

        # 2. Italian Postal Codes (CAP)
        for pattern in (_LEADING_ZERO_CAP_PATTERN, _EXPLICIT_CAP_PATTERN):
            for m in pattern.finditer(text):
                start, end = m.span()
                if overlaps_any(start, end, protected_spans) or overlaps_any(start, end, occupied):
                    continue
                add_candidate(start, end, _spell_digits(m.group(0)), "postal_code")

        # 3. Alphanumeric Reference/Shipping Codes
        for m in _ALPHANUM_CODE_PATTERN.finditer(text):
            start, end = m.span()
            if overlaps_any(start, end, protected_spans) or overlaps_any(start, end, occupied):
                continue
            add_candidate(start, end, _spell_alphanumeric_code(m.group(0)), "alphanumeric_code")

        # 4. Roman Centuries ('XXI secolo' -> 'ventunesimo secolo')
        for m in _ROMAN_CENTURY_PATTERN.finditer(text):
            rom = m.group("rom")
            if not rom:
                continue
            start, end = m.span()
            if overlaps_any(start, end, protected_spans) or overlaps_any(start, end, occupied):
                continue
            val = _roman_to_int(rom)
            if 1 <= val <= 39:
                add_candidate(start, end, ordinal_to_italian(val, gender="m"), "century")

        # 5. Abbreviated Centuries ("'900" -> "Novecento")
        for m in _APOSTROPHE_CENTURY_PATTERN.finditer(text):
            start, end = m.span()
            if overlaps_any(start, end, protected_spans) or overlaps_any(start, end, occupied):
                continue
            replacement = _CENTURY_APOSTROPHE_MAP.get(m.group("cent"))
            if replacement is not None:
                add_candidate(start, end, replacement, "century")

        # 6. Numeric Ordinals ('1°', '2ª')
        for m in _ORDINAL_PATTERN.finditer(text):
            start, end = m.span()
            if overlaps_any(start, end, protected_spans) or overlaps_any(start, end, occupied):
                continue
            # Preserve degrees if Stage 4 units were disabled (e.g. '20°C')
            if m.group("marker") == "°" and re.match(r"^\s*[CFK]\b", text[end:]):
                continue
            num_str = m.group("num")
            if len(num_str) > _MAX_SUPPORTED_INT_DIGITS:
                continue
            val = int(num_str)
            if 1 <= val <= MAX_SUPPORTED_INT:
                gender = "f" if m.group("marker") == "ª" else "m"
                add_candidate(start, end, ordinal_to_italian(val, gender=gender), "ordinal")

        # 7. Cardinals, Negatives, and Decimals (with morphological gender agreement on '1')
        for m in _NUMBER_PATTERN.finditer(text):
            start, end = m.span()
            if overlaps_any(start, end, protected_spans) or overlaps_any(start, end, occupied):
                continue

            sign = m.group("sign")
            int_token = m.group("int")
            frac_token = m.group("frac")

            # Prevent reading leading-zero non-CAP multi-digit integers like '007' as bare cardinal '7'
            if frac_token is None and is_leading_zero_integer(int_token):
                continue

            if frac_token is None and not sign and int_token == "1":
                adj_end, one_word = _agree_cardinal_one(text, end)
                add_candidate(start, adj_end, one_word, "cardinal")
                continue

            replacement = format_decimal_italian(int_token, frac_token, bool(sign))
            if replacement is not None:
                category = "decimal" if frac_token is not None else "cardinal"
                add_candidate(start, end, replacement, category)

        if not candidates:
            return text

        # Apply replacements right-to-left to preserve earlier intermediate offsets
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

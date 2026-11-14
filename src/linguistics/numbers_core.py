"""Shared Core Number conversion module for Italian Text Normalization (Milestone M2, WP1).

Provides pure Python standard (Zero-Dependency) cardinal and ordinal integer expansion
conforming to official Italian orthography (Treccani, DOP, Accademia della Crusca) (S1, S3, P4).
Serves as shared foundation for WP2 (dates/hours) and WP3 (currencies).
"""

from typing import Literal

# Base units 0..9
_UNITS = [
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

# Teens 10..19
_TEENS = [
    "dieci",
    "undici",
    "dodici",
    "tredici",
    "quattordici",
    "quindici",
    "sedici",
    "diciassette",
    "diciotto",
    "diciannove",
]

# Tens 20, 30, ... 90
_TENS = [
    "",
    "",
    "venti",
    "trenta",
    "quaranta",
    "cinquanta",
    "sessanta",
    "settanta",
    "ottanta",
    "novanta",
]

# Base ordinals 1..10
_ORDINALS_1_TO_10_M = [
    "",
    "primo",
    "secondo",
    "terzo",
    "quarto",
    "quinto",
    "sesto",
    "settimo",
    "ottavo",
    "nono",
    "decimo",
]

_ORDINALS_1_TO_10_F = [
    "",
    "prima",
    "seconda",
    "terza",
    "quarta",
    "quinta",
    "sesta",
    "settima",
    "ottava",
    "nona",
    "decima",
]

MAX_SUPPORTED_INT = 999_999_999_999


def _cardinal_under_100(n: int) -> str:
    """Convert integer 0..99 into Italian cardinal string with phonotactic apocope and accents."""
    if n < 10:
        return _UNITS[n]
    if n < 20:
        return _TEENS[n - 10]

    ten_idx = n // 10
    unit_idx = n % 10
    ten_str = _TENS[ten_idx]

    if unit_idx == 0:
        return ten_str

    # Phonotactic truncation: tens drop trailing vowel before 'uno' (1) and 'otto' (8)
    if unit_idx in (1, 8):
        ten_str = ten_str[:-1]

    # Compounds with 'tre' require acute accent in Italian standard orthography (Treccani)
    if unit_idx == 3:
        unit_str = "tré"
    else:
        unit_str = _UNITS[unit_idx]

    return f"{ten_str}{unit_str}"


def _cardinal_under_1000(n: int) -> str:
    """Convert integer 0..999 into unified Italian cardinal string."""
    if n < 100:
        return _cardinal_under_100(n)

    hundreds_digit = n // 100
    rem = n % 100

    if hundreds_digit == 1:
        prefix = "cento"
    else:
        prefix = f"{_UNITS[hundreds_digit]}cento"

    if rem == 0:
        return prefix

    # Optional canonical elision: cento + ottanta -> centottanta
    if 80 <= rem <= 89:
        prefix = prefix[:-1]

    return f"{prefix}{_cardinal_under_100(rem)}"


def cardinal_to_italian(n: int) -> str:
    """Convert an integer up to 999,999,999,999 into standard Italian words.

    Follows unified orthographic convention (grafia unita):
    e.g. 42 -> "quarantadue", 1945 -> "millenovecentoquarantacinque",
         1250000 -> "un milione duecentocinquantamila".

    Args:
        n: Non-negative integer up to 999,999,999,999.

    Returns:
        Standard Italian cardinal string.

    Raises:
        TypeError: If n is not an integer (or is a boolean).
        ValueError: If n is negative or exceeds supported limit.
    """
    if isinstance(n, bool) or not isinstance(n, int):
        raise TypeError(f"cardinal_to_italian requires an int, got {type(n).__name__}")
    if n < 0:
        raise ValueError(f"cardinal_to_italian only accepts non-negative integers, got {n}")
    if n > MAX_SUPPORTED_INT:
        raise ValueError(f"cardinal_to_italian upper limit is {MAX_SUPPORTED_INT:,}, got {n}")

    if n == 0:
        return "zero"
    if n < 1000:
        return _cardinal_under_1000(n)

    parts = []

    # Billions (Miliardi)
    billions = n // 1_000_000_000
    n %= 1_000_000_000
    if billions == 1:
        parts.append("un miliardo")
    elif billions > 1:
        parts.append(f"{cardinal_to_italian(billions)} miliardi")

    # Millions (Milioni)
    millions = n // 1_000_000
    n %= 1_000_000
    if millions == 1:
        parts.append("un milione")
    elif millions > 1:
        parts.append(f"{cardinal_to_italian(millions)} milioni")

    # Thousands (Mila)
    thousands = n // 1_000
    n %= 1_000
    if thousands == 1:
        thousand_str = "mille"
    elif thousands > 1:
        thousand_str = f"{_cardinal_under_1000(thousands)}mila"
    else:
        thousand_str = ""

    # Remainder under 1000
    if n > 0:
        rem_str = _cardinal_under_1000(n)
    else:
        rem_str = ""

    if thousand_str and rem_str:
        parts.append(f"{thousand_str}{rem_str}")
    elif thousand_str:
        parts.append(thousand_str)
    elif rem_str:
        parts.append(rem_str)

    return " ".join(parts)


def ordinal_to_italian(n: int, gender: Literal["m", "f"] = "m") -> str:
    """Convert a positive integer into standard Italian ordinal words.

    Special support for base ordinals:
    1 -> "primo" / "prima"
    2 -> "secondo" / "seconda"
    ...
    21 -> "ventunesimo" / "ventunesima"

    Args:
        n: Positive integer (1 <= n <= 999,999,999,999).
        gender: "m" for masculine, "f" for feminine.

    Returns:
        Italian ordinal word.

    Raises:
        TypeError: If n is not an integer (or is a boolean).
        ValueError: If n < 1, n exceeds limit, or gender is invalid.
    """
    if isinstance(n, bool) or not isinstance(n, int):
        raise TypeError(f"ordinal_to_italian requires an int, got {type(n).__name__}")
    if n < 1:
        raise ValueError(f"Ordinal numbers must be >= 1, got {n}")
    if n > MAX_SUPPORTED_INT:
        raise ValueError(f"ordinal_to_italian upper limit is {MAX_SUPPORTED_INT:,}, got {n}")
    if gender not in ("m", "f"):
        raise ValueError(f"gender must be 'm' or 'f', got {gender!r}")

    if n <= 10:
        return _ORDINALS_1_TO_10_M[n] if gender == "m" else _ORDINALS_1_TO_10_F[n]

    suffix = "esimo" if gender == "m" else "esima"
    card = cardinal_to_italian(n)

    # General rule: drop trailing vowel of cardinal and append -esimo / -esima
    # Exceptions: numbers ending in 3 (tre) keep vowel: ventitreesimo
    #             numbers ending in 6 (sei) keep vowel: ventiseiesimo
    if card.endswith(("tré", "tre")):
        base = card[:-1] + "e"
        return f"{base}{suffix}"
    elif card.endswith("sei"):
        return f"{card}{suffix}"
    else:
        base = card[:-1]
        return f"{base}{suffix}"

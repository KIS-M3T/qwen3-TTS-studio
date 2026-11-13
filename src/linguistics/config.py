"""Configuration for Italian Text Normalizer (Milestone M2, WP0).

Encapsulates ratified conservative policies:
- D1.B: No-Op on heterophonic homographs (enable_homograph_disambiguation=False).
- D2.A: Digit-by-digit phone reading on certified patterns (phone_reading_mode="digit_by_digit").
- D4.B: Closed dictionary for acronyms and initialisms (strict_acronym_dictionary=True).
- D5.A: Formal monetary register (currency_register="formal").
"""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class ItalianNormalizerConfig:
    """Configuration options for ItalianTextNormalizer."""

    # D1.B: Conservative No-Op on heterophonic homographs (ancora, subito, principi)
    enable_homograph_disambiguation: bool = False

    # D2.A / D2.C: Certified phone reading mode ("digit_by_digit" vs "grouped")
    phone_reading_mode: Literal["digit_by_digit", "grouped"] = "digit_by_digit"

    # D4.B: Closed curated catalog for acronyms; unclassified uppercase tokens remain untouched
    strict_acronym_dictionary: bool = True

    # D5.A: Formal currency register (explicit zero, negative, cents declination)
    currency_register: Literal["formal", "colloquial"] = "formal"

    # Stage 1 / Stage 8: Entity shielding protection for URLs, emails, file paths, UUIDs
    enable_shielding: bool = True

    # Stage 7: Phonotactic elision & accent normalization rules
    preserve_apostrophes: bool = True
    normalize_accent_gravis_acutus: bool = True

    def validate(self) -> None:
        """Validate configuration values."""
        if self.phone_reading_mode not in ("digit_by_digit", "grouped"):
            raise ValueError(f"Invalid phone_reading_mode: {self.phone_reading_mode}")
        if self.currency_register not in ("formal", "colloquial"):
            raise ValueError(f"Invalid currency_register: {self.currency_register}")

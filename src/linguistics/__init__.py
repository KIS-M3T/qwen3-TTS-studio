"""Italian linguistics package (Text Normalization and prosodic preparation)."""

from src.linguistics.base import BaseTextNormalizer
from src.linguistics.config import ItalianNormalizerConfig
from src.linguistics.currencies import CurrencyNormalizer
from src.linguistics.dates import DateNormalizer
from src.linguistics.exceptions import (
    EntityShieldingCorruptionError,
    LinguisticNormalizationError,
)
from src.linguistics.identity import IdentityNormalizer
from src.linguistics.normalizer import ItalianTextNormalizer
from src.linguistics.numbers import NumberNormalizer
from src.linguistics.numbers_core import cardinal_to_italian, ordinal_to_italian
from src.linguistics.phonotactics import PhonotacticNormalizer
from src.linguistics.shielding import EntityShieldManager, ShieldedEntity
from src.linguistics.times import TimeNormalizer
from src.linguistics.titles import TitlesAcronymsNormalizer
from src.linguistics.tracker import OffsetTracker
from src.linguistics.types import NormalizedText, TNAnnotation
from src.linguistics.units import UnitsNormalizer

__all__ = [
    "BaseTextNormalizer",
    "IdentityNormalizer",
    "ItalianNormalizerConfig",
    "ItalianTextNormalizer",
    "LinguisticNormalizationError",
    "EntityShieldingCorruptionError",
    "EntityShieldManager",
    "ShieldedEntity",
    "OffsetTracker",
    "DateNormalizer",
    "TimeNormalizer",
    "CurrencyNormalizer",
    "UnitsNormalizer",
    "NumberNormalizer",
    "TitlesAcronymsNormalizer",
    "PhonotacticNormalizer",
    "cardinal_to_italian",
    "ordinal_to_italian",
    "NormalizedText",
    "TNAnnotation",
]

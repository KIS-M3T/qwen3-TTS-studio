"""Italian linguistics package (Text Normalization and prosodic preparation)."""

from src.linguistics.base import BaseTextNormalizer
from src.linguistics.config import ItalianNormalizerConfig
from src.linguistics.exceptions import (
    EntityShieldingCorruptionError,
    LinguisticNormalizationError,
)
from src.linguistics.identity import IdentityNormalizer
from src.linguistics.normalizer import ItalianTextNormalizer
from src.linguistics.types import NormalizedText, TNAnnotation

__all__ = [
    "BaseTextNormalizer",
    "IdentityNormalizer",
    "ItalianNormalizerConfig",
    "ItalianTextNormalizer",
    "LinguisticNormalizationError",
    "EntityShieldingCorruptionError",
    "NormalizedText",
    "TNAnnotation",
]

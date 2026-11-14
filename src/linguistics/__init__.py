"""Italian linguistics package (Text Normalization and prosodic preparation)."""

from src.linguistics.base import BaseTextNormalizer
from src.linguistics.config import ItalianNormalizerConfig
from src.linguistics.exceptions import (
    EntityShieldingCorruptionError,
    LinguisticNormalizationError,
)
from src.linguistics.identity import IdentityNormalizer
from src.linguistics.normalizer import ItalianTextNormalizer
from src.linguistics.numbers_core import cardinal_to_italian, ordinal_to_italian
from src.linguistics.shielding import EntityShieldManager, ShieldedEntity
from src.linguistics.tracker import OffsetTracker
from src.linguistics.types import NormalizedText, TNAnnotation

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
    "cardinal_to_italian",
    "ordinal_to_italian",
    "NormalizedText",
    "TNAnnotation",
]

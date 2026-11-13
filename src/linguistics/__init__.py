"""Italian linguistics package (Text Normalization and prosodic preparation)."""

from src.linguistics.base import BaseTextNormalizer
from src.linguistics.identity import IdentityNormalizer
from src.linguistics.types import NormalizedText, TNAnnotation

__all__ = [
    "BaseTextNormalizer",
    "IdentityNormalizer",
    "NormalizedText",
    "TNAnnotation",
]

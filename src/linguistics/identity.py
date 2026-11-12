"""Pass-through Identity Text Normalizer for baseline A/B contrasts (R4, WP3)."""

from src.linguistics.base import BaseTextNormalizer
from src.linguistics.types import NormalizedText


class IdentityNormalizer(BaseTextNormalizer):
    """No-op baseline normalizer returning identical raw text."""

    def normalize(self, text: str) -> NormalizedText:
        return NormalizedText(
            raw_text=text,
            normalized_text=text,
            annotations=[],
            metadata={"normalizer": "IdentityNormalizer", "modified": False},
        )

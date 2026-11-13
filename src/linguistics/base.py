"""Abstract interface for Italian Text Normalization (TN) (R4)."""

from abc import ABC, abstractmethod
from src.linguistics.types import NormalizedText


class BaseTextNormalizer(ABC):
    """Abstract protocol for text normalization prior to speech synthesis."""

    @abstractmethod
    def normalize(self, text: str) -> NormalizedText:
        """Transform raw input text into pronounced expanded phonetic form.

        Args:
            text: Raw user input text.

        Returns:
            NormalizedText: Normalized output text with token annotations.
        """
        pass

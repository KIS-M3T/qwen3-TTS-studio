"""Italian Text Normalizer skeleton and pipeline orchestrator (Milestone M2, WP0)."""

import logging
from typing import Optional

from src.linguistics.base import BaseTextNormalizer
from src.linguistics.config import ItalianNormalizerConfig
from src.linguistics.exceptions import LinguisticNormalizationError
from src.linguistics.types import NormalizedText

logger = logging.getLogger(__name__)


class ItalianTextNormalizer(BaseTextNormalizer):
    """Standard Italian Text Normalizer implementing the 8-stage pipeline contract."""

    def __init__(self, config: Optional[ItalianNormalizerConfig] = None):
        """Initialize ItalianTextNormalizer with configuration.

        Args:
            config: Optional ItalianNormalizerConfig. Defaults to conservative baseline settings.
        """
        self.config = config or ItalianNormalizerConfig()
        self.config.validate()

    def normalize(self, text: str) -> NormalizedText:
        """Transform raw input text into canonical Italian spoken orthographic form.

        Args:
            text: Raw input text.

        Returns:
            NormalizedText: Result with raw_text, normalized_text, and annotations.

        Raises:
            LinguisticNormalizationError: If unrecoverable normalization corruption occurs.
        """
        if not isinstance(text, str):
            raise TypeError(f"Expected str input, got {type(text).__name__}")

        # REQ-TXT-07: Invariance on empty input and whitespace
        if not text:
            return NormalizedText(
                raw_text="",
                normalized_text="",
                annotations=[],
                metadata={"pipeline_stage": "wp0_skeleton", "annotations_count": 0},
            )

        if text.strip() == "":
            return NormalizedText(
                raw_text=text,
                normalized_text=text,
                annotations=[],
                metadata={"pipeline_stage": "wp0_skeleton", "annotations_count": 0},
            )

        # WP0 skeleton: pass-through baseline establishing interface contract
        # Pipeline stages 1-8 will be incrementally integrated across WP1-WP5
        return NormalizedText(
            raw_text=text,
            normalized_text=text,
            annotations=[],
            metadata={"pipeline_stage": "wp0_skeleton", "annotations_count": 0},
        )

"""Italian Text Normalizer orchestrating the 8-stage pipeline (Milestone M2, WP1)."""

import logging
from typing import Optional

from src.linguistics.base import BaseTextNormalizer
from src.linguistics.config import ItalianNormalizerConfig
from src.linguistics.exceptions import LinguisticNormalizationError
from src.linguistics.shielding import EntityShieldManager
from src.linguistics.tracker import OffsetTracker
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

        Executes the 8-stage pipeline:
        Stage 1: Entity Shielding (isolating URLs, emails, paths, UUIDs, technical codes)
        Stages 2-7: Linguistic Transformations (WP2-WP5 modules)
        Stage 8: Entity Deshielding & Validation (faithfully restoring shielded tokens)

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
                metadata={"pipeline_stage": "wp1_shielding", "annotations_count": 0},
            )

        if text.strip() == "":
            return NormalizedText(
                raw_text=text,
                normalized_text=text,
                annotations=[],
                metadata={"pipeline_stage": "wp1_shielding", "annotations_count": 0},
            )

        # Stage 1: Entity Shielding
        if self.config.enable_shielding:
            shielded_text, shielded_entities, session_nonce = EntityShieldManager.shield(text)
        else:
            shielded_text, shielded_entities, session_nonce = text, [], ""

        # Initialize tracker with original raw_text and synchronize with shielded state
        tracker = OffsetTracker(text)
        if shielded_entities:
            tracker.apply_shielding(shielded_entities)

        # Intermediate Stages 2-7 will execute on shielded_text in WP2-WP5.
        # In WP1, intermediate_text is tracker.current_text (aligned with shielded_text).
        intermediate_text = tracker.current_text

        # Stage 8: Entity Deshielding & Validation (WP1-B2)
        if shielded_entities:
            final_text = EntityShieldManager.deshield(
                intermediate_text, shielded_entities, session_nonce=session_nonce
            )
            for entity in shielded_entities:
                tracker.deshield_entity(entity.placeholder, entity.original_text)
        else:
            final_text = intermediate_text

        return NormalizedText(
            raw_text=text,
            normalized_text=final_text,
            annotations=tracker.annotations,
            metadata={
                "pipeline_stage": "wp1_shielding",
                "shielded_entities_count": len(shielded_entities),
                "annotations_count": len(tracker.annotations),
            },
        )

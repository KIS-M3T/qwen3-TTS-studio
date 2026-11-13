"""A/B testing harness for Italian Text Normalization (TN) evaluation (WP3 - R4, C7)."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import pytest

from src.engine.types import AudioBuffer, DecodingParams
from src.linguistics import BaseTextNormalizer, IdentityNormalizer, NormalizedText, TNAnnotation


@dataclass(frozen=True)
class ABComparisonResult:
    """Differential evaluation report between Condition A (raw) and Condition B (normalized)."""
    test_id: str
    condition_a_raw: str
    condition_b_norm: str
    duration_sec_a: float
    duration_sec_b: float
    delta_duration_sec: float
    asr_diagnostic_transcript_a: Optional[str]
    asr_diagnostic_transcript_b: Optional[str]
    human_evaluation_rubric: Dict[str, Optional[int]]


class ABTestingHarness:
    """Harness to coordinate pairwise A/B generation and differential reporting."""

    HUMAN_EVAL_CRITERIA = [
        "general_intelligibility",
        "numbers_dates_accuracy",
        "acronyms_pronunciation",
        "prosody_and_pauses",
        "absence_of_truncation",
        "absence_of_artifacts_hallucinations",
        "voice_naturalness",
    ]

    def __init__(self, normalizer: BaseTextNormalizer):
        self.normalizer = normalizer

    def evaluate_pair(
        self,
        test_id: str,
        raw_text: str,
        audio_a: AudioBuffer,
        audio_b: AudioBuffer,
        asr_a: Optional[str] = None,
        asr_b: Optional[str] = None,
    ) -> ABComparisonResult:
        """Produce pairwise differential report."""
        norm_result = self.normalizer.normalize(raw_text)

        delta_dur = round(audio_b.duration_sec - audio_a.duration_sec, 4)

        rubric = {criterion: None for criterion in self.HUMAN_EVAL_CRITERIA}

        return ABComparisonResult(
            test_id=test_id,
            condition_a_raw=raw_text,
            condition_b_norm=norm_result.normalized_text,
            duration_sec_a=round(audio_a.duration_sec, 4),
            duration_sec_b=round(audio_b.duration_sec, 4),
            delta_duration_sec=delta_dur,
            asr_diagnostic_transcript_a=asr_a,
            asr_diagnostic_transcript_b=asr_b,
            human_evaluation_rubric=rubric,
        )


def test_identity_normalizer_contract():
    """Verify IdentityNormalizer preserves input text unchanged."""
    normalizer = IdentityNormalizer()
    res = normalizer.normalize("Il 15/05 alle 14:30")
    assert res.raw_text == "Il 15/05 alle 14:30"
    assert res.normalized_text == "Il 15/05 alle 14:30"
    assert len(res.annotations) == 0
    assert res.metadata["modified"] is False


def test_ab_harness_differential_reporting():
    """Verify ABTestingHarness produces compliant pairwise differential report."""
    normalizer = IdentityNormalizer()
    harness = ABTestingHarness(normalizer)

    import numpy as np

    # Condition A: 1.0s audio
    audio_a = AudioBuffer(samples=np.zeros(24000, dtype=np.float32), sample_rate=24000)
    # Condition B: 1.25s audio
    audio_b = AudioBuffer(samples=np.zeros(30000, dtype=np.float32), sample_rate=24000)

    report = harness.evaluate_pair(
        test_id="NUM_01",
        raw_text="Il treno arriva alle 14:30.",
        audio_a=audio_a,
        audio_b=audio_b,
        asr_a="il treno arriva alle quattordici e trenta",
        asr_b="il treno arriva alle quattordici e trenta",
    )

    assert report.test_id == "NUM_01"
    assert report.delta_duration_sec == pytest.approx(0.25)
    assert len(report.human_evaluation_rubric) == 7
    assert "absence_of_truncation" in report.human_evaluation_rubric
    assert "numbers_dates_accuracy" in report.human_evaluation_rubric

"""Unit tests for Milestone M2 Workpackage WP0 (Skeleton, Types, and Configuration)."""

import time
import pytest
from src.linguistics import (
    BaseTextNormalizer,
    IdentityNormalizer,
    ItalianNormalizerConfig,
    ItalianTextNormalizer,
    LinguisticNormalizationError,
    EntityShieldingCorruptionError,
    NormalizedText,
    TNAnnotation,
)


class TestWP0SkeletonAndContracts:
    """Verifies interface contracts, configuration defaults, and boundary conditions for WP0."""

    def test_normalizer_interface_inheritance(self):
        """Verify ItalianTextNormalizer adheres to BaseTextNormalizer abstract protocol."""
        normalizer = ItalianTextNormalizer()
        assert isinstance(normalizer, BaseTextNormalizer)

    def test_config_ratified_conservative_defaults(self):
        """Verify default configuration values strictly match ratified decisions D1.B, D2.A, D4.B, D5.A."""
        cfg = ItalianNormalizerConfig()
        # D1.B: Conservative No-Op on homographs
        assert cfg.enable_homograph_disambiguation is False
        # D2.A: Certified phone reading mode
        assert cfg.phone_reading_mode == "digit_by_digit"
        # D4.B: Strict closed catalog for acronyms
        assert cfg.strict_acronym_dictionary is True
        # D5.A: Formal currency register
        assert cfg.currency_register == "formal"
        # Shielding enabled
        assert cfg.enable_shielding is True
        # Phonotactic preservation
        assert cfg.preserve_apostrophes is True
        assert cfg.normalize_accent_gravis_acutus is True

    def test_config_validation_invalid_modes(self):
        """Verify configuration validation rejects invalid options."""
        with pytest.raises(ValueError, match="Invalid phone_reading_mode"):
            ItalianNormalizerConfig(phone_reading_mode="invalid_mode").validate()

        with pytest.raises(ValueError, match="Invalid currency_register"):
            ItalianNormalizerConfig(currency_register="slang").validate()

    def test_exceptions_hierarchy(self):
        """Verify domain exception hierarchy."""
        assert issubclass(LinguisticNormalizationError, Exception)
        assert issubclass(EntityShieldingCorruptionError, LinguisticNormalizationError)

        err = EntityShieldingCorruptionError("Placeholder corruption detected")
        assert isinstance(err, LinguisticNormalizationError)
        assert isinstance(err, Exception)

    def test_empty_input_invariance(self):
        """Verify REQ-TXT-07: empty string returns empty NormalizedText without error."""
        normalizer = ItalianTextNormalizer()
        res = normalizer.normalize("")
        assert isinstance(res, NormalizedText)
        assert res.raw_text == ""
        assert res.normalized_text == ""
        assert res.annotations == []

    def test_whitespace_only_invariance(self):
        """Verify REQ-TXT-07: whitespace-only string returns unchanged text without error."""
        normalizer = ItalianTextNormalizer()
        raw = "   \n\t  "
        res = normalizer.normalize(raw)
        assert isinstance(res, NormalizedText)
        assert res.raw_text == raw
        assert res.normalized_text == raw
        assert res.annotations == []

    def test_type_validation_rejects_non_str(self):
        """Verify normalizer raises TypeError on non-string inputs."""
        normalizer = ItalianTextNormalizer()
        with pytest.raises(TypeError, match="Expected str input"):
            normalizer.normalize(None)  # type: ignore

        with pytest.raises(TypeError, match="Expected str input"):
            normalizer.normalize(12345)  # type: ignore

    def test_normalized_text_field_names(self):
        """Verify NormalizedText strictly uses raw_text and normalized_text (contract P1)."""
        nt = NormalizedText(
            raw_text="ciao",
            normalized_text="ciao",
            annotations=[],
            metadata={"test": True},
        )
        assert hasattr(nt, "raw_text")
        assert hasattr(nt, "normalized_text")
        assert hasattr(nt, "annotations")
        assert hasattr(nt, "metadata")
        assert nt.raw_text == "ciao"
        assert nt.normalized_text == "ciao"

    def test_wp0_skeleton_latency_baseline(self):
        """Verify WP0 skeleton latency is well within < 1.0 ms budget across 1000 iterations."""
        normalizer = ItalianTextNormalizer()
        sample = "Il dott. Rossi ha versato 1.250,50 € in data 01/05/2026."

        times = []
        for _ in range(1000):
            t0 = time.perf_counter()
            _ = normalizer.normalize(sample)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)

        times.sort()
        p50 = times[500]
        p95 = times[950]

        # In WP0 skeleton, latency should be < 0.1 ms
        assert p50 < 0.5, f"p50 latency too high: {p50:.4f} ms"
        assert p95 < 1.0, f"p95 latency too high: {p95:.4f} ms"

"""Unit tests for AudioBuffer contract validation, range checks, and immutability (F2)."""

import numpy as np
import pytest

from src.engine.types import AudioBuffer


def test_valid_audio_buffer():
    """Verify standard valid float32 1D array in [-1.0, 1.0] succeeds."""
    samples = np.array([-0.5, 0.0, 0.5, 0.8], dtype=np.float32)
    buf = AudioBuffer(samples=samples, sample_rate=24000)
    assert buf.sample_rate == 24000
    assert len(buf.samples) == 4
    assert abs(buf.duration_sec - (4.0 / 24000.0)) < 1e-4


def test_reject_out_of_range_positive():
    """F2: Verify sample > +1.0 raises ValueError."""
    samples = np.array([0.0, 0.5, 1.2], dtype=np.float32)
    with pytest.raises(ValueError, match="Audio samples outside normalized range"):
        AudioBuffer(samples=samples, sample_rate=24000)


def test_reject_out_of_range_negative():
    """F2: Verify sample < -1.0 raises ValueError."""
    samples = np.array([-1.2, 0.0, 0.5], dtype=np.float32)
    with pytest.raises(ValueError, match="Audio samples outside normalized range"):
        AudioBuffer(samples=samples, sample_rate=24000)


def test_reject_nan():
    """F2: Verify NaN sample raises ValueError."""
    samples = np.array([0.0, np.nan, 0.5], dtype=np.float32)
    with pytest.raises(ValueError, match="Audio samples contain non-finite values"):
        AudioBuffer(samples=samples, sample_rate=24000)


def test_reject_inf():
    """F2: Verify Inf sample raises ValueError."""
    samples = np.array([0.0, np.inf, 0.5], dtype=np.float32)
    with pytest.raises(ValueError, match="Audio samples contain non-finite values"):
        AudioBuffer(samples=samples, sample_rate=24000)


def test_reject_empty_array():
    """F2: Verify empty array (size == 0) raises ValueError."""
    samples = np.array([], dtype=np.float32)
    with pytest.raises(ValueError, match="samples cannot be an empty array"):
        AudioBuffer(samples=samples, sample_rate=24000)


def test_reject_pcm_int16():
    """F2: Verify unnormalized int16 raises TypeError with conversion guidance."""
    samples = np.array([0, 1000, 32000], dtype=np.int16)
    with pytest.raises(TypeError, match="samples must have dtype float32 normalized in"):
        AudioBuffer(samples=samples, sample_rate=24000)


def test_reject_multi_channel_2d():
    """Verify multi-channel 2D audio raises ValueError (requires 1D mono)."""
    samples = np.zeros((2, 100), dtype=np.float32)
    with pytest.raises(ValueError, match="samples must be 1D mono"):
        AudioBuffer(samples=samples, sample_rate=24000)


def test_reject_non_positive_sample_rate():
    """Verify non-positive sample rate raises ValueError."""
    samples = np.zeros(100, dtype=np.float32)
    with pytest.raises(ValueError, match="sample_rate must be a positive integer"):
        AudioBuffer(samples=samples, sample_rate=0)


def test_coherence_duration_mismatch():
    """Verify duration mismatch > 1ms raises ValueError."""
    samples = np.zeros(24000, dtype=np.float32)  # exactly 1.0s
    with pytest.raises(ValueError, match="Duration mismatch"):
        AudioBuffer(samples=samples, sample_rate=24000, duration_sec=2.5)


def test_defensive_copy_isolation_and_readonly():
    """F2: Verify defensive copy isolates buffer from caller and marks read-only."""
    orig = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    buf = AudioBuffer(samples=orig, sample_rate=24000)

    # Mutate caller array
    orig[0] = 0.9
    assert buf.samples[0] == pytest.approx(0.1)  # Buffer unchanged!

    # Attempt direct mutation on buffer array
    with pytest.raises(ValueError, match="read-only"):
        buf.samples[0] = 0.5

"""Unit tests for acoustic metrics, dynamic range, DC offset, and trailing silence detection (B7, G6)."""

import numpy as np
import pytest

from src.audio.metrics import analyze_acoustic_quality, detect_trailing_silence


def test_analyze_acoustic_quality_clean():
    """Verify acoustic quality analysis on non-clipping signal."""
    samples = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 24000)).astype(np.float32) * 0.5
    metrics = analyze_acoustic_quality(samples, 24000)
    assert metrics["duration_sec"] == 1.0
    assert metrics["is_clipped"] is False
    assert metrics["peak_dbfs"] < 0.0
    assert abs(metrics["dc_offset"]) < 0.001
    assert "crest_factor_db" in metrics
    assert "dynamic_range_db" in metrics


def test_analyze_acoustic_quality_dc_offset_and_crest_factor():
    """G6: Verify DC offset and crest factor are calculated accurately and distinguished."""
    # 1. Sine wave with explicit +0.1 DC bias
    pure_sine = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 24000)).astype(np.float32) * 0.4
    offset_sine = pure_sine + 0.1
    metrics_offset = analyze_acoustic_quality(offset_sine, 24000)
    assert round(metrics_offset["dc_offset"], 2) == 0.10

    # 2. Modulated signal (active audio followed by quieter region)
    active = np.sin(2 * np.pi * 440 * np.linspace(0, 0.5, 12000)).astype(np.float32) * 0.8
    quiet = np.sin(2 * np.pi * 440 * np.linspace(0, 0.5, 12000)).astype(np.float32) * 0.05
    varying_audio = np.concatenate([active, quiet])

    metrics_varying = analyze_acoustic_quality(varying_audio, 24000)
    assert metrics_varying["crest_factor_db"] > 0.0
    # On non-stationary audio, crest factor (peak/overall RMS) and dynamic range (peak/noise floor) are clearly distinct
    assert metrics_varying["crest_factor_db"] != metrics_varying["dynamic_range_db"]
    assert metrics_varying["dynamic_range_db"] > metrics_varying["crest_factor_db"]


def test_analyze_acoustic_quality_clipped():
    """Verify clipping indicator flags when peak reaches >= 0.999."""
    samples = np.array([0.0, 1.0, 0.5], dtype=np.float32)
    metrics = analyze_acoustic_quality(samples, 24000)
    assert metrics["is_clipped"] is True


def test_detect_trailing_silence_no_truncation():
    """Verify clean signal is not flagged as truncated."""
    samples = np.sin(2 * np.pi * 440 * np.linspace(0, 3, 72000)).astype(np.float32) * 0.5
    is_truncated, ratio = detect_trailing_silence(samples, 24000)
    assert is_truncated is False
    assert ratio == 0.0


def test_detect_trailing_silence_flagged():
    """Verify signal followed by > 2s of silence is correctly flagged as truncated."""
    speech = np.sin(2 * np.pi * 440 * np.linspace(0, 1.5, 36000)).astype(np.float32) * 0.5
    silence = np.zeros(24000 * 3, dtype=np.float32)  # 3.0s of silence
    full_audio = np.concatenate([speech, silence])

    is_truncated, ratio = detect_trailing_silence(full_audio, 24000)
    assert is_truncated is True
    assert ratio > 0.5

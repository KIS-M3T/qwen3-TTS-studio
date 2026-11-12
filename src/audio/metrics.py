"""Acoustic signal metrics and diagnostic truncation detection (B7)."""

from typing import Dict, Tuple
import numpy as np


def detect_trailing_silence(
    audio: np.ndarray,
    sr: int,
    segment_sec: float = 0.5,
    min_speech_segments: int = 2,
    min_trailing_silent_sec: float = 2.0,
) -> Tuple[bool, float]:
    """Diagnostic detector of audio truncation: speech followed by sustained trailing silence (B7).

    Note: This is a diagnostic detector; it does NOT trim audio.

    Args:
        audio: Audio array float32 normalized in [-1.0, 1.0].
        sr: Sample rate in Hz.
        segment_sec: Window size in seconds.
        min_speech_segments: Minimum speech segments required at start.
        min_trailing_silent_sec: Minimum trailing silence duration to flag as truncated.

    Returns:
        Tuple[bool, float]: (is_truncated, trailing_silence_ratio)
    """
    segment_samples = int(sr * segment_sec)
    min_trailing_segments = int(min_trailing_silent_sec / segment_sec)

    if len(audio) < segment_samples * (min_speech_segments + min_trailing_segments):
        return False, 0.0

    num_segments = len(audio) // segment_samples
    segment_rms = np.array(
        [
            np.sqrt(np.mean(audio[i * segment_samples : (i + 1) * segment_samples] ** 2))
            for i in range(num_segments)
        ]
    )

    if len(segment_rms) == 0:
        return False, 0.0

    p95_rms = float(np.percentile(segment_rms, 95))
    adaptive_threshold = max(0.001, 0.02 * p95_rms)

    trailing_silent_count = 0
    for rms in reversed(segment_rms):
        if rms < adaptive_threshold:
            trailing_silent_count += 1
        else:
            break

    trailing_sec = trailing_silent_count * segment_sec
    is_truncated = trailing_sec >= min_trailing_silent_sec
    ratio = trailing_sec / (len(audio) / float(sr))

    return is_truncated, round(ratio, 4)


def analyze_acoustic_quality(audio: np.ndarray, sr: int) -> Dict[str, float]:
    """Calculate RMS, peak, dynamic range, and digital clipping indicators."""
    if len(audio) == 0:
        return {
            "duration_sec": 0.0,
            "peak_dbfs": -100.0,
            "rms_dbfs": -100.0,
            "is_clipped": False,
        }

    duration_sec = round(len(audio) / float(sr), 4)
    peak_linear = float(np.max(np.abs(audio)))
    rms_linear = float(np.sqrt(np.mean(audio**2)))

    peak_dbfs = round(20.0 * np.log10(max(1e-5, peak_linear)), 2)
    rms_dbfs = round(20.0 * np.log10(max(1e-5, rms_linear)), 2)
    is_clipped = bool(peak_linear >= 0.999)

    return {
        "duration_sec": duration_sec,
        "peak_dbfs": peak_dbfs,
        "rms_dbfs": rms_dbfs,
        "is_clipped": is_clipped,
    }

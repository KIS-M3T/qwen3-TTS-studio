"""Audio signal processing and acoustic metrics package."""

from src.audio.metrics import analyze_acoustic_quality, detect_trailing_silence

__all__ = [
    "analyze_acoustic_quality",
    "detect_trailing_silence",
]

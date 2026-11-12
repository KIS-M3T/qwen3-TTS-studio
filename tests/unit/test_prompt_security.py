"""Unit tests for voice prompt security and rejection of untrusted pickle (F1)."""

from pathlib import Path
import tempfile
import pytest

from src.engine.errors import SecurityError
from src.engine.prompt_manager import load_voice_prompt, validate_voice_directory


def test_untrusted_legacy_pickle_blocked_by_default():
    """F1: Verify legacy .pkl files are blocked by default with SecurityError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)
        voice_dir = base_dir / "untrusted_voice"
        voice_dir.mkdir()
        pkl_file = voice_dir / "prompt.pkl"
        pkl_file.write_bytes(b"dummy pickle payload")

        with pytest.raises(SecurityError, match="blocked by default"):
            load_voice_prompt(voice_dir=voice_dir, allow_legacy_pickle=False, base_dir=base_dir)


def test_path_traversal_rejected():
    """Verify directory traversal outside base directory raises SecurityError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir) / "voices"
        base_dir.mkdir()
        outside_dir = Path(tmpdir) / "outside"
        outside_dir.mkdir()

        with pytest.raises(SecurityError, match="Path traversal detected"):
            validate_voice_directory(outside_dir, base_dir)

"""Unit tests for voice prompt security, safetensors serialization, and rejection of untrusted pickle (F1, G2)."""

from pathlib import Path
import tempfile
import pytest
import torch

from qwen_tts.inference.qwen3_tts_model import VoiceClonePromptItem
from src.engine.errors import SecurityError
from src.engine.prompt_manager import (
    load_voice_prompt,
    load_voice_prompt_items,
    save_voice_prompt_items,
    validate_voice_directory,
    validate_voice_name,
)


def test_untrusted_legacy_pickle_blocked():
    """F1, G2: Verify legacy .pkl files are blocked with SecurityError and no runtime bypass."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)
        voice_dir = base_dir / "untrusted_voice"
        voice_dir.mkdir()
        pkl_file = voice_dir / "prompt.pkl"
        pkl_file.write_bytes(b"dummy pickle payload")

        with pytest.raises(SecurityError, match="blocked by default"):
            load_voice_prompt_items(voice_dir=voice_dir, base_dir=base_dir)


def test_path_traversal_rejected():
    """Verify directory traversal outside base directory raises SecurityError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir) / "voices"
        base_dir.mkdir()
        outside_dir = Path(tmpdir) / "outside"
        outside_dir.mkdir()

        with pytest.raises(SecurityError, match="Path traversal detected"):
            validate_voice_directory(outside_dir, base_dir)


def test_validate_voice_name_invalid():
    """G2: Verify invalid voice identifiers are rejected prior to filesystem operations."""
    with pytest.raises(SecurityError):
        validate_voice_name("../traversal")

    with pytest.raises(SecurityError):
        validate_voice_name("")

    with pytest.raises(SecurityError):
        validate_voice_name("..")

    # Valid voice names should pass
    assert validate_voice_name("marco_voce-01") == "marco_voce-01"


def test_save_and_load_voice_prompt_items_multisample():
    """G2: Verify multi-sample VoiceClonePromptItem serialization and reconstruction roundtrip."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)

        # Create 2 mock VoiceClonePromptItem instances
        item1 = VoiceClonePromptItem(
            ref_code=torch.tensor([1, 2, 3, 4], dtype=torch.int32),
            ref_spk_embedding=torch.randn(192, dtype=torch.float32),
            x_vector_only_mode=False,
            icl_mode=True,
            ref_text="Primo testo di prova",
        )
        item2 = VoiceClonePromptItem(
            ref_code=torch.tensor([5, 6, 7, 8], dtype=torch.int32),
            ref_spk_embedding=torch.randn(192, dtype=torch.float32),
            x_vector_only_mode=False,
            icl_mode=True,
            ref_text="Secondo testo di prova",
        )

        saved_dir = save_voice_prompt_items(
            items=[item1, item2],
            voice_id="speaker_multi",
            base_dir=base_dir,
            description="Voce multi-sample per test G2",
        )

        assert (saved_dir / "voice_prompt.safetensors").exists()
        assert (saved_dir / "metadata.json").exists()

        # Reconstruct items
        loaded_items = load_voice_prompt_items(saved_dir, base_dir)
        assert len(loaded_items) == 2

        # Verify item 0
        assert loaded_items[0].x_vector_only_mode is False
        assert loaded_items[0].icl_mode is True
        assert loaded_items[0].ref_text == "Primo testo di prova"
        assert torch.equal(loaded_items[0].ref_code, item1.ref_code)
        assert torch.allclose(loaded_items[0].ref_spk_embedding, item1.ref_spk_embedding)

        # Verify item 1
        assert loaded_items[1].ref_text == "Secondo testo di prova"
        assert torch.equal(loaded_items[1].ref_code, item2.ref_code)
        assert torch.allclose(loaded_items[1].ref_spk_embedding, item2.ref_spk_embedding)


def test_save_and_load_x_vector_only_mode():
    """G2: Verify x-vector-only mode where ref_code is None."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)

        item = VoiceClonePromptItem(
            ref_code=None,
            ref_spk_embedding=torch.randn(192, dtype=torch.float32),
            x_vector_only_mode=True,
            icl_mode=False,
            ref_text=None,
        )

        saved_dir = save_voice_prompt_items(
            items=item,
            voice_id="speaker_xvec",
            base_dir=base_dir,
        )

        loaded_items = load_voice_prompt_items(saved_dir, base_dir)
        assert len(loaded_items) == 1
        assert loaded_items[0].x_vector_only_mode is True
        assert loaded_items[0].icl_mode is False
        assert loaded_items[0].ref_code is None
        assert torch.allclose(loaded_items[0].ref_spk_embedding, item.ref_spk_embedding)

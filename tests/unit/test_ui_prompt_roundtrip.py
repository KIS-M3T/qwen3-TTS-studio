"""Tests for UI to Engine prompt serialization and deserialization compatibility (H2, J3).

Verifies roundtrip compatibility between UI handlers (save_prompt_to_safetensors_file,
save_cloned_voice_multi, save_cloned_voice) and engine loader (load_voice_prompt_items),
and enforces strict validation invariants via PromptValidationError on corrupted schemas.
"""

import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from src.engine.errors import PromptValidationError
from src.engine.prompt_manager import (
    load_voice_prompt_items,
    save_voice_prompt_items,
    save_prompt_to_safetensors_file,
)
from qwen_tts.inference.qwen3_tts_model import VoiceClonePromptItem
import qwen_tts_ui


def test_ui_multi_sample_prompt_roundtrip_no_index_field():
    """H2: Verify that prompts saved by UI without 'index' field or with UI-specific keys load without KeyError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)
        voice_dir = base_dir / "ui_voice_01"
        voice_dir.mkdir(parents=True, exist_ok=True)

        # 1. UI saves safetensors with sample_0 and sample_1
        spk0 = torch.randn(192, dtype=torch.float32)
        code0 = torch.tensor([10, 20, 30], dtype=torch.int32)
        spk1 = torch.randn(192, dtype=torch.float32)
        code1 = torch.tensor([40, 50, 60], dtype=torch.int32)

        tensors = {
            "sample_0_ref_spk_embedding": spk0,
            "sample_0_ref_code": code0,
            "sample_1_ref_spk_embedding": spk1,
            "sample_1_ref_code": code1,
        }
        save_file(tensors, str(voice_dir / "voice_prompt.safetensors"))

        # 2. UI metadata.json with descriptive UI fields (NO 'index', 'transcript' instead of 'ref_text')
        ui_metadata = {
            "name": "ui_voice_01",
            "description": "Saved from UI multi-sample tab",
            "style_note": "Natural",
            "ref_text": "Trascrizione primaria",
            "model": "1.7B-Base",
            "multi_sample": True,
            "num_samples": 2,
            "samples": [
                {
                    "path": "ref_audios/sample_00_audio1.wav",
                    "duration": 3.45,
                    "transcript": "Primo audio di test UI",
                    "snr_estimate": 24.5,
                    "is_primary": True,
                    "weight": 1.0,
                },
                {
                    "path": "ref_audios/sample_01_audio2.wav",
                    "duration": 2.80,
                    "transcript": "Secondo audio di test UI",
                    "snr_estimate": 22.0,
                    "is_primary": False,
                    "weight": 0.8,
                },
            ],
            "ref_audio_paths": [
                "ref_audios/sample_00_audio1.wav",
                "ref_audios/sample_01_audio2.wav",
            ],
        }
        with open(voice_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(ui_metadata, f, indent=2)

        # 3. Load via engine
        items = load_voice_prompt_items(voice_dir, base_dir=base_dir)

        assert isinstance(items, list)
        assert len(items) == 2
        assert all(isinstance(it, VoiceClonePromptItem) for it in items)

        # Check field mappings
        assert items[0].ref_text == "Primo audio di test UI"
        assert items[1].ref_text == "Secondo audio di test UI"
        assert items[0].x_vector_only_mode is False
        assert items[0].icl_mode is True
        assert torch.allclose(items[0].ref_spk_embedding, spk0)
        assert torch.equal(items[0].ref_code, code0)
        assert torch.allclose(items[1].ref_spk_embedding, spk1)
        assert torch.equal(items[1].ref_code, code1)


def test_ui_single_sample_prompt_roundtrip():
    """H2: Verify single-sample prompt saved by UI loads correctly with ref_text fallback."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)
        voice_dir = base_dir / "ui_single_voice"
        voice_dir.mkdir(parents=True, exist_ok=True)

        spk0 = torch.randn(192, dtype=torch.float32)
        code0 = torch.tensor([1, 2, 3], dtype=torch.int32)
        tensors = {
            "sample_0_ref_spk_embedding": spk0,
            "sample_0_ref_code": code0,
        }
        save_file(tensors, str(voice_dir / "voice_prompt.safetensors"))

        # UI saves single sample without samples array
        ui_metadata = {
            "name": "ui_single_voice",
            "description": "Single sample voice",
            "ref_text": "Frase di riferimento singola",
            "model": "0.6B-Base",
        }
        with open(voice_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(ui_metadata, f, indent=2)

        items = load_voice_prompt_items(voice_dir, base_dir=base_dir)
        assert len(items) == 1
        assert items[0].ref_text == "Frase di riferimento singola"
        assert torch.allclose(items[0].ref_spk_embedding, spk0)
        assert torch.equal(items[0].ref_code, code0)


def test_prompt_fallback_direct_tensor_discovery():
    """H2: Verify that when metadata.json has no samples, tensors are discovered automatically."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)
        voice_dir = base_dir / "headless_voice"
        voice_dir.mkdir(parents=True, exist_ok=True)

        spk0 = torch.randn(192, dtype=torch.float32)
        spk1 = torch.randn(192, dtype=torch.float32)
        tensors = {
            "sample_0_ref_spk_embedding": spk0,
            "sample_1_ref_spk_embedding": spk1,
        }
        save_file(tensors, str(voice_dir / "voice_prompt.safetensors"))

        # Empty metadata
        with open(voice_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump({}, f)

        items = load_voice_prompt_items(voice_dir, base_dir=base_dir)
        assert len(items) == 2
        assert items[0].x_vector_only_mode is True  # no ref_code means x-vector only
        assert items[1].x_vector_only_mode is True


def test_j3_real_ui_save_cloned_voice_multi_roundtrip():
    """J3 Mandatory test: UI writer flow save_prompt_to_safetensors_file -> save_cloned_voice_multi -> load_voice_prompt_items."""
    with tempfile.TemporaryDirectory() as tmpdir:
        saved_root = Path(tmpdir)

        # 1. Simulate UI prompt items created by create_combined_voice_clone_prompt
        spk1 = torch.randn(512, dtype=torch.float32)
        code1 = torch.tensor([101, 102, 103], dtype=torch.int64)
        spk2 = torch.randn(512, dtype=torch.float32)
        code2 = torch.tensor([201, 202, 203], dtype=torch.int64)

        prompt_item1 = VoiceClonePromptItem(
            ref_spk_embedding=spk1,
            ref_code=code1,
            x_vector_only_mode=False,
            icl_mode=True,
            ref_text="Test trascrizione 1",
        )
        prompt_item2 = VoiceClonePromptItem(
            ref_spk_embedding=spk2,
            ref_code=code2,
            x_vector_only_mode=False,
            icl_mode=True,
            ref_text="Test trascrizione 2",
        )

        prompt_items = [prompt_item1, prompt_item2]

        # 2. UI writes temporary safetensors file
        prompt_temp = saved_root / "temp_prompt.safetensors"
        save_prompt_to_safetensors_file(prompt_items, prompt_temp)
        assert prompt_temp.exists()

        # 3. Create dummy audio file for UI reference
        audio_dummy = saved_root / "sample_0.wav"
        audio_dummy.write_bytes(b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00D\xac\x00\x00\x88X\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00")

        samples_meta = [
            {"path": str(audio_dummy), "duration": 1.5, "transcript": "Test trascrizione 1", "is_primary": True, "weight": 1.0},
            {"path": str(audio_dummy), "duration": 1.2, "transcript": "Test trascrizione 2", "is_primary": False, "weight": 1.0},
        ]

        # 4. Invoke UI handler save_cloned_voice_multi
        with patch.object(qwen_tts_ui, "SAVED_VOICES_DIR", saved_root):
            msg, _ = qwen_tts_ui.save_cloned_voice_multi(
                voice_name="mario_rossi",
                description="Voce per test roundtrip UI",
                style_note="Naturale",
                ref_language="italian",
                audio_files=[str(audio_dummy), str(audio_dummy)],
                transcripts_json=json.dumps({"0": "Test 1", "1": "Test 2"}),
                prompt_path=str(prompt_temp),
                model_name="0.6B-Base",
                samples_meta_json=json.dumps(samples_meta),
            )
            assert "Saved voice: mario_rossi" in msg

        voice_dir = saved_root / "mario_rossi"
        assert voice_dir.exists()

        # 5. Load and verify via engine loader
        loaded_items = load_voice_prompt_items(voice_dir, base_dir=saved_root)
        assert len(loaded_items) == 2
        assert torch.allclose(loaded_items[0].ref_spk_embedding, spk1)
        assert torch.equal(loaded_items[0].ref_code, code1)
        assert torch.allclose(loaded_items[1].ref_spk_embedding, spk2)
        assert torch.equal(loaded_items[1].ref_code, code2)
        assert loaded_items[0].x_vector_only_mode is False
        assert loaded_items[0].icl_mode is True


def test_j3_real_ui_save_cloned_voice_single_roundtrip():
    """J3: UI writer flow save_prompt_to_safetensors_file -> save_cloned_voice (single-sample) -> load_voice_prompt_items."""
    with tempfile.TemporaryDirectory() as tmpdir:
        saved_root = Path(tmpdir)

        spk = torch.randn(512, dtype=torch.float32)
        code = torch.tensor([301, 302], dtype=torch.int64)

        prompt_item = VoiceClonePromptItem(
            ref_spk_embedding=spk,
            ref_code=code,
            x_vector_only_mode=False,
            icl_mode=True,
            ref_text="Test frase singola",
        )

        prompt_temp = saved_root / "temp_prompt_single.safetensors"
        save_prompt_to_safetensors_file(prompt_item, prompt_temp)

        audio_dummy = saved_root / "sample_single.wav"
        audio_dummy.write_bytes(b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00D\xac\x00\x00\x88X\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00")

        with patch.object(qwen_tts_ui, "SAVED_VOICES_DIR", saved_root):
            msg, _ = qwen_tts_ui.save_cloned_voice(
                voice_name="giulia_bianchi",
                description="Voce singola UI",
                style_note="Espressiva",
                ref_language="italian",
                ref_audio=str(audio_dummy),
                ref_text="Test frase singola",
                prompt_path=str(prompt_temp),
                model_name="0.6B-Base",
            )
            assert "Saved voice: giulia_bianchi" in msg

        voice_dir = saved_root / "giulia_bianchi"
        loaded_items = load_voice_prompt_items(voice_dir, base_dir=saved_root)
        assert len(loaded_items) == 1
        assert torch.allclose(loaded_items[0].ref_spk_embedding, spk)
        assert torch.equal(loaded_items[0].ref_code, code)
        assert loaded_items[0].ref_text == "Test frase singola"


def test_j3_xvector_only_prompt_roundtrip():
    """J3: Verify x-vector only prompt roundtrip (ref_code is None, x_vector_only_mode=True)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)
        voice_dir = base_dir / "xvec_voice"
        voice_dir.mkdir(parents=True, exist_ok=True)

        spk = torch.randn(512, dtype=torch.float32)
        prompt_item = VoiceClonePromptItem(
            ref_spk_embedding=spk,
            ref_code=None,
            x_vector_only_mode=True,
            icl_mode=False,
            ref_text="",
        )

        safetensors_file = voice_dir / "voice_prompt.safetensors"
        save_prompt_to_safetensors_file([prompt_item], safetensors_file)

        loaded_items = load_voice_prompt_items(voice_dir, base_dir=base_dir)
        assert len(loaded_items) == 1
        assert loaded_items[0].ref_code is None
        assert loaded_items[0].x_vector_only_mode is True
        assert loaded_items[0].icl_mode is False
        assert torch.allclose(loaded_items[0].ref_spk_embedding, spk)


def test_j3_negative_corrupted_header_raises_prompt_validation_error():
    """J3 Negative: Corrupted JSON in safetensors metadata header must raise PromptValidationError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)
        voice_dir = base_dir / "corrupted_header_voice"
        voice_dir.mkdir(parents=True, exist_ok=True)

        tensors = {"sample_0_ref_spk_embedding": torch.randn(512, dtype=torch.float32)}
        # Write corrupted JSON header in metadata
        save_file(tensors, str(voice_dir / "voice_prompt.safetensors"), metadata={"samples_json": "{not-valid-json"})

        with pytest.raises(PromptValidationError, match="Corrupted samples_json in safetensors header"):
            load_voice_prompt_items(voice_dir, base_dir=base_dir)


def test_j3_negative_missing_speaker_embedding_raises_prompt_validation_error():
    """J3 Negative: Declared sample with missing speaker embedding must raise PromptValidationError without silent continue."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)
        voice_dir = base_dir / "missing_embedding_voice"
        voice_dir.mkdir(parents=True, exist_ok=True)

        # Safetensors has sample_0 but lacks sample_1
        tensors = {"sample_0_ref_spk_embedding": torch.randn(512, dtype=torch.float32)}
        save_file(tensors, str(voice_dir / "voice_prompt.safetensors"))

        metadata = {
            "name": "missing_embedding_voice",
            "num_samples": 2,
            "samples": [
                {"index": 0, "transcript": "sample 0"},
                {"index": 1, "transcript": "sample 1"},
            ],
        }
        with open(voice_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f)

        with pytest.raises(PromptValidationError, match="Missing speaker embedding"):
            load_voice_prompt_items(voice_dir, base_dir=base_dir)


def test_j3_negative_inconsistent_sample_count_raises_prompt_validation_error():
    """J3 Negative: Mismatch between declared num_samples and reconstructed items must raise PromptValidationError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)
        voice_dir = base_dir / "count_mismatch_voice"
        voice_dir.mkdir(parents=True, exist_ok=True)

        tensors = {
            "sample_0_ref_spk_embedding": torch.randn(512, dtype=torch.float32),
            "sample_1_ref_spk_embedding": torch.randn(512, dtype=torch.float32),
        }
        save_file(tensors, str(voice_dir / "voice_prompt.safetensors"))

        metadata = {
            "name": "count_mismatch_voice",
            "num_samples": 5,  # Declared 5 samples, but only 2 present
        }
        with open(voice_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f)

        with pytest.raises(PromptValidationError, match="Sample count mismatch"):
            load_voice_prompt_items(voice_dir, base_dir=base_dir)


def test_j3_negative_corrupted_metadata_json_raises_prompt_validation_error():
    """J3 Negative: Corrupted companion metadata.json must raise PromptValidationError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)
        voice_dir = base_dir / "corrupted_meta_voice"
        voice_dir.mkdir(parents=True, exist_ok=True)

        tensors = {"sample_0_ref_spk_embedding": torch.randn(512, dtype=torch.float32)}
        save_file(tensors, str(voice_dir / "voice_prompt.safetensors"))

        # Write invalid JSON content
        (voice_dir / "metadata.json").write_text("{bad-json-content", encoding="utf-8")

        with pytest.raises(PromptValidationError, match="Corrupted metadata.json"):
            load_voice_prompt_items(voice_dir, base_dir=base_dir)


def test_j3_negative_invalid_tensor_shape_raises_prompt_validation_error():
    """J3 Negative: Speaker embedding with invalid shape (e.g. 3D tensor) must raise PromptValidationError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)
        voice_dir = base_dir / "invalid_shape_voice"
        voice_dir.mkdir(parents=True, exist_ok=True)

        # 3D tensor instead of 1D/2D
        tensors = {"sample_0_ref_spk_embedding": torch.randn(2, 4, 128, dtype=torch.float32)}
        save_file(tensors, str(voice_dir / "voice_prompt.safetensors"))

        with pytest.raises(PromptValidationError, match="Invalid speaker embedding shape"):
            load_voice_prompt_items(voice_dir, base_dir=base_dir)


def test_j3_negative_invalid_tensor_dtype_raises_prompt_validation_error():
    """J3 Negative: Speaker embedding with non-floating point dtype (e.g. int64) must raise PromptValidationError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)
        voice_dir = base_dir / "invalid_dtype_voice"
        voice_dir.mkdir(parents=True, exist_ok=True)

        # Integer tensor instead of float
        tensors = {"sample_0_ref_spk_embedding": torch.tensor([1, 2, 3, 4], dtype=torch.int64)}
        save_file(tensors, str(voice_dir / "voice_prompt.safetensors"))

        with pytest.raises(PromptValidationError, match="Invalid speaker embedding dtype"):
            load_voice_prompt_items(voice_dir, base_dir=base_dir)

"""Tests for UI to Engine prompt serialization and deserialization compatibility (H2)."""

import json
from pathlib import Path
import tempfile
import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from src.engine.prompt_manager import (
    load_voice_prompt_items,
    save_voice_prompt_items,
    save_prompt_to_safetensors_file,
)
from qwen_tts.inference.qwen3_tts_model import VoiceClonePromptItem


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

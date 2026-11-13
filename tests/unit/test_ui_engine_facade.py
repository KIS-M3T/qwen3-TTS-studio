"""Unit tests for TTSEngineFacade and UI integration regression (G5)."""

from pathlib import Path
import tempfile
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
import torch

from qwen_tts.inference.qwen3_tts_model import VoiceClonePromptItem
from src.engine.facade import TTSEngineFacade
from src.engine.types import AudioBuffer, DecodingParams, GenerationResult, MemorySnapshot, VoicePrompt


class DummyMockAdapter:
    """Mock adapter for facade testing."""

    def __init__(self, model_id: str):
        self.model_id = model_id
        self.state = "READY"
        self.unloaded = False

    def load(self, device="cuda:0", dtype_str="bfloat16"):
        self.state = "READY"

    def unload(self):
        self.unloaded = True
        self.state = "UNLOADED"

    def synthesize_base(self, text, language="italian", params=None):
        return self._make_res("base")

    def synthesize_voice_clone(self, text, voice_prompt, language="italian", params=None):
        return self._make_res("clone")

    def synthesize_custom_voice(self, text, speaker_id, language="italian", params=None):
        return self._make_res("custom")

    def _make_res(self, mode):
        buf = AudioBuffer(samples=np.zeros(24000, dtype=np.float32), sample_rate=24000)
        mem = MemorySnapshot(0, 0, 0, None, 0)
        return GenerationResult(
            audio=buf,
            inference_time_sec=0.5,
            rtf=0.5,
            memory=mem,
            model_version=self.model_id,
            decoding_params=DecodingParams(),
            environment_manifest_sha256="fake_sha",
            metadata={"mode": mode},
        )


def test_facade_synthesis_routing():
    """Verify facade properly routes base, clone, and custom voice calls (G5)."""
    facade = TTSEngineFacade()
    mock_adapter = DummyMockAdapter("0.6B-Base")
    facade.get_or_load_adapter = MagicMock(return_value=mock_adapter)

    # 1. Base synthesis
    res_base = facade.synthesize(text="Base test", model_id="0.6B-Base", mode="base")
    assert res_base.metadata["mode"] == "base"

    # 2. Custom voice
    res_custom = facade.synthesize(text="Custom test", model_id="0.6B-CustomVoice", mode="custom", speaker_id="alessandro")
    assert res_custom.metadata["mode"] == "custom"

    # 3. Voice clone
    fake_prompt = VoicePrompt(
        voice_id="v1",
        ref_audio_path=Path("/fake.wav"),
        raw_prompt_payload={"test": "payload"},
    )
    res_clone = facade.synthesize(text="Clone test", model_id="0.6B-Base", mode="clone", voice_prompt=fake_prompt)
    assert res_clone.metadata["mode"] == "clone"


def test_facade_save_and_load_voice_profile():
    """Verify facade voice profile management roundtrip (G2, G5)."""
    facade = TTSEngineFacade()

    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)
        item = VoiceClonePromptItem(
            ref_code=torch.tensor([1, 2, 3]),
            ref_spk_embedding=torch.randn(192),
            x_vector_only_mode=False,
            icl_mode=True,
            ref_text="Test trascrizione",
        )

        saved_path = facade.save_voice_profile(
            voice_id="facade_voice",
            prompt_items=item,
            description="Voce salvata via facade",
            base_dir=base_dir,
        )
        assert (saved_path / "voice_prompt.safetensors").exists()

        loaded_items = facade.load_voice_profile(voice_id="facade_voice", base_dir=base_dir)
        assert len(loaded_items) == 1
        assert loaded_items[0].ref_text == "Test trascrizione"

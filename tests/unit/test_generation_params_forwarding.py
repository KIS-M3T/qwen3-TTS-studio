"""Unit tests verifying generation parameter forwarding and vendor spy (G1)."""

from unittest.mock import MagicMock
import numpy as np
import pytest
import torch

from src.engine.adapters.qwen3_adapter import Qwen3EngineAdapter
from src.engine.types import DecodingParams, LifecycleState, VoicePrompt


class MockVendorModel:
    """Mock spy tracking keyword arguments received by Qwen3TTSModel."""

    def __init__(self):
        self.last_kwargs = {}
        self.last_call = None
        self.tts_model_type = "base"

    def generate_voice_clone(self, *args, **kwargs):
        self.last_call = "generate_voice_clone"
        self.last_kwargs = kwargs
        # Return fake 1 sec 24000Hz mono audio
        return [np.zeros(24000, dtype=np.float32)], 24000

    def generate_custom_voice(self, *args, **kwargs):
        self.last_call = "generate_custom_voice"
        self.last_kwargs = kwargs
        return [np.zeros(24000, dtype=np.float32)], 24000


def test_generation_params_forwarding_voice_clone():
    """Verify that DecodingParams are passed to vendor generate_voice_clone."""
    adapter = Qwen3EngineAdapter("0.6B-Base", model_path="/fake/path")
    mock_model = MockVendorModel()
    adapter.model = mock_model
    adapter.lifecycle.mark_ready()

    params = DecodingParams(
        temperature=0.35,
        top_p=0.88,
        top_k=45,
        repetition_penalty=1.12,
        min_new_tokens=60,
        max_new_tokens=512,
        seed=12345,
        subtalker_params={
            "subtalker_temperature": 0.4,
            "subtalker_top_k": 30,
            "unsupported_flag": True,
        },
    )

    fake_voice_prompt = VoicePrompt(
        voice_id="test_voice",
        ref_audio_path="/fake/audio.wav",
        raw_prompt_payload={"mock": "payload"},
    )

    result = adapter.synthesize_voice_clone(
        text="Test sintesi vocale",
        voice_prompt=fake_voice_prompt,
        language="italian",
        params=params,
    )

    # 1. Verify vendor spy received expected kwargs
    assert mock_model.last_call == "generate_voice_clone"
    assert mock_model.last_kwargs["temperature"] == 0.35
    assert mock_model.last_kwargs["top_p"] == 0.88
    assert mock_model.last_kwargs["top_k"] == 45
    assert mock_model.last_kwargs["repetition_penalty"] == 1.12
    assert mock_model.last_kwargs["max_new_tokens"] == 512
    assert mock_model.last_kwargs["subtalker_temperature"] == 0.4
    assert mock_model.last_kwargs["subtalker_top_k"] == 30
    assert "unsupported_flag" not in mock_model.last_kwargs

    # 2. Verify metadata structure
    assert result.metadata["applied_params"]["temperature"] == 0.35
    assert result.metadata["applied_params"]["top_p"] == 0.88
    assert result.metadata["applied_params"]["max_new_tokens"] == 512
    assert result.metadata["requested_params"]["seed"] == 12345
    assert "unsupported_flag" in result.metadata["unsupported_params"]

    adapter.unload()


def test_generation_params_forwarding_custom_voice():
    """Verify that DecodingParams are passed to vendor generate_custom_voice."""
    adapter = Qwen3EngineAdapter("0.6B-CustomVoice", model_path="/fake/path")
    mock_model = MockVendorModel()
    adapter.model = mock_model
    adapter.lifecycle.mark_ready()

    params = DecodingParams(
        temperature=0.5,
        top_p=0.9,
        top_k=30,
        repetition_penalty=1.0,
        min_new_tokens=60,
        max_new_tokens=256,
        seed=999,
    )

    result = adapter.synthesize_custom_voice(
        text="Test sintesi custom voice",
        speaker_id="alessandro",
        language="italian",
        params=params,
    )

    assert mock_model.last_call == "generate_custom_voice"
    assert mock_model.last_kwargs["temperature"] == 0.5
    assert mock_model.last_kwargs["top_p"] == 0.9
    assert mock_model.last_kwargs["top_k"] == 30
    assert mock_model.last_kwargs["max_new_tokens"] == 256
    assert result.metadata["seed_applied"] == 999

    adapter.unload()


def test_rng_state_restored_after_generation():
    """Verify RNG state is restored after seeded generation (seed isolation)."""
    adapter = Qwen3EngineAdapter("0.6B-Base", model_path="/fake/path")
    mock_model = MockVendorModel()
    adapter.model = mock_model
    adapter.lifecycle.mark_ready()

    torch.manual_seed(100)
    expected_rand = torch.rand(5)

    # Reset seed to 100
    torch.manual_seed(100)

    params = DecodingParams(seed=4242)
    fake_voice_prompt = VoicePrompt(
        voice_id="test_voice",
        ref_audio_path="/fake/audio.wav",
        raw_prompt_payload={"mock": "payload"},
    )
    adapter.synthesize_voice_clone(
        text="Test",
        voice_prompt=fake_voice_prompt,
        params=params,
    )

    # Next random tensor should match expected_rand (since RNG state was restored)
    actual_rand = torch.rand(5)
    assert torch.allclose(expected_rand, actual_rand)

    adapter.unload()

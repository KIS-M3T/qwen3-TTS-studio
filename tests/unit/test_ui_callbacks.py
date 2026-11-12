"""Tests verifying that UI generation callbacks route through TTSEngineFacade (H3)."""

from pathlib import Path
import tempfile
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from src.engine.types import AudioBuffer, DecodingParams, GenerationResult, MemorySnapshot


def _create_mock_generation_result():
    buf = AudioBuffer(
        samples=np.zeros(24000, dtype=np.float32),
        sample_rate=24000,
        duration_sec=1.0,
    )
    mem = MemorySnapshot(
        torch_allocated_bytes=1000,
        torch_reserved_bytes=2000,
        torch_peak_allocated_bytes=1500,
        driver_used_bytes_at_end=3000,
    )
    return GenerationResult(
        audio=buf,
        inference_time_sec=0.5,
        rtf=0.5,
        memory=mem,
        model_version="0.6B-CustomVoice",
        decoding_params=DecodingParams(),
        environment_manifest_sha256="fake_sha",
    )


def test_generate_custom_voice_invokes_facade():
    """H3: Ensure generate_custom_voice routes through TTSEngineFacade.synthesize without calling get_model."""
    from qwen_tts_ui import generate_custom_voice

    mock_facade = MagicMock()
    mock_facade.synthesize.return_value = _create_mock_generation_result()

    with patch("qwen_tts_ui.get_engine_facade", return_value=mock_facade), \
         patch("qwen_tts_ui.get_model") as mock_get_model, \
         patch("qwen_tts_ui.save_to_history", return_value="fake_history.wav"):

        audio_path, error_msg = generate_custom_voice(
            text="Buongiorno da Milano.",
            speaker="vivian",
            language="italian",
            instruct="Parla con tono chiaro",
            temperature=0.7,
            top_k=20,
            top_p=0.8,
            repetition_penalty=1.05,
            max_new_tokens=2048,
            model_name="0.6B-CustomVoice",
            sub_temp=0.7,
            sub_top_k=20,
            sub_top_p=0.8,
        )

        # 1. Verify get_model was NOT called
        mock_get_model.assert_not_called()

        # 2. Verify TTSEngineFacade.synthesize WAS called with expected mode and parameters
        assert mock_facade.synthesize.call_count == 1
        call_kwargs = mock_facade.synthesize.call_args.kwargs
        assert call_kwargs["mode"] == "custom"
        assert call_kwargs["text"] == "Buongiorno da Milano."
        assert call_kwargs["model_id"] == "0.6B-CustomVoice"
        assert call_kwargs["speaker_id"] == "vivian"
        assert call_kwargs["language"] == "italian"
        assert isinstance(call_kwargs["params"], DecodingParams)
        assert call_kwargs["params"].subtalker_params.get("instruct") == "Parla con tono chiaro"
        assert "❌" not in error_msg


def test_generate_with_saved_voice_invokes_facade():
    """H3: Ensure generate_with_saved_voice routes through TTSEngineFacade.synthesize."""
    from qwen_tts_ui import generate_with_saved_voice

    mock_facade = MagicMock()
    mock_facade.synthesize.return_value = _create_mock_generation_result()

    with tempfile.TemporaryDirectory() as tmpdir:
        voices_dir = Path(tmpdir)
        voice_folder = voices_dir / "mario_rossi"
        voice_folder.mkdir(parents=True, exist_ok=True)

        # Save dummy safetensors prompt and metadata
        tensors = {"sample_0_ref_spk_embedding": torch.randn(192, dtype=torch.float32)}
        save_file(tensors, str(voice_folder / "voice_prompt.safetensors"))
        with open(voice_folder / "metadata.json", "w", encoding="utf-8") as f:
            import json
            json.dump({"name": "mario_rossi", "model": "0.6B-Base", "ref_text": "Voce di Mario"}, f)

        with patch("qwen_tts_ui.SAVED_VOICES_DIR", voices_dir), \
             patch("qwen_tts_ui.get_engine_facade", return_value=mock_facade), \
             patch("qwen_tts_ui.get_model") as mock_get_model, \
             patch("qwen_tts_ui.save_to_history", return_value="fake_history.wav"):

            audio_path, error_msg = generate_with_saved_voice(
                text="Test voce salvata.",
                saved_voice_id="mario_rossi",
                language="italian",
                crosslingual_opt=False,
                temperature=0.7,
                top_k=20,
                top_p=0.8,
                repetition_penalty=1.05,
                max_new_tokens=2048,
                sub_temp=0.7,
                sub_top_k=20,
                sub_top_p=0.8,
            )

            # 1. get_model must not be called
            mock_get_model.assert_not_called()

            # 2. facade.synthesize must be invoked with clone mode
            assert mock_facade.synthesize.call_count == 1
            call_kwargs = mock_facade.synthesize.call_args.kwargs
            assert call_kwargs["mode"] == "clone"
            assert call_kwargs["text"] == "Test voce salvata."
            assert call_kwargs["voice_prompt"].voice_id == "mario_rossi"
            assert "❌" not in error_msg


def test_clone_voice_multi_invokes_facade():
    """H3: Ensure clone_voice_multi loads model via Facade adapter and synthesizes via Facade."""
    from qwen_tts_ui import clone_voice_multi

    mock_facade = MagicMock()
    mock_adapter = MagicMock()
    mock_model = MagicMock()
    mock_adapter.model = mock_model
    mock_facade.get_or_load_adapter.return_value = mock_adapter
    mock_facade.synthesize.return_value = _create_mock_generation_result()

    fake_sample_info = MagicMock()
    fake_sample_info.path = "ref.wav"
    fake_sample_info.transcript = "Test trascrizione"
    fake_sample_info.duration = 3.0
    fake_sample_info.snr_estimate = 20.0
    fake_sample_info.is_primary = True
    fake_sample_info.weight = 1.0

    with patch("qwen_tts_ui.get_engine_facade", return_value=mock_facade), \
         patch("qwen_tts_ui.get_model") as mock_get_model, \
         patch("qwen_tts_ui.analyze_audio_samples", return_value=[fake_sample_info]), \
         patch("qwen_tts_ui.save_to_history", return_value="fake_history.wav"), \
         patch("src.engine.prompt_manager.save_prompt_to_safetensors_file"):

        result = clone_voice_multi(
            audio_files=["test.wav"],
            transcripts_json="{}",
            model_name="0.6B-Base",
            test_text="Sintesi clone test.",
            language="italian",
            ref_language="italian",
            crosslingual_opt=False,
            combine_samples=False,
            temperature=0.7,
            top_k=20,
            top_p=0.8,
            repetition_penalty=1.05,
            max_new_tokens=2048,
            sub_temp=0.7,
            sub_top_k=20,
            sub_top_p=0.8,
        )

        # 1. Legacy get_model must NOT be called
        mock_get_model.assert_not_called()

        # 2. Facade adapter loading must be invoked
        mock_facade.get_or_load_adapter.assert_called_with("0.6B-Base")

        # 3. Facade synthesis must be invoked for test_text
        assert mock_facade.synthesize.call_count == 1
        call_kwargs = mock_facade.synthesize.call_args.kwargs
        assert call_kwargs["mode"] == "clone"
        assert call_kwargs["text"] == "Sintesi clone test."
        assert call_kwargs["model_id"] == "0.6B-Base"

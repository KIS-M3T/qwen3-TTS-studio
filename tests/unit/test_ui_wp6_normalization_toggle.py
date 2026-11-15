"""WP6 UI regression tests for M2 opt-in normalization toggle and diff preview."""

import inspect
import json
from pathlib import Path
import tempfile
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from src.engine.types import AudioBuffer, DecodingParams, GenerationResult, MemorySnapshot


def _mock_generation_result(model_version: str = "1.7B-Base") -> GenerationResult:
    buf = AudioBuffer(
        samples=np.zeros(24000, dtype=np.float32),
        sample_rate=24000,
        duration_sec=1.0,
    )
    mem = MemorySnapshot(1000, 2000, 1500, 3000)
    return GenerationResult(
        audio=buf,
        inference_time_sec=0.25,
        rtf=0.25,
        memory=mem,
        model_version=model_version,
        decoding_params=DecodingParams(),
        environment_manifest_sha256="wp6_test_sha",
    )


def test_ui_callbacks_default_m2_toggle_off():
    """Every primary synthesis callback must default enable_m2_normalization to False."""
    import qwen_tts_ui

    for fn in (
        qwen_tts_ui.generate_custom_voice,
        qwen_tts_ui.generate_voice_design,
        qwen_tts_ui.clone_voice_multi,
        qwen_tts_ui.generate_with_saved_voice,
    ):
        sig = inspect.signature(fn)
        assert "enable_m2_normalization" in sig.parameters
        assert sig.parameters["enable_m2_normalization"].default is False


def test_ui_gradio_components_optin_disabled_by_default():
    """Verify Gradio UI defines M2 toggles disabled by default and collapsible diff panels."""
    import gradio as gr
    import qwen_tts_ui

    checkboxes = [
        comp
        for comp in qwen_tts_ui.demo.blocks.values()
        if isinstance(comp, gr.Checkbox) and comp.label == qwen_tts_ui.M2_TOGGLE_LABEL
    ]
    accordions = [
        comp
        for comp in qwen_tts_ui.demo.blocks.values()
        if isinstance(comp, gr.Accordion) and comp.label == qwen_tts_ui.M2_PREVIEW_LABEL
    ]
    assert len(checkboxes) == 4
    assert all(cb.value is False for cb in checkboxes)
    assert len(accordions) == 4
    assert all(acc.open is False for acc in accordions)


def test_ui_diff_preview_helper_off_vs_on():
    """Verify update_m2_diff_preview reflects OFF state and ON raw->normalized preview."""
    from qwen_tts_ui import update_m2_diff_preview

    sample = "Il dott. Rossi paga 15 €."
    assert update_m2_diff_preview("", False) == ""
    assert update_m2_diff_preview("   ", True) == ""

    off_preview = update_m2_diff_preview(sample, False)
    assert "M2 OFF" in off_preview
    assert sample in off_preview
    assert "quindici euro" not in off_preview

    on_preview = update_m2_diff_preview(sample, True)
    assert "M2 ON" in on_preview
    assert sample in on_preview
    assert "Il dottor Rossi paga quindici euro." in on_preview


@pytest.mark.parametrize("toggle_on", [False, True])
def test_generate_custom_voice_toggle_off_and_on(toggle_on):
    """Toggle OFF preserves exact M1 text; Toggle ON sends normalized text without altering params."""
    from qwen_tts_ui import generate_custom_voice

    raw = "Il dott. Rossi ha pagato 15 € il 01/05/2026."
    expected = (
        "Il dottor Rossi ha pagato quindici euro il primo maggio duemilaventisei."
        if toggle_on
        else raw
    )

    mock_facade = MagicMock()
    mock_facade.synthesize.return_value = _mock_generation_result("1.7B-CustomVoice")

    with patch("qwen_tts_ui.get_engine_facade", return_value=mock_facade), \
         patch("qwen_tts_ui.save_to_history", return_value="hist.wav") as mock_hist:
        audio_path, status = generate_custom_voice(
            text=raw,
            model_name="1.7B-CustomVoice",
            speaker="vivian",
            language="italian",
            instruct="Tono calmo",
            temperature=0.7,
            top_k=20,
            top_p=0.9,
            repetition_penalty=1.05,
            max_new_tokens=2048,
            sub_temp=0.7,
            sub_top_k=20,
            sub_top_p=0.8,
            enable_m2_normalization=toggle_on,
        )

    assert audio_path == "hist.wav"
    assert "❌" not in status
    call_kwargs = mock_facade.synthesize.call_args.kwargs
    assert call_kwargs["text"] == expected
    assert call_kwargs["mode"] == "custom"
    assert call_kwargs["model_id"] == "1.7B-CustomVoice"
    assert call_kwargs["speaker_id"] == "vivian"
    assert call_kwargs["params"].temperature == pytest.approx(0.7)
    assert mock_hist.call_args.kwargs["params"]["m2_normalization"] is toggle_on


@pytest.mark.parametrize("toggle_on", [False, True])
def test_generate_with_saved_voice_toggle_off_and_on(toggle_on):
    """Saved voices callback keeps prompt/mode untouched and applies M2 only when enabled."""
    from qwen_tts_ui import generate_with_saved_voice

    raw = "Appuntamento alle 09:00."
    expected = "Appuntamento alle nove." if toggle_on else raw

    mock_facade = MagicMock()
    mock_facade.synthesize.return_value = _mock_generation_result("1.7B-Base")

    with tempfile.TemporaryDirectory() as tmpdir:
        voices_dir = Path(tmpdir)
        voice_folder = voices_dir / "it_voice"
        voice_folder.mkdir(parents=True, exist_ok=True)
        save_file({"sample_0_ref_spk_embedding": torch.randn(192)}, str(voice_folder / "voice_prompt.safetensors"))
        (voice_folder / "metadata.json").write_text(
            json.dumps({"name": "it_voice", "model": "1.7B-Base", "ref_text": "Voce test"}),
            encoding="utf-8",
        )

        with patch("qwen_tts_ui.SAVED_VOICES_DIR", voices_dir), \
             patch("qwen_tts_ui.get_engine_facade", return_value=mock_facade), \
             patch("qwen_tts_ui.save_to_history", return_value="hist_saved.wav"):
            audio_path, status = generate_with_saved_voice(
                text=raw,
                saved_voice_id="it_voice",
                language="italian",
                crosslingual_opt=False,
                temperature=0.7,
                top_k=20,
                top_p=0.9,
                repetition_penalty=1.05,
                max_new_tokens=2048,
                sub_temp=0.7,
                sub_top_k=20,
                sub_top_p=0.8,
                enable_m2_normalization=toggle_on,
            )

    assert audio_path == "hist_saved.wav"
    assert "❌" not in status
    call_kwargs = mock_facade.synthesize.call_args.kwargs
    assert call_kwargs["text"] == expected
    assert call_kwargs["mode"] == "clone"
    assert call_kwargs["voice_prompt"].voice_id == "it_voice"


@pytest.mark.parametrize("toggle_on", [False, True])
def test_clone_voice_multi_toggle_off_and_on(toggle_on):
    """Clone callback preserves prompt creation/saving and toggles M2 normalization cleanly."""
    from qwen_tts_ui import clone_voice_multi

    raw = "Sede INPS al 12%."
    expected = "Sede I N P S al dodici per cento." if toggle_on else raw

    mock_facade = MagicMock()
    mock_adapter = MagicMock()
    mock_adapter.model = MagicMock()
    mock_facade.use_adapter.return_value.__enter__.return_value = mock_adapter
    mock_facade.synthesize.return_value = _mock_generation_result("1.7B-Base")

    fake_sample = MagicMock()
    fake_sample.path = "ref.wav"
    fake_sample.transcript = "Trascrizione test"
    fake_sample.duration = 3.0
    fake_sample.snr_estimate = 20.0
    fake_sample.is_primary = True
    fake_sample.weight = 1.0

    with patch("qwen_tts_ui.get_engine_facade", return_value=mock_facade), \
         patch("qwen_tts_ui.analyze_audio_samples", return_value=[fake_sample]), \
         patch("qwen_tts_ui.save_to_history", return_value="hist_clone.wav"), \
         patch("src.engine.prompt_manager.save_prompt_to_safetensors_file"):
        clone_voice_multi(
            audio_files=["ref.wav"],
            transcripts_json="{}",
            model_name="1.7B-Base",
            test_text=raw,
            language="italian",
            ref_language="italian",
            crosslingual_opt=False,
            combine_samples=False,
            temperature=0.7,
            top_k=20,
            top_p=0.9,
            repetition_penalty=1.05,
            max_new_tokens=2048,
            sub_temp=0.7,
            sub_top_k=20,
            sub_top_p=0.8,
            enable_m2_normalization=toggle_on,
        )

    assert mock_facade.synthesize.call_args.kwargs["text"] == expected
    assert mock_facade.synthesize.call_args.kwargs["mode"] == "clone"


@pytest.mark.parametrize("toggle_on", [False, True])
def test_generate_voice_design_toggle_off_and_on(toggle_on):
    """Voice Design callback respects opt-in toggle without touching description."""
    from qwen_tts_ui import generate_voice_design

    raw = "Prezzo 20 €."
    expected = "Prezzo venti euro." if toggle_on else raw

    mock_facade = MagicMock()
    mock_facade._coordinator_lock = MagicMock()
    mock_vd_model = MagicMock()
    mock_vd_model.generate_voice_design.return_value = ([np.zeros(24000, dtype=np.float32)], 24000)

    with patch("qwen_tts_ui.get_engine_facade", return_value=mock_facade), \
         patch("qwen_tts_ui.get_model", return_value=mock_vd_model), \
         patch("qwen_tts_ui.save_to_history", return_value="hist_vd.wav"):
        audio_path, status = generate_voice_design(
            text=raw,
            voice_description="Voce calda italiana",
            language="italian",
            temperature=0.7,
            top_k=20,
            top_p=0.9,
            repetition_penalty=1.05,
            max_new_tokens=2048,
            sub_temp=0.7,
            sub_top_k=20,
            sub_top_p=0.8,
            enable_m2_normalization=toggle_on,
        )

    assert audio_path == "hist_vd.wav"
    assert "❌" not in status
    gen_kwargs = mock_vd_model.generate_voice_design.call_args.kwargs
    assert gen_kwargs["text"] == expected
    assert gen_kwargs["instruct"] == "Voce calda italiana"

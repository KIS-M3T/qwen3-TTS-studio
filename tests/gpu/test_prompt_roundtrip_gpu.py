"""Mandatory GPU validation test for VoicePrompt roundtrip and synthesis (G2).

Verifies the full pipeline on real GPU hardware:
create_voice_prompt -> save (safetensors) -> unload -> load -> generate_voice_clone
"""

from pathlib import Path
import json
import tempfile
from unittest.mock import patch
import pytest
import torch

from src.engine.adapters.qwen3_adapter import Qwen3EngineAdapter
from src.engine.prompt_manager import (
    load_voice_prompt,
    load_voice_prompt_items,
    save_prompt_to_safetensors_file,
    save_voice_prompt_items,
)
from src.engine.types import DecodingParams, LifecycleState, VoicePrompt
import qwen_tts_ui


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required for this test")
def test_gpu_prompt_roundtrip_and_clone():
    """Mandatory G2 test: create -> save -> unload -> load -> generate_voice_clone on real GPU."""
    ref_audio = Path("bench/reference_voices/ref_voice_it_01.wav")
    assert ref_audio.exists(), f"Reference audio not found at {ref_audio}"
    ref_text = "Pone sotto la propria egida il campionato e la Nazionale libanese."

    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)
        voice_id = "test_gpu_voice"

        # 1. Load model and create prompt from real audio
        adapter = Qwen3EngineAdapter("0.6B-Base")
        adapter.load(device="cuda:0", dtype_str="bfloat16")
        assert adapter.state == LifecycleState.READY

        vp_original = adapter.create_voice_prompt(
            ref_audio_path=str(ref_audio),
            ref_transcript=ref_text,
            x_vector_only=False,
        )
        assert vp_original.raw_prompt_payload is not None

        # 2. Save prompt to non-executable safetensors
        saved_dir = save_voice_prompt_items(
            items=vp_original.raw_prompt_payload,
            voice_id=voice_id,
            base_dir=base_dir,
            description="GPU test voice for G2 verification",
            ref_audio_paths=[str(ref_audio)],
        )
        assert (saved_dir / "voice_prompt.safetensors").exists()
        assert (saved_dir / "metadata.json").exists()

        # 3. Unload model to prove memory release and state reset
        adapter.unload()
        assert adapter.state == LifecycleState.UNLOADED
        assert adapter.model is None

        # 4. Re-load model and load saved prompt from safetensors
        adapter.load(device="cuda:0", dtype_str="bfloat16")
        assert adapter.state == LifecycleState.READY

        vp_loaded = load_voice_prompt(saved_dir, base_dir=base_dir)
        assert vp_loaded.raw_prompt_payload is not None

        # 5. Execute actual voice clone synthesis with loaded prompt
        dec_params = DecodingParams(temperature=0.7, top_p=0.8, top_k=20, seed=42)
        gen_result = adapter.synthesize_voice_clone(
            text="Questa è una verifica reale di clonazione vocale su GPU.",
            voice_prompt=vp_loaded,
            language="italian",
            params=dec_params,
        )

        # 6. Validate generated audio and metadata
        assert gen_result.audio.duration_sec > 1.0
        assert gen_result.audio.sample_rate == 24000
        assert gen_result.rtf > 0.0
        assert gen_result.metadata["mode"] == "clone"
        assert gen_result.metadata["applied_params"]["temperature"] == 0.7
        assert gen_result.metadata["applied_params"]["max_new_tokens"] == 2048

        # 7. Final clean unload
        adapter.unload()
        assert adapter.state == LifecycleState.UNLOADED


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required for this test")
def test_gpu_real_ui_save_cloned_voice_multi_and_synthesis():
    """J3 Mandatory GPU test: real UI flow save_prompt_to_safetensors_file -> save_cloned_voice_multi -> load_voice_prompt_items -> synthesize_voice_clone."""
    ref_audio = Path("bench/reference_voices/ref_voice_it_01.wav")
    assert ref_audio.exists(), f"Reference audio not found at {ref_audio}"
    ref_text = "Pone sotto la propria egida il campionato e la Nazionale libanese."

    with tempfile.TemporaryDirectory() as tmpdir:
        saved_root = Path(tmpdir)
        voice_id = "gpu_ui_mario_rossi"

        # 1. Load model on GPU
        adapter = Qwen3EngineAdapter("0.6B-Base")
        adapter.load(device="cuda:0", dtype_str="bfloat16")
        assert adapter.state == LifecycleState.READY

        # 2. Create native prompt via model
        prompt_items = adapter.model.create_voice_clone_prompt(
            ref_audio=str(ref_audio),
            ref_text=ref_text,
            x_vector_only_mode=False,
        )
        assert prompt_items is not None

        # 3. Save temporary safetensors as done by UI
        prompt_temp = saved_root / "temp_prompt_ui.safetensors"
        save_prompt_to_safetensors_file(prompt_items, prompt_temp)
        assert prompt_temp.exists()

        # 4. Invoke real UI save handler
        samples_meta = [
            {
                "path": str(ref_audio),
                "duration": 4.1,
                "transcript": ref_text,
                "is_primary": True,
                "weight": 1.0,
            }
        ]

        with patch.object(qwen_tts_ui, "SAVED_VOICES_DIR", saved_root):
            msg, _ = qwen_tts_ui.save_cloned_voice_multi(
                voice_name=voice_id,
                description="Test GPU UI voice clone multi",
                style_note="Naturale",
                ref_language="italian",
                audio_files=[str(ref_audio)],
                transcripts_json=json.dumps({"0": ref_text}),
                prompt_path=str(prompt_temp),
                model_name="0.6B-Base",
                samples_meta_json=json.dumps(samples_meta),
            )
            assert f"Saved voice: {voice_id}" in msg

        voice_dir = saved_root / voice_id
        assert (voice_dir / "voice_prompt.safetensors").exists()
        assert (voice_dir / "metadata.json").exists()

        # 5. Load reconstructed prompt items via engine loader
        reconstructed_items = load_voice_prompt_items(voice_dir, base_dir=saved_root)
        assert len(reconstructed_items) == 1
        assert reconstructed_items[0].ref_spk_embedding is not None
        assert reconstructed_items[0].ref_code is not None

        # 6. Synthesize audio on GPU using loaded prompt
        vp = VoicePrompt(
            voice_id=voice_id,
            ref_audio_path=ref_audio,
            ref_transcript=ref_text,
            x_vector_only=False,
            raw_prompt_payload=reconstructed_items,
        )

        dec_params = DecodingParams(temperature=0.7, top_p=0.8, top_k=20, seed=42)
        gen_res = adapter.synthesize_voice_clone(
            text="Verifica di sintesi neurale con prompt salvato dalla UI.",
            voice_prompt=vp,
            language="italian",
            params=dec_params,
        )

        assert gen_res.audio.duration_sec > 1.0
        assert gen_res.audio.sample_rate == 24000
        assert gen_res.rtf > 0.0
        assert gen_res.metadata["mode"] == "clone"

        # 7. Unload model
        adapter.unload()
        assert adapter.state == LifecycleState.UNLOADED

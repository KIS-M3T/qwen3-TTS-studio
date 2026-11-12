"""Concrete adapter for Alibaba Qwen3-TTS models (WP2, C1, C5, F5, G1, G3, H1, H5, H6)."""

import gc
import logging
import os
from pathlib import Path
import shutil
import time
from typing import Any, Dict, List, Optional
import uuid
import numpy as np
import torch

from src.engine.base import BaseTTSEngine
from src.engine.errors import ModelLoadError, EngineOOMError, ConfigurationError
from src.engine.lifecycle import ModelLifecycleManager
from src.engine.types import (
    AudioBuffer,
    DecodingParams,
    GenerationResult,
    LifecycleState,
    MemorySnapshot,
    VoicePrompt,
)
from src.instrumentation.manifest import get_environment_manifest
from src.instrumentation.memory import get_memory_snapshot

logger = logging.getLogger(__name__)

KNOWN_REF_TRANSCRIPTS = {
    "ref_voice_it_01.wav": "Pone sotto la propria egida il campionato e la Nazionale libanese.",
    "ref_voice_it_02.wav": "A tutt'oggi viene venduto su eBay a cifre esorbitanti.",
}

MODEL_PATHS = {
    "1.7B-Base": "./Qwen3-TTS-12Hz-1.7B-Base",
    "0.6B-Base": "./Qwen3-TTS-12Hz-0.6B-Base",
    "1.7B-CustomVoice": "./Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "0.6B-CustomVoice": "./Qwen3-TTS-12Hz-0.6B-CustomVoice",
    "1.7B-VoiceDesign": "./Qwen3-TTS-12Hz-1.7B-VoiceDesign",
}

# Standard vendor supported decoding kwargs for Qwen3-TTS
SUPPORTED_VENDOR_KWARGS = {
    "temperature",
    "top_p",
    "top_k",
    "repetition_penalty",
    "max_new_tokens",
    "subtalker_temperature",
    "subtalker_top_p",
    "subtalker_top_k",
    "subtalker_dosample",
    "do_sample",
    "instruct",
    "non_streaming_mode",
}


def _check_qwen_tts_version() -> None:
    """Verify qwen-tts package >= 0.1.1 for 12Hz tokenizer padding bugfix."""
    try:
        from importlib.metadata import version as pkg_version
        from packaging.version import Version

        installed = pkg_version("qwen-tts")
        if Version(installed) < Version("0.1.1"):
            raise RuntimeError(
                f"qwen-tts {installed} is installed, but >= 0.1.1 is required to avoid audio corruption."
            )
    except ImportError:
        pass


class Qwen3EngineAdapter(BaseTTSEngine):
    """Adapter wrapping vendor Qwen3TTSModel within the modular architecture."""

    def __init__(self, model_id: str, model_path: Optional[str] = None):
        self._model_id = model_id
        self.instance_id = f"{model_id}_{uuid.uuid4().hex[:8]}"
        self.model_path = model_path or MODEL_PATHS.get(model_id, f"./Qwen3-TTS-12Hz-{model_id}")
        self.lifecycle = ModelLifecycleManager(model_id, owner_id=self.instance_id)
        self.model: Any = None
        self.cached_prompt: Any = None
        self.cached_prompt_ref: Optional[str] = None
        self.manifest = get_environment_manifest()

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def state(self) -> LifecycleState:
        return self.lifecycle.state

    def _release_model_refs(self) -> None:
        """Sever all model and cache references before GC / empty cache (G3, H1)."""
        self.model = None
        self.cached_prompt = None
        self.cached_prompt_ref = None

        gc.collect()
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass

    def load(self, device: str = "cuda:0", dtype_str: str = "bfloat16") -> None:
        """Transactionally load model weights with complete rollback on failure (H6)."""
        self.lifecycle.begin_loading()

        try:
            # 1. Version check
            _check_qwen_tts_version()

            # 2. Enforce single resident model policy bound to instance identity (H5)
            self.lifecycle.acquire_residency(self.instance_id, self.model_id)

            # 3. Check preventive VRAM requirements (G3)
            required_vram_mb = 3500.0 if "0.6b" in self.model_id.lower() else 7500.0
            if not self.lifecycle.check_vram_requirement(required_vram_mb):
                raise EngineOOMError(
                    f"Insufficient VRAM for {self.model_id}: requires at least {required_vram_mb} MB available.",
                    is_recoverable=True,
                )

            # 4. Check model path existence
            if not os.path.exists(self.model_path):
                raise ModelLoadError(f"Model path not found: {self.model_path}")

            # 5. Ensure speech tokenizer weights exist in destination
            tok_src = Path("Qwen3-TTS-Tokenizer-12Hz/model.safetensors")
            tok_dst = Path(self.model_path) / "speech_tokenizer" / "model.safetensors"
            if tok_src.exists() and not tok_dst.exists():
                tok_dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(tok_src, tok_dst)

            # 6. Load model weights
            from qwen_tts import Qwen3TTSModel

            target_dtype = torch.bfloat16 if dtype_str == "bfloat16" and torch.cuda.is_bf16_supported() else torch.float32

            load_kwargs: Dict[str, Any] = {
                "device_map": device,
                "dtype": target_dtype,
            }
            if "cuda" in device:
                load_kwargs["attn_implementation"] = "sdpa"

            self.model = Qwen3TTSModel.from_pretrained(self.model_path, **load_kwargs)
            self.model.model.eval()

            # 7. Apply baseline anti-truncation patch wrapper
            self._apply_anti_truncation_patch()

            self.lifecycle.mark_ready()

        except torch.cuda.OutOfMemoryError as oom:
            err = self.lifecycle.handle_oom(oom, release_callback=self._release_model_refs)
            raise err from oom
        except Exception as e:
            # Transactional rollback: guarantees state returns to UNLOADED without orphan refs (H6)
            self._release_model_refs()
            self.lifecycle.mark_unloaded()
            if isinstance(e, (EngineOOMError, ModelLoadError)):
                raise e
            raise ModelLoadError(f"Failed to load {self.model_id}: {e}") from e

    def unload(self) -> None:
        """Unload model, release VRAM, and release residency (G3, H5)."""
        self._release_model_refs()
        self.lifecycle.mark_unloaded()

    def _apply_anti_truncation_patch(self) -> None:
        """Apply baseline min_new_tokens=60 wrapper with explicit F5 validation."""
        if not hasattr(self.model, "model") or not hasattr(self.model.model, "generate"):
            return

        original_generate = self.model.model.generate

        def patched_generate(*args, **kwargs):
            min_tokens = kwargs.get("min_new_tokens", 60)
            effective_min = max(min_tokens, 60)

            # F5: Prevent incompatible configuration with explicit ValueError
            max_tokens = kwargs.get("max_new_tokens")
            if max_tokens is not None and max_tokens < effective_min:
                raise ValueError(
                    f"Invalid configuration: max_new_tokens ({max_tokens}) cannot be less than "
                    f"anti-truncation requirement min_new_tokens ({effective_min})."
                )

            kwargs["min_new_tokens"] = effective_min
            return original_generate(*args, **kwargs)

        self.model.model.generate = patched_generate

    def synthesize_base(
        self,
        text: str,
        language: str = "italian",
        params: Optional[DecodingParams] = None,
    ) -> GenerationResult:
        """Base zero-shot synthesis."""
        return self._generate_internal(text=text, language=language, params=params, mode="base")

    def synthesize_voice_clone(
        self,
        text: str,
        voice_prompt: VoicePrompt,
        language: str = "italian",
        params: Optional[DecodingParams] = None,
    ) -> GenerationResult:
        """Voice cloning synthesis."""
        return self._generate_internal(
            text=text,
            language=language,
            params=params,
            mode="clone",
            voice_prompt=voice_prompt,
        )

    def synthesize_custom_voice(
        self,
        text: str,
        speaker_id: str,
        language: str = "italian",
        params: Optional[DecodingParams] = None,
    ) -> GenerationResult:
        """Multi-speaker CustomVoice synthesis."""
        return self._generate_internal(
            text=text,
            language=language,
            params=params,
            mode="custom",
            speaker_id=speaker_id,
        )

    def create_voice_prompt(
        self,
        ref_audio_path: str,
        ref_transcript: Optional[str] = None,
        x_vector_only: bool = False,
    ) -> VoicePrompt:
        """Extract conditioning voice prompt embedding from reference clip (G2)."""
        if self.state != LifecycleState.READY or self.model is None:
            raise RuntimeError(f"Model {self.model_id} must be loaded and READY to extract voice prompts.")

        prompt_items = self.model.create_voice_clone_prompt(
            ref_audio=ref_audio_path,
            ref_text=ref_transcript,
            x_vector_only_mode=x_vector_only,
        )

        return VoicePrompt(
            voice_id=Path(ref_audio_path).stem,
            ref_audio_path=Path(ref_audio_path),
            ref_transcript=ref_transcript,
            x_vector_only=x_vector_only,
            serialized_format="safetensors",
            raw_prompt_payload=prompt_items,
        )

    def _generate_internal(
        self,
        text: str,
        language: str,
        params: Optional[DecodingParams],
        mode: str,
        voice_prompt: Optional[VoicePrompt] = None,
        speaker_id: Optional[str] = None,
    ) -> GenerationResult:
        """Internal execution helper with parameter forwarding, seed isolation, and OOM intercept (G1, G3, H1, H5)."""
        if self.state != LifecycleState.READY or self.model is None:
            raise RuntimeError(f"Model {self.model_id} is not in READY state (current: {self.state})")

        p = params or DecodingParams()

        # G1: Parameter extraction and validation
        requested_params: Dict[str, Any] = {
            "temperature": p.temperature,
            "top_p": p.top_p,
            "top_k": p.top_k,
            "repetition_penalty": p.repetition_penalty,
            "min_new_tokens": p.min_new_tokens,
            "max_new_tokens": p.max_new_tokens,
            "seed": p.seed,
            **p.subtalker_params,
        }

        # Build kwargs for vendor API
        applied_params: Dict[str, Any] = {
            "temperature": float(p.temperature),
            "top_p": float(p.top_p),
            "top_k": int(p.top_k),
            "repetition_penalty": float(p.repetition_penalty),
            "max_new_tokens": int(p.max_new_tokens),
        }
        for sub_k in (
            "subtalker_temperature",
            "subtalker_top_p",
            "subtalker_top_k",
            "subtalker_dosample",
            "instruct",
            "non_streaming_mode",
        ):
            if sub_k in p.subtalker_params:
                applied_params[sub_k] = p.subtalker_params[sub_k]

        unsupported_params: Dict[str, Any] = {
            k: v for k, v in p.subtalker_params.items()
            if k not in SUPPORTED_VENDOR_KWARGS
        }
        if unsupported_params:
            logger.warning(f"Unsupported decoding parameters ignored: {list(unsupported_params.keys())}")

        # H5: Critical section for GPU generation and RNG state isolation
        with ModelLifecycleManager.gpu_generation_lock():
            prev_rng_state = None
            prev_cuda_rng_state = None

            if p.seed is not None:
                try:
                    prev_rng_state = torch.get_rng_state()
                    if torch.cuda.is_available():
                        prev_cuda_rng_state = torch.cuda.get_rng_state_all()
                    torch.manual_seed(p.seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(p.seed)
                except Exception as rng_err:
                    logger.warning(f"Unable to isolate RNG seed: {rng_err}")

            try:
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()

                t0 = time.perf_counter()

                vendor_kwargs: Dict[str, Any] = dict(applied_params)

                try:
                    if mode == "custom":
                        wavs, sr = self.model.generate_custom_voice(
                            text=text,
                            speaker=speaker_id or "vivian",
                            language=language,
                            **vendor_kwargs,
                        )
                    elif mode == "clone":
                        prompt_payload = voice_prompt.raw_prompt_payload if voice_prompt else None
                        if prompt_payload is None and voice_prompt:
                            prompt_payload = self.model.create_voice_clone_prompt(
                                ref_audio=str(voice_prompt.ref_audio_path),
                                ref_text=voice_prompt.ref_transcript,
                                x_vector_only_mode=voice_prompt.x_vector_only,
                            )
                        wavs, sr = self.model.generate_voice_clone(
                            text=text,
                            voice_clone_prompt=prompt_payload,
                            language=language,
                            **vendor_kwargs,
                        )
                    else:
                        # Base zero-shot synthesis: use default technical reference voice prompt
                        if self.cached_prompt is None:
                            default_ref = Path("bench/reference_voices/ref_voice_it_01.wav")
                            ref_text = KNOWN_REF_TRANSCRIPTS.get(default_ref.name)
                            if default_ref.exists():
                                self.cached_prompt = self.model.create_voice_clone_prompt(
                                    ref_audio=str(default_ref),
                                    ref_text=ref_text,
                                )
                            else:
                                raise RuntimeError(f"Reference voice audio required for Base model {self.model_id}.")
                        wavs, sr = self.model.generate_voice_clone(
                            text=text,
                            voice_clone_prompt=self.cached_prompt,
                            language=language,
                            **vendor_kwargs,
                        )
                except torch.cuda.OutOfMemoryError as oom:
                    # H1: Intercept OOM during generation, activate lifecycle recovery and release references
                    logger.error(f"OOM during inference execution in {self.model_id}: {oom}")
                    err = self.lifecycle.handle_oom(oom, release_callback=self._release_model_refs)
                    raise err from oom

                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                inf_time = time.perf_counter() - t0

                raw_audio = np.array(wavs[0], dtype=np.float32)

                max_abs = float(np.max(np.abs(raw_audio))) if raw_audio.size > 0 else 0.0
                if max_abs > 1.0:
                    raw_audio = raw_audio / max_abs

                audio_buf = AudioBuffer(samples=raw_audio, sample_rate=sr)
                rtf = round(inf_time / audio_buf.duration_sec, 4)

                mem_dict = get_memory_snapshot()
                mem_snap = MemorySnapshot(
                    torch_allocated_bytes=int(mem_dict["torch_allocated_mb"] * 1024 * 1024),
                    torch_reserved_bytes=int(mem_dict["torch_reserved_mb"] * 1024 * 1024),
                    torch_peak_allocated_bytes=int(mem_dict["torch_peak_allocated_mb"] * 1024 * 1024),
                    driver_used_bytes_at_end=int(mem_dict["driver_used_mb_at_end"] * 1024 * 1024) if mem_dict["driver_used_mb_at_end"] else None,
                    baseline_process_bytes=int(mem_dict["baseline_process_mb"] * 1024 * 1024) if mem_dict.get("baseline_process_mb") is not None else None,
                )

                metadata: Dict[str, Any] = {
                    "requested_params": requested_params,
                    "applied_params": applied_params,
                    "unsupported_params": unsupported_params,
                    "seed_applied": p.seed,
                    "mode": mode,
                }

                return GenerationResult(
                    audio=audio_buf,
                    inference_time_sec=round(inf_time, 4),
                    rtf=rtf,
                    memory=mem_snap,
                    model_version=self.model_id,
                    decoding_params=p,
                    environment_manifest_sha256=self.manifest.get("manifest_sha256", "unknown"),
                    metadata=metadata,
                )
            finally:
                # G1, H5: Restore RNG state
                if prev_rng_state is not None:
                    try:
                        torch.set_rng_state(prev_rng_state)
                        if prev_cuda_rng_state is not None and torch.cuda.is_available():
                            torch.cuda.set_rng_state_all(prev_cuda_rng_state)
                    except Exception:
                        pass

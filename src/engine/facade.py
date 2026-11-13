"""High-level Facade decoupling UI layers from engine internals (G5).

Provides a unified interface for Gradio UI and service endpoints,
coordinating lifecycle residency, parameter validation, prompt management,
and non-regression fallbacks.
"""

import logging
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np

from src.engine.adapters.qwen3_adapter import Qwen3EngineAdapter
from src.engine.lifecycle import ModelLifecycleManager
from src.engine.prompt_manager import (
    DEFAULT_SAVED_VOICES_DIR,
    load_voice_prompt_items,
    save_voice_prompt_items,
    validate_voice_name,
)
from src.engine.types import (
    AudioBuffer,
    DecodingParams,
    GenerationResult,
    LifecycleState,
    VoicePrompt,
)

logger = logging.getLogger(__name__)


class TTSEngineFacade:
    """Unified Facade for speech synthesis services with thread-safe lifecycle coordination (G5, J2)."""

    def __init__(self, default_device: str = "cuda:0", default_dtype: str = "bfloat16"):
        self.default_device = default_device
        self.default_dtype = default_dtype
        self._active_adapter: Optional[Qwen3EngineAdapter] = None

        # Level 1 coordinator lock shared with legacy model loader (J2, K2)
        try:
            import audio.model_loader as ml
            self._coordinator_lock = ml.get_global_execution_lock()
            ml.register_facade_unload_callback(self.unload)
        except Exception:
            self._coordinator_lock = threading.RLock()

    def close(self) -> None:
        """Release coordinator hooks upon facade shutdown (K2)."""
        try:
            import audio.model_loader as ml
            ml.unregister_facade_unload_callback(self.unload)
        except Exception:
            pass

    def unload_legacy_models(self) -> None:
        """Purge models from audio.model_loader legacy cache to prevent VRAM overcommit (J2, K2)."""
        try:
            import audio.model_loader as ml
            with ml._model_lock:
                for m_name in list(ml.loaded_models.keys()):
                    if ml.get_active_lease_count(m_name) == 0:
                        ml._unload_model(m_name)
                    else:
                        logger.warning(f"Skipping unload of legacy model '{m_name}': active lease in progress (K2)")
        except Exception as e:
            logger.debug(f"Legacy model unload skipped or failed: {e}")

    def get_or_load_adapter(
        self,
        model_id: str,
        device: Optional[str] = None,
        dtype_str: Optional[str] = None,
    ) -> Qwen3EngineAdapter:
        """Obtain resident adapter, unloading previous model if different (thread-safe, G3, G5, J2)."""
        with self._coordinator_lock:
            target_device = device or self.default_device
            target_dtype = dtype_str or self.default_dtype

            if self._active_adapter is not None:
                if self._active_adapter.model_id == model_id and self._active_adapter.state == LifecycleState.READY:
                    return self._active_adapter
                else:
                    logger.info(f"Unloading current resident model '{self._active_adapter.model_id}' before loading '{model_id}'")
                    self._active_adapter.unload()
                    self._active_adapter = None

            self.unload_legacy_models()
            adapter = Qwen3EngineAdapter(model_id=model_id)
            adapter.load(device=target_device, dtype_str=target_dtype)
            self._active_adapter = adapter
            return adapter

    @contextmanager
    def use_adapter(
        self,
        model_id: str,
        device: Optional[str] = None,
        dtype_str: Optional[str] = None,
    ):
        """Context manager acquiring the concurrency coordinator lock during model access & generation (J2).

        Guarantees that while an inference or generation context is executing,
        no concurrent thread can unload or replace this resident adapter.
        """
        with self._coordinator_lock:
            adapter = self.get_or_load_adapter(model_id=model_id, device=device, dtype_str=dtype_str)
            try:
                yield adapter
            finally:
                pass

    def synthesize(
        self,
        text: str,
        model_id: str,
        mode: str = "base",               # "base", "clone", "custom"
        language: str = "italian",
        voice_prompt: Optional[VoicePrompt] = None,
        speaker_id: Optional[str] = None,
        params: Optional[DecodingParams] = None,
    ) -> GenerationResult:
        """Unified synthesis entrypoint for UI and applications (thread-safe, G5, J2).

        Args:
            text: Text to synthesize.
            model_id: Model checkpoint ID.
            mode: Mode ('base', 'clone', 'custom').
            language: Synthesis language.
            voice_prompt: Conditioning VoicePrompt for 'clone' mode.
            speaker_id: Speaker ID for 'custom' mode.
            params: Decoding parameters.

        Returns:
            GenerationResult: Validated result containing AudioBuffer and metadata.
        """
        with self.use_adapter(model_id) as adapter:
            if mode == "clone":
                if voice_prompt is None:
                    raise ValueError("voice_prompt is required for mode='clone'")
                return adapter.synthesize_voice_clone(
                    text=text,
                    voice_prompt=voice_prompt,
                    language=language,
                    params=params,
                )
            elif mode == "custom":
                return adapter.synthesize_custom_voice(
                    text=text,
                    speaker_id=speaker_id or "vivian",
                    language=language,
                    params=params,
                )
            else:
                return adapter.synthesize_base(
                    text=text,
                    language=language,
                    params=params,
                )

    def save_voice_profile(
        self,
        voice_id: str,
        prompt_items: Any,
        description: str = "",
        ref_audio_paths: Optional[List[str]] = None,
        base_dir: Path = DEFAULT_SAVED_VOICES_DIR,
    ) -> Path:
        """Securely save a voice profile in safetensors format (G2, G5)."""
        return save_voice_prompt_items(
            items=prompt_items,
            voice_id=voice_id,
            description=description,
            ref_audio_paths=ref_audio_paths,
            base_dir=base_dir,
        )

    def load_voice_profile(
        self,
        voice_id: str,
        base_dir: Path = DEFAULT_SAVED_VOICES_DIR,
    ) -> List[Any]:
        """Securely load voice prompt items without pickle (G2, G5)."""
        target_dir = base_dir / validate_voice_name(voice_id)
        return load_voice_prompt_items(target_dir, base_dir)

    def unload(self) -> None:
        """Explicitly unload current resident adapter and release memory in a thread-safe manner (J2)."""
        with self._coordinator_lock:
            if self._active_adapter is not None:
                self._active_adapter.unload()
                self._active_adapter = None

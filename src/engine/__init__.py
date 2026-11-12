"""Core Audio Engine package for Italian TTS Studio."""

from src.engine.base import BaseTTSEngine
from src.engine.adapters.qwen3_adapter import Qwen3EngineAdapter
from src.engine.lifecycle import ModelLifecycleManager
from src.engine.prompt_manager import load_voice_prompt, save_voice_prompt
from src.engine.types import (
    AudioBuffer,
    DecodingParams,
    GenerationResult,
    LifecycleState,
    MemorySnapshot,
    VoicePrompt,
)

__all__ = [
    "BaseTTSEngine",
    "Qwen3EngineAdapter",
    "ModelLifecycleManager",
    "load_voice_prompt",
    "save_voice_prompt",
    "AudioBuffer",
    "DecodingParams",
    "GenerationResult",
    "LifecycleState",
    "MemorySnapshot",
    "VoicePrompt",
]

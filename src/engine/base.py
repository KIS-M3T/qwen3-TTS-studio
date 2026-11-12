"""Abstract interface for TTS engines with explicit multimodal contracts (C5)."""

from abc import ABC, abstractmethod
from typing import Optional

from src.engine.types import (
    DecodingParams,
    GenerationResult,
    LifecycleState,
    VoicePrompt,
)


class BaseTTSEngine(ABC):
    """Abstract interface of the speech synthesis engine."""

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Identifier of the model."""
        pass

    @property
    @abstractmethod
    def state(self) -> LifecycleState:
        """Current lifecycle state."""
        pass

    @abstractmethod
    def load(self, device: str = "cuda:0", dtype_str: str = "bfloat16") -> None:
        """Load model weights and allocate hardware resources."""
        pass

    @abstractmethod
    def unload(self) -> None:
        """Unload model weights and release active allocations."""
        pass

    @abstractmethod
    def synthesize_base(
        self,
        text: str,
        language: str = "italian",
        params: Optional[DecodingParams] = None,
    ) -> GenerationResult:
        """Base zero-shot synthesis without prompt conditioning."""
        pass

    @abstractmethod
    def synthesize_voice_clone(
        self,
        text: str,
        voice_prompt: VoicePrompt,
        language: str = "italian",
        params: Optional[DecodingParams] = None,
    ) -> GenerationResult:
        """Synthesis with voice cloning conditioning."""
        pass

    @abstractmethod
    def synthesize_custom_voice(
        self,
        text: str,
        speaker_id: str,
        language: str = "italian",
        params: Optional[DecodingParams] = None,
    ) -> GenerationResult:
        """Multi-speaker preset voice generation for CustomVoice checkpoints."""
        pass

    @abstractmethod
    def create_voice_prompt(
        self,
        ref_audio_path: str,
        ref_transcript: Optional[str] = None,
        x_vector_only: bool = False,
    ) -> VoicePrompt:
        """Extract conditioning voice prompt embedding from reference clip."""
        pass

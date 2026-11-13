"""Core Audio Engine data types, validated AudioBuffer, and lifecycle states (C1, F2)."""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np


class LifecycleState(str, Enum):
    """Formal lifecycle states of the TTS engine."""
    UNLOADED = "unloaded"
    LOADING = "loading"
    READY = "ready"
    FAILED = "failed"
    RECOVERING = "recovering"


class EngineDevice(str, Enum):
    CUDA_0 = "cuda:0"
    CPU = "cpu"


class AttentionBackend(str, Enum):
    SDPA = "sdpa"
    EAGER = "eager"
    FLASH_ATTENTION_2 = "flash_attention_2"


@dataclass(frozen=True)
class AudioBuffer:
    """1D normalized mono float32 audio buffer with defensive copy and contract validation (F2)."""
    samples: np.ndarray
    sample_rate: int
    duration_sec: float

    def __init__(self, samples: np.ndarray, sample_rate: int, duration_sec: Optional[float] = None):
        # 1. Base type validation
        if not isinstance(samples, np.ndarray):
            raise TypeError(f"samples must be an np.ndarray, got {type(samples)}")

        # 2. Reject empty arrays
        if samples.size == 0:
            raise ValueError("samples cannot be an empty array (size == 0)")

        # 3. Reject non-float32 types (explicit conversion required for PCM int16)
        if samples.dtype != np.float32:
            raise TypeError(
                f"samples must have dtype float32 normalized in [-1.0, 1.0], got {samples.dtype}. "
                "For PCM int16 arrays, explicitly convert via (samples.astype(np.float32) / 32768.0) "
                "before instantiating AudioBuffer."
            )

        # 4. Dimension validation (1D mono)
        if samples.ndim != 1:
            raise ValueError(f"samples must be 1D mono, got ndim={samples.ndim} (shape={samples.shape})")

        # 5. Sample rate validation
        if not isinstance(sample_rate, int) or sample_rate <= 0:
            raise ValueError(f"sample_rate must be a positive integer, got {sample_rate}")

        # 6. Defensive copy isolating instance from caller's memory
        samples_copy = np.array(samples, dtype=np.float32, copy=True)

        # 7. Finite check (no NaN or Inf)
        if not np.all(np.isfinite(samples_copy)):
            raise ValueError("Audio samples contain non-finite values (NaN or Inf)")

        # 8. Range validation [-1.0, 1.0]
        min_val = float(np.min(samples_copy))
        max_val = float(np.max(samples_copy))
        if min_val < -1.0 or max_val > 1.0:
            raise ValueError(
                f"Audio samples outside normalized range [-1.0, 1.0]: "
                f"min={min_val:.5f}, max={max_val:.5f}. Normalize or clamp audio before instantiation."
            )

        # 9. Coherence between sample count and duration
        calc_duration = len(samples_copy) / float(sample_rate)
        if duration_sec is None:
            duration_sec = calc_duration
        else:
            if duration_sec <= 0:
                raise ValueError(f"duration_sec must be positive, got {duration_sec}")
            if abs(calc_duration - duration_sec) > 1e-3:
                raise ValueError(
                    f"Duration mismatch: declared {duration_sec:.4f}s, calculated {calc_duration:.4f}s (delta > 1ms)"
                )

        # 10. Mark array as read-only
        samples_copy.flags.writeable = False

        object.__setattr__(self, "samples", samples_copy)
        object.__setattr__(self, "sample_rate", sample_rate)
        object.__setattr__(self, "duration_sec", duration_sec)


@dataclass(frozen=True)
class VoicePrompt:
    """Conditioning voice prompt for voice cloning."""
    voice_id: str
    ref_audio_path: Optional[Path] = None
    ref_transcript: Optional[str] = None
    x_vector_only: bool = False
    serialized_format: str = "safetensors"
    raw_prompt_payload: Any = field(default=None, repr=False)


@dataclass(frozen=True)
class DecodingParams:
    """Deterministic decoding parameters."""
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    repetition_penalty: float = 1.05
    min_new_tokens: int = 60      # Baseline M0 anti-truncation patch
    max_new_tokens: int = 2048
    seed: Optional[int] = 42
    subtalker_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # F5: Prevent incompatible configuration
        if self.max_new_tokens < self.min_new_tokens:
            raise ValueError(
                f"Invalid configuration: max_new_tokens ({self.max_new_tokens}) cannot be less than "
                f"anti-truncation requirement min_new_tokens ({self.min_new_tokens})."
            )


@dataclass(frozen=True)
class MemorySnapshot:
    """Multi-tier GPU memory snapshot (R3, C6)."""
    torch_allocated_bytes: int
    torch_reserved_bytes: int
    torch_peak_allocated_bytes: int
    driver_used_bytes_at_end: Optional[int]
    baseline_process_bytes: Optional[int] = None


@dataclass(frozen=True)
class GenerationResult:
    """Result of speech synthesis generation."""
    audio: AudioBuffer
    inference_time_sec: float
    rtf: float
    memory: MemorySnapshot
    model_version: str
    decoding_params: DecodingParams
    environment_manifest_sha256: str
    metadata: Dict[str, Any] = field(default_factory=dict)


"""Exception taxonomy for the Core Audio Engine."""

from typing import Optional


class TTSEngineError(Exception):
    """Base exception for all TTS engine errors."""


class ModelLoadError(TTSEngineError):
    """Error during model discovery, initialization, or weight loading."""


class EngineOOMError(TTSEngineError):
    """CUDA Out Of Memory error during synthesis or loading."""

    def __init__(self, message: str, is_recoverable: bool = False, memory_delta_mb: Optional[float] = None):
        super().__init__(message)
        self.is_recoverable = is_recoverable
        self.memory_delta_mb = memory_delta_mb


class UnrecoverableCudaError(TTSEngineError):
    """Irreversible CUDA runtime or driver context fault requiring worker restart."""


class AudioContractError(TTSEngineError):
    """Violation of AudioBuffer contract or acoustic formatting."""


class SecurityError(TTSEngineError):
    """Security policy violation (e.g. untrusted pickle deserialization, path traversal)."""


class ConfigurationError(TTSEngineError):
    """Incompatible engine or generation configuration."""


class PromptValidationError(TTSEngineError):
    """Schema violation or corruption in voice prompt safetensors/metadata (J3)."""

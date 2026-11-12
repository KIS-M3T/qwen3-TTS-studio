"""Voice prompt management and secure non-executable serialization (F1)."""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

from src.engine.errors import SecurityError
from src.engine.types import VoicePrompt

logger = logging.getLogger(__name__)

DEFAULT_SAVED_VOICES_DIR = Path("saved_voices")


def validate_voice_directory(voice_dir: Path, base_dir: Path = DEFAULT_SAVED_VOICES_DIR) -> Path:
    """Ensure voice directory resides safely inside base_dir and is not a symlink.

    Args:
        voice_dir: Target directory path.
        base_dir: Base allowed root directory.

    Returns:
        Path: Resolved safe directory path.

    Raises:
        SecurityError: If path attempts directory traversal or is a symlink.
    """
    resolved_base = base_dir.resolve()
    resolved_target = voice_dir.resolve()

    if os.path.islink(voice_dir):
        raise SecurityError(f"Symlinks are not permitted for voice directories: {voice_dir}")

    try:
        resolved_target.relative_to(resolved_base)
    except ValueError:
        raise SecurityError(
            f"Path traversal detected: {voice_dir} is not within {base_dir}"
        )

    return resolved_target


def load_voice_prompt(
    voice_dir: Path,
    allow_legacy_pickle: bool = False,
    base_dir: Path = DEFAULT_SAVED_VOICES_DIR,
) -> VoicePrompt:
    """Load voice prompt from non-executable safetensors format or reject pickle (F1).

    Args:
        voice_dir: Directory containing voice prompt.
        allow_legacy_pickle: Whether to allow legacy .pkl (disabled by default).
        base_dir: Base allowed root directory.

    Returns:
        VoicePrompt: Validated voice prompt.

    Raises:
        SecurityError: If untrusted pickle is encountered or path is insecure.
        FileNotFoundError: If prompt files are missing.
    """
    safe_dir = validate_voice_directory(voice_dir, base_dir)

    safetensors_path = safe_dir / "voice_prompt.safetensors"
    metadata_path = safe_dir / "metadata.json"
    legacy_pkl_path = safe_dir / "prompt.pkl"

    # 1. Preferred secure format: safetensors + metadata
    if safetensors_path.exists() and metadata_path.exists():
        from safetensors.torch import load_file

        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        tensors = load_file(str(safetensors_path))
        ref_audio = safe_dir / metadata.get("ref_audio_filename", "ref_audio.wav")

        return VoicePrompt(
            voice_id=safe_dir.name,
            ref_audio_path=ref_audio,
            ref_transcript=metadata.get("ref_transcript"),
            x_vector_only=metadata.get("x_vector_only", False),
            serialized_format="safetensors",
            raw_prompt_payload=tensors,
        )

    # 2. Legacy pickle handling (strictly controlled)
    if legacy_pkl_path.exists():
        if not allow_legacy_pickle:
            raise SecurityError(
                f"Loading legacy pickle prompt at {legacy_pkl_path} is blocked by default (F1). "
                "Convert the prompt to safetensors using scripts/migrate_legacy_prompts.py "
                "or reconstruct from the original reference audio."
            )

    raise FileNotFoundError(f"No valid voice prompt found in {safe_dir}")


def save_voice_prompt(
    voice_id: str,
    prompt_payload: Dict[str, Any],
    ref_audio_path: Path,
    ref_transcript: Optional[str] = None,
    x_vector_only: bool = False,
    base_dir: Path = DEFAULT_SAVED_VOICES_DIR,
) -> Path:
    """Save voice prompt in non-executable safetensors format (F1).

    Args:
        voice_id: Identifier for the voice.
        prompt_payload: Dictionary of PyTorch tensors.
        ref_audio_path: Path to original reference audio WAV.
        ref_transcript: Optional reference transcript.
        x_vector_only: Whether x-vector mode was used.
        base_dir: Target root directory.

    Returns:
        Path: Directory where prompt was saved.
    """
    from safetensors.torch import save_file

    target_dir = base_dir / voice_id
    target_dir.mkdir(parents=True, exist_ok=True)
    safe_dir = validate_voice_directory(target_dir, base_dir)

    safetensors_path = safe_dir / "voice_prompt.safetensors"
    metadata_path = safe_dir / "metadata.json"

    # Filter tensors for safetensors saving
    tensor_dict = {
        k: v for k, v in prompt_payload.items()
        if hasattr(v, "cpu") and callable(v.cpu)
    }

    save_file(tensor_dict, str(safetensors_path))

    metadata = {
        "voice_id": voice_id,
        "ref_audio_filename": ref_audio_path.name,
        "ref_transcript": ref_transcript,
        "x_vector_only": x_vector_only,
        "tensor_keys": list(tensor_dict.keys()),
        "format": "safetensors",
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return safe_dir

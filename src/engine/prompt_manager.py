"""Voice prompt management and secure non-executable serialization (F1, G2).

Supports single-sample, multi-sample, x-vector-only, and ICL modes using
native safetensors format with zero arbitrary pickle deserialization in runtime paths.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import torch

from src.engine.errors import SecurityError
from src.engine.types import VoicePrompt

logger = logging.getLogger(__name__)

DEFAULT_SAVED_VOICES_DIR = Path("saved_voices")


def validate_voice_name(voice_name: str) -> str:
    """Validate and sanitize voice name before any filesystem access (G2).

    Args:
        voice_name: User-provided voice name.

    Returns:
        str: Sanitized safe voice name.

    Raises:
        SecurityError: If voice name contains path traversal or invalid characters.
    """
    stripped = voice_name.strip()
    if not stripped or stripped in (".", ".."):
        raise SecurityError(f"Invalid voice identifier: '{voice_name}'.")

    for c in stripped:
        if not (c.isalnum() or c in ("_", "-")):
            raise SecurityError(
                f"Invalid character '{c}' in voice identifier '{voice_name}'. Use only letters, digits, '_' or '-'."
            )
    return stripped


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


def save_voice_prompt_items(
    items: Any,
    voice_id: str,
    base_dir: Path = DEFAULT_SAVED_VOICES_DIR,
    description: str = "",
    ref_audio_paths: Optional[List[str]] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """Save VoiceClonePromptItem list in versioned, non-executable safetensors format (G2).

    Supports:
      - Single-sample voice prompt
      - Multi-sample voice prompt
      - X-vector-only mode
      - ICL mode with transcripts

    Args:
        items: List[VoiceClonePromptItem] or single VoiceClonePromptItem, or dict.
        voice_id: Voice identifier.
        base_dir: Root directory for saved voices.
        description: Optional voice description.
        ref_audio_paths: Optional list of relative reference audio paths.
        extra_metadata: Optional extra metadata dictionary.

    Returns:
        Path: Resolved target voice directory.
    """
    safe_name = validate_voice_name(voice_id)
    target_dir = base_dir / safe_name
    safe_dir = validate_voice_directory(target_dir, base_dir)
    safe_dir.mkdir(parents=True, exist_ok=True)

    safetensors_path = safe_dir / "voice_prompt.safetensors"
    metadata_path = safe_dir / "metadata.json"

    # Normalize items to list
    from qwen_tts.inference.qwen3_tts_model import VoiceClonePromptItem

    if isinstance(items, VoiceClonePromptItem):
        item_list = [items]
    elif isinstance(items, list):
        item_list = items
    elif isinstance(items, dict):
        # Dictionary of tensors (e.g. legacy structure)
        return _save_dict_prompt(items, safe_dir, voice_id, description, extra_metadata)
    else:
        raise ValueError(f"Unsupported prompt payload type for safetensors serialization: {type(items)}")

    tensor_dict: Dict[str, torch.Tensor] = {}
    samples_meta: List[Dict[str, Any]] = []

    for i, it in enumerate(item_list):
        if not isinstance(it, VoiceClonePromptItem):
            raise ValueError(f"Item at index {i} is not a VoiceClonePromptItem (got {type(it)})")

        spk_key = f"sample_{i}_ref_spk_embedding"
        tensor_dict[spk_key] = it.ref_spk_embedding.detach().cpu().contiguous()

        has_code = it.ref_code is not None
        if has_code:
            code_key = f"sample_{i}_ref_code"
            tensor_dict[code_key] = it.ref_code.detach().cpu().contiguous()

        sample_entry = {
            "index": i,
            "x_vector_only_mode": bool(it.x_vector_only_mode),
            "icl_mode": bool(it.icl_mode),
            "has_ref_code": has_code,
            "ref_text": it.ref_text if it.ref_text is not None else "",
        }
        samples_meta.append(sample_entry)

    from safetensors.torch import save_file

    save_file(tensor_dict, str(safetensors_path))

    full_metadata: Dict[str, Any] = {
        "schema_version": "1.0",
        "voice_id": safe_name,
        "format": "safetensors",
        "num_samples": len(item_list),
        "samples": samples_meta,
        "description": description,
        "ref_audio_paths": ref_audio_paths or [],
    }
    if extra_metadata:
        full_metadata.update(extra_metadata)

    # Atomic write of metadata
    tmp_meta = safe_dir / "metadata.json.tmp"
    with open(tmp_meta, "w", encoding="utf-8") as f:
        json.dump(full_metadata, f, indent=2, ensure_ascii=False)
    os.replace(tmp_meta, metadata_path)

    logger.info(f"Successfully saved voice prompt '{safe_name}' with {len(item_list)} sample(s) in safetensors format.")
    return safe_dir


def _save_dict_prompt(
    tensor_dict: Dict[str, Any],
    safe_dir: Path,
    voice_id: str,
    description: str,
    extra_metadata: Optional[Dict[str, Any]],
) -> Path:
    """Fallback helper to save a raw dict of tensors."""
    from safetensors.torch import save_file

    clean_tensors = {
        k: v.detach().cpu().contiguous()
        for k, v in tensor_dict.items()
        if hasattr(v, "detach") and callable(v.detach)
    }
    safetensors_path = safe_dir / "voice_prompt.safetensors"
    metadata_path = safe_dir / "metadata.json"

    save_file(clean_tensors, str(safetensors_path))

    full_metadata = {
        "schema_version": "1.0",
        "voice_id": voice_id,
        "format": "safetensors_dict",
        "tensor_keys": list(clean_tensors.keys()),
        "description": description,
    }
    if extra_metadata:
        full_metadata.update(extra_metadata)

    tmp_meta = safe_dir / "metadata.json.tmp"
    with open(tmp_meta, "w", encoding="utf-8") as f:
        json.dump(full_metadata, f, indent=2, ensure_ascii=False)
    os.replace(tmp_meta, metadata_path)
    return safe_dir


def save_prompt_to_safetensors_file(items: Any, filepath: Path) -> None:
    """Save prompt items directly to a single safetensors file with embedded JSON header (G2)."""
    from qwen_tts.inference.qwen3_tts_model import VoiceClonePromptItem
    from safetensors.torch import save_file

    if isinstance(items, VoiceClonePromptItem):
        item_list = [items]
    elif isinstance(items, list):
        item_list = items
    elif isinstance(items, dict):
        clean_tensors = {
            k: v.detach().cpu().contiguous()
            for k, v in items.items()
            if hasattr(v, "detach")
        }
        save_file(clean_tensors, str(filepath))
        return
    else:
        raise ValueError(f"Unsupported prompt items type: {type(items)}")

    tensor_dict: Dict[str, torch.Tensor] = {}
    samples_meta: List[Dict[str, Any]] = []

    for i, it in enumerate(item_list):
        spk_key = f"sample_{i}_ref_spk_embedding"
        tensor_dict[spk_key] = it.ref_spk_embedding.detach().cpu().contiguous()

        has_code = it.ref_code is not None
        if has_code:
            code_key = f"sample_{i}_ref_code"
            tensor_dict[code_key] = it.ref_code.detach().cpu().contiguous()

        samples_meta.append({
            "index": i,
            "x_vector_only_mode": bool(it.x_vector_only_mode),
            "icl_mode": bool(it.icl_mode),
            "has_ref_code": has_code,
            "ref_text": it.ref_text if it.ref_text is not None else "",
        })

    # Embedded header in safetensors metadata
    header_meta = {
        "schema_version": "1.0",
        "num_samples": str(len(item_list)),
        "samples_json": json.dumps(samples_meta),
    }
    save_file(tensor_dict, str(filepath), metadata=header_meta)


def load_voice_prompt_items(
    voice_dir: Path,
    base_dir: Path = DEFAULT_SAVED_VOICES_DIR,
) -> List[Any]:
    """Load VoiceClonePromptItem list reconstructed from safetensors (G2).

    Strictly rejects arbitrary pickle deserialization in runtime paths.

    Args:
        voice_dir: Directory containing voice prompt.
        base_dir: Root directory for validation.

    Returns:
        List[VoiceClonePromptItem]: Reconstructed native vendor prompt items.

    Raises:
        SecurityError: If untrusted legacy pickle is encountered.
        FileNotFoundError: If prompt files do not exist.
    """
    safe_dir = validate_voice_directory(voice_dir, base_dir)

    safetensors_path = safe_dir / "voice_prompt.safetensors"
    metadata_path = safe_dir / "metadata.json"
    legacy_pkl_path = safe_dir / "prompt.pkl"

    if safetensors_path.exists():
        from safetensors.torch import load_file
        from qwen_tts.inference.qwen3_tts_model import VoiceClonePromptItem

        tensors = load_file(str(safetensors_path))

        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        else:
            metadata = {}

        samples_meta = metadata.get("samples")
        if samples_meta is not None and isinstance(samples_meta, list):
            items: List[VoiceClonePromptItem] = []
            for s_info in samples_meta:
                idx = s_info["index"]
                spk_emb = tensors[f"sample_{idx}_ref_spk_embedding"]
                has_code = s_info.get("has_ref_code", True)
                ref_code = tensors.get(f"sample_{idx}_ref_code") if has_code else None
                xvec = s_info.get("x_vector_only_mode", False)
                icl = s_info.get("icl_mode", not xvec)
                ref_txt = s_info.get("ref_text")

                items.append(
                    VoiceClonePromptItem(
                        ref_code=ref_code,
                        ref_spk_embedding=spk_emb,
                        x_vector_only_mode=xvec,
                        icl_mode=icl,
                        ref_text=ref_txt if ref_txt else None,
                    )
                )
            return items
        else:
            # Single sample legacy or raw dict fallback
            if "sample_0_ref_spk_embedding" in tensors:
                spk_emb = tensors["sample_0_ref_spk_embedding"]
                ref_code = tensors.get("sample_0_ref_code")
                return [
                    VoiceClonePromptItem(
                        ref_code=ref_code,
                        ref_spk_embedding=spk_emb,
                        x_vector_only_mode=ref_code is None,
                        icl_mode=ref_code is not None,
                        ref_text=metadata.get("ref_transcript"),
                    )
                ]
            return tensors  # type: ignore

    if legacy_pkl_path.exists():
        raise SecurityError(
            f"Loading legacy pickle prompt at {legacy_pkl_path} is blocked by default (F1, G2). "
            "Please migrate this voice to the secure safetensors format using the offline migration tool: "
            "'python scripts/migrate_legacy_prompts.py' before using it."
        )

    raise FileNotFoundError(f"No voice_prompt.safetensors found in {safe_dir}")


def load_voice_prompt(
    voice_dir: Path,
    base_dir: Path = DEFAULT_SAVED_VOICES_DIR,
) -> VoicePrompt:
    """Load VoicePrompt container for engine adapter usage (G2)."""
    safe_dir = validate_voice_directory(voice_dir, base_dir)
    items = load_voice_prompt_items(safe_dir, base_dir)

    metadata_path = safe_dir / "metadata.json"
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    ref_audio = safe_dir / metadata.get("ref_audio_filename", "ref_audio.wav")

    return VoicePrompt(
        voice_id=safe_dir.name,
        ref_audio_path=ref_audio,
        ref_transcript=metadata.get("ref_transcript"),
        x_vector_only=metadata.get("x_vector_only", False),
        serialized_format="safetensors",
        raw_prompt_payload=items,
    )


# Alias for backwards compatibility
save_voice_prompt = save_voice_prompt_items


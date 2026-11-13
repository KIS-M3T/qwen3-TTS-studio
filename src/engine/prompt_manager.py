"""Voice prompt management and secure non-executable serialization (F1, G2).

Supports single-sample, multi-sample, x-vector-only, and ICL modes using
native safetensors format with zero arbitrary pickle deserialization in runtime paths.
"""

import json
import logging
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Union
import torch

from src.engine.errors import PromptValidationError, SecurityError
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

    Note on sample counts vs. reference audio files:
        The `num_samples` and prompt tensor structure strictly represent the number of
        `VoiceClonePromptItem` conditioning units (each containing a speaker embedding,
        optional reference code tokens, and optional transcription). When a workflow combines
        multiple raw audio recordings into fewer conditioning items (e.g., concatenating
        audio clips or averaging embeddings into a single VoiceClonePromptItem), `len(ref_audio_paths)`
        can legitimately differ from `num_samples`. The engine's strict validation governs
        the consistency between declared `num_samples` and the actual prompt tensors in
        `voice_prompt.safetensors`.

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

    header_meta = {
        "schema_version": "1.0",
        "num_samples": str(len(item_list)),
        "samples_json": json.dumps(samples_meta),
    }
    save_file(tensor_dict, str(safetensors_path), metadata=header_meta)

    full_metadata: Dict[str, Any] = {
        "schema_version": "1.0",
        "voice_id": safe_name,
        "format": "safetensors",
        "num_samples": len(item_list),
        "prompt_items": samples_meta,
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
    """Save prompt items directly to a single safetensors file with embedded JSON header (G2, J3)."""
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
        header_meta = {
            "schema_version": "1.0",
            "format": "safetensors_dict",
        }
        save_file(clean_tensors, str(filepath), metadata=header_meta)
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
    """Load VoiceClonePromptItem list reconstructed from safetensors (G2, H2, J3).

    Strictly rejects arbitrary pickle deserialization in runtime paths.
    Enforces strict schema validation on safetensors headers, companion metadata,
    tensor shapes, and dtypes.

    Args:
        voice_dir: Directory containing voice prompt.
        base_dir: Root directory for validation.

    Returns:
        List[VoiceClonePromptItem]: Reconstructed native vendor prompt items.

    Raises:
        SecurityError: If untrusted legacy pickle or path traversal is encountered.
        PromptValidationError: If schema, metadata, tensor keys, shapes, or dtypes are invalid.
        FileNotFoundError: If prompt files do not exist.
    """
    safe_dir = validate_voice_directory(voice_dir, base_dir)

    safetensors_path = safe_dir / "voice_prompt.safetensors"
    metadata_path = safe_dir / "metadata.json"
    legacy_pkl_path = safe_dir / "prompt.pkl"

    if safetensors_path.exists():
        from safetensors.torch import load_file
        from safetensors import safe_open
        from qwen_tts.inference.qwen3_tts_model import VoiceClonePromptItem

        # 1. Read and validate safetensors header metadata (G2, J3)
        header_samples: Optional[List[Dict[str, Any]]] = None
        declared_sample_count: Optional[int] = None
        try:
            with safe_open(str(safetensors_path), framework="pt") as sf:
                header_meta = sf.metadata()
                if header_meta:
                    if "num_samples" in header_meta:
                        try:
                            declared_sample_count = int(header_meta["num_samples"])
                        except (ValueError, TypeError) as ve:
                            raise PromptValidationError(
                                f"Invalid num_samples '{header_meta.get('num_samples')}' in safetensors header: {ve}"
                            ) from ve
                    if "samples_json" in header_meta:
                        try:
                            header_samples = json.loads(header_meta["samples_json"])
                            if not isinstance(header_samples, list):
                                raise PromptValidationError(
                                    f"samples_json in header must be a list, got {type(header_samples)}"
                                )
                        except json.JSONDecodeError as jde:
                            raise PromptValidationError(
                                f"Corrupted samples_json in safetensors header in {safetensors_path}: {jde}"
                            ) from jde
        except PromptValidationError:
            raise
        except Exception as e:
            raise PromptValidationError(
                f"Failed to open or parse safetensors header in {safetensors_path}: {e}"
            ) from e

        tensors = load_file(str(safetensors_path))

        # 2. Read and validate companion metadata.json (J3)
        metadata: Dict[str, Any] = {}
        if metadata_path.exists():
            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                if not isinstance(metadata, dict):
                    raise PromptValidationError(
                        f"metadata.json must contain a JSON object, got {type(metadata)}"
                    )
            except json.JSONDecodeError as jde:
                raise PromptValidationError(
                    f"Corrupted metadata.json in {safe_dir}: {jde}"
                ) from jde
            except PromptValidationError:
                raise
            except Exception as e:
                raise PromptValidationError(
                    f"Failed to read metadata.json in {safe_dir}: {e}"
                ) from e

        if declared_sample_count is None and "num_samples" in metadata:
            try:
                declared_sample_count = int(metadata["num_samples"])
            except (ValueError, TypeError) as ve:
                raise PromptValidationError(
                    f"Invalid num_samples in metadata.json: {metadata['num_samples']}"
                ) from ve

        def validate_spk_embedding(spk_tensor: Any, sample_idx: int) -> torch.Tensor:
            if spk_tensor is None:
                raise PromptValidationError(
                    f"Missing speaker embedding tensor for sample {sample_idx} in {safe_dir}"
                )
            if not isinstance(spk_tensor, torch.Tensor):
                raise PromptValidationError(
                    f"Speaker embedding for sample {sample_idx} must be a torch.Tensor, got {type(spk_tensor)}"
                )
            if spk_tensor.ndim not in (1, 2):
                raise PromptValidationError(
                    f"Invalid speaker embedding shape {spk_tensor.shape} for sample {sample_idx}; expected 1D or 2D"
                )
            if not spk_tensor.is_floating_point():
                raise PromptValidationError(
                    f"Invalid speaker embedding dtype {spk_tensor.dtype} for sample {sample_idx}; expected floating point"
                )
            return spk_tensor

        # 3. Determine samples specification (H2, J3, K3)
        # Count actual sample indices present in safetensors (spk_embedding or ref_code)
        actual_tensor_samples = set()
        for k in tensors.keys():
            m = re.match(r"^sample_(\d+)_(?:ref_spk_embedding|ref_code)$", k)
            if m:
                actual_tensor_samples.add(int(m.group(1)))
            elif k in ("ref_spk_embedding", "ref_code"):
                actual_tensor_samples.add(0)

        actual_tensor_count = len(actual_tensor_samples)
        if declared_sample_count is not None and actual_tensor_count > 0 and actual_tensor_count != declared_sample_count:
            raise PromptValidationError(
                f"Inconsistent sample count in {safe_dir}: declared {declared_sample_count}, but safetensors contains tensors for {actual_tensor_count} sample(s) (K3)"
            )

        raw_samples = header_samples
        if not raw_samples:
            raw_samples = metadata.get("prompt_items") or metadata.get("prompt_metadata") or metadata.get("samples")

        if raw_samples is not None and isinstance(raw_samples, list) and len(raw_samples) > 0:
            if actual_tensor_count > 0 and len(raw_samples) < actual_tensor_count:
                raise PromptValidationError(
                    f"Inconsistent sample count in {safe_dir}: metadata declares {len(raw_samples)} samples, but safetensors contains {actual_tensor_count} sample tensors without representation (K3)"
                )

            items: List[VoiceClonePromptItem] = []
            for i, s_info in enumerate(raw_samples):
                if not isinstance(s_info, dict):
                    raise PromptValidationError(
                        f"Sample info for entry {i} must be a dict, got {type(s_info)}"
                    )

                idx = s_info.get("index", i)

                # Find speaker embedding tensor
                spk_emb = tensors.get(f"sample_{idx}_ref_spk_embedding")
                if spk_emb is None:
                    spk_emb = tensors.get(f"sample_{i}_ref_spk_embedding")
                if spk_emb is None and i == 0:
                    spk_emb = tensors.get("ref_spk_embedding")

                spk_emb = validate_spk_embedding(spk_emb, i)

                # Extract mode declarations
                has_code_decl = s_info.get("has_ref_code")
                xvec_decl = s_info.get("x_vector_only_mode")
                icl_decl = s_info.get("icl_mode")

                # K3: Check contradictory declared flags
                if xvec_decl is True and icl_decl is True:
                    raise PromptValidationError(
                        f"Contradictory prompt schema for sample {i}: both x_vector_only_mode and icl_mode are True (K3)"
                    )
                if has_code_decl is False and icl_decl is True:
                    raise PromptValidationError(
                        f"Contradictory prompt schema for sample {i}: icl_mode is True but has_ref_code is False (K3)"
                    )
                if has_code_decl is True and xvec_decl is True:
                    raise PromptValidationError(
                        f"Contradictory prompt schema for sample {i}: x_vector_only_mode is True but has_ref_code is True (K3)"
                    )

                # Find ref code tensor
                ref_code = tensors.get(f"sample_{idx}_ref_code")
                if ref_code is None:
                    ref_code = tensors.get(f"sample_{i}_ref_code")
                if ref_code is None and i == 0:
                    ref_code = tensors.get("ref_code")

                # K3: Validate presence of ref_code against declared mode
                if has_code_decl is True and ref_code is None:
                    raise PromptValidationError(
                        f"Contradictory prompt schema for sample {i}: declared has_ref_code=True but ref_code tensor is missing in safetensors (K3)"
                    )
                if icl_decl is True and ref_code is None:
                    raise PromptValidationError(
                        f"Contradictory prompt schema for sample {i}: declared icl_mode=True but ref_code tensor is missing in safetensors (K3)"
                    )
                if xvec_decl is False and has_code_decl is not False and ref_code is None:
                    raise PromptValidationError(
                        f"Contradictory prompt schema for sample {i}: declared x_vector_only_mode=False but ref_code tensor is missing in safetensors (K3)"
                    )

                if ref_code is not None:
                    if has_code_decl is False:
                        raise PromptValidationError(
                            f"Contradictory prompt schema for sample {i}: ref_code tensor is present but declared has_ref_code=False (K3)"
                        )
                    if xvec_decl is True:
                        raise PromptValidationError(
                            f"Contradictory prompt schema for sample {i}: ref_code tensor is present but declared x_vector_only_mode=True (K3)"
                        )
                    if not isinstance(ref_code, torch.Tensor):
                        raise PromptValidationError(
                            f"Ref code for sample {i} must be a torch.Tensor, got {type(ref_code)}"
                        )

                # Reconcile text
                ref_txt = (
                    s_info.get("ref_text")
                    or s_info.get("transcript")
                    or metadata.get("ref_text")
                    or metadata.get("ref_transcript")
                )

                # Reconcile mode flags
                if "x_vector_only_mode" in s_info:
                    xvec = bool(s_info["x_vector_only_mode"])
                else:
                    xvec = bool(ref_code is None)

                if "icl_mode" in s_info:
                    icl = bool(s_info["icl_mode"])
                else:
                    icl = bool(not xvec and ref_code is not None)

                items.append(
                    VoiceClonePromptItem(
                        ref_code=ref_code,
                        ref_spk_embedding=spk_emb,
                        x_vector_only_mode=xvec,
                        icl_mode=icl,
                        ref_text=ref_txt if ref_txt else None,
                    )
                )

            if declared_sample_count is not None and len(items) != declared_sample_count:
                raise PromptValidationError(
                    f"Sample count mismatch in {safe_dir}: declared {declared_sample_count}, reconstructed {len(items)}"
                )

            if not items:
                raise PromptValidationError(f"No valid voice clone prompt items found in {safe_dir}")

            return items

        # 4. Fallback: discover samples directly from tensor keys if no sample list
        discovered_items: List[VoiceClonePromptItem] = []
        i = 0
        while True:
            spk_key = f"sample_{i}_ref_spk_embedding"
            if spk_key not in tensors:
                break
            spk_emb = validate_spk_embedding(tensors[spk_key], i)
            code_key = f"sample_{i}_ref_code"
            ref_code = tensors.get(code_key)
            if ref_code is not None and not isinstance(ref_code, torch.Tensor):
                raise PromptValidationError(
                    f"Ref code for sample {i} must be a torch.Tensor, got {type(ref_code)}"
                )
            discovered_items.append(
                VoiceClonePromptItem(
                    ref_code=ref_code,
                    ref_spk_embedding=spk_emb,
                    x_vector_only_mode=ref_code is None,
                    icl_mode=ref_code is not None,
                    ref_text=metadata.get("ref_text") or metadata.get("ref_transcript"),
                )
            )
            i += 1

        if discovered_items:
            if declared_sample_count is not None and len(discovered_items) != declared_sample_count:
                raise PromptValidationError(
                    f"Sample count mismatch in {safe_dir}: declared {declared_sample_count}, reconstructed {len(discovered_items)}"
                )
            return discovered_items

        # 5. Fallback for single flat tensor dictionary
        if "sample_0_ref_spk_embedding" in tensors or "ref_spk_embedding" in tensors:
            spk_emb = tensors.get("sample_0_ref_spk_embedding")
            if spk_emb is None:
                spk_emb = tensors.get("ref_spk_embedding")
            spk_emb = validate_spk_embedding(spk_emb, 0)
            ref_code = tensors.get("sample_0_ref_code")
            if ref_code is None:
                ref_code = tensors.get("ref_code")
            if ref_code is not None and not isinstance(ref_code, torch.Tensor):
                raise PromptValidationError(
                    f"Ref code must be a torch.Tensor, got {type(ref_code)}"
                )
            items = [
                VoiceClonePromptItem(
                    ref_code=ref_code,
                    ref_spk_embedding=spk_emb,
                    x_vector_only_mode=ref_code is None,
                    icl_mode=ref_code is not None,
                    ref_text=metadata.get("ref_text") or metadata.get("ref_transcript"),
                )
            ]
            if declared_sample_count is not None and len(items) != declared_sample_count:
                raise PromptValidationError(
                    f"Sample count mismatch in {safe_dir}: declared {declared_sample_count}, reconstructed {len(items)}"
                )
            return items

        raise PromptValidationError(
            f"No valid voice clone prompt items could be reconstructed from {safe_dir}. Safetensors lacks required embeddings."
        )

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


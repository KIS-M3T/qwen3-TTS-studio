"""Offline migration utility to convert legacy voice prompts to secure safetensors (F1)."""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("prompt_migrator")


def migrate_voice_directory(
    voice_dir: Path,
    reconstruct_with_model: bool = False,
    model_adapter: Any = None,
) -> bool:
    """Migrate a single voice directory to safetensors.

    Prioritizes clean reconstruction from source audio (Priority 1)
    over insecure pickle deserialization (Priority 2).
    """
    if os.path.islink(voice_dir):
        logger.error(f"Skipping symlinked directory: {voice_dir}")
        return False

    safetensors_path = voice_dir / "voice_prompt.safetensors"
    pkl_path = voice_dir / "prompt.pkl"
    meta_path = voice_dir / "metadata.json"

    if safetensors_path.exists():
        logger.info(f"[{voice_dir.name}] Already migrated to safetensors.")
        return True

    if not pkl_path.exists():
        logger.warning(f"[{voice_dir.name}] No prompt.pkl found. Nothing to migrate.")
        return False

    # Check file size limit (< 50 MiB)
    if pkl_path.stat().st_size > 50 * 1024 * 1024:
        logger.error(f"[{voice_dir.name}] prompt.pkl exceeds 50 MiB limit.")
        return False

    # Priority 1: Check if source audio is available for clean reconstruction
    ref_audio_candidates = [
        voice_dir / "ref_audio.wav",
        voice_dir / "reference.wav",
        voice_dir / "source.wav",
    ]
    existing_ref = next((p for p in ref_audio_candidates if p.exists()), None)

    if existing_ref and reconstruct_with_model and model_adapter:
        logger.info(f"[{voice_dir.name}] Source audio found at {existing_ref.name}. Reconstructing prompt cleanly...")
        transcript = None
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as mf:
                meta_data = json.load(mf)
                transcript = meta_data.get("ref_text")

        vp = model_adapter.create_voice_prompt(str(existing_ref), ref_transcript=transcript)
        from src.engine.prompt_manager import save_voice_prompt

        save_voice_prompt(
            voice_id=voice_dir.name,
            prompt_payload=vp.raw_prompt_payload,
            ref_audio_path=existing_ref,
            ref_transcript=transcript,
            base_dir=voice_dir.parent,
        )
        logger.info(f"[{voice_dir.name}] Successfully reconstructed from audio into safetensors!")
        return True

    # Priority 2: Attempt restricted weights_only=True load (F1)
    logger.info(f"[{voice_dir.name}] Attempting weights_only=True load...")
    try:
        data = torch.load(str(pkl_path), weights_only=True)
    except Exception as e:
        logger.error(
            f"[{voice_dir.name}] weights_only=True deserialization failed: {e}. "
            "File cannot be safely verified without arbitrary code execution risk. "
            "Marked as UNSUPPORTED_INSECURE_PICKLE (F1)."
        )
        return False

    # If data is dictionary of tensors, save to safetensors
    if isinstance(data, dict):
        from safetensors.torch import save_file

        tensors = {k: v for k, v in data.items() if hasattr(v, "cpu")}
        if tensors:
            save_file(tensors, str(safetensors_path))
            logger.info(f"[{voice_dir.name}] Migrated {len(tensors)} tensors to safetensors.")
            return True

    logger.warning(f"[{voice_dir.name}] Loaded data is not a tensor dictionary. Migration failed.")
    return False


def main():
    parser = argparse.ArgumentParser(description="Migrate legacy voice prompts to non-executable safetensors (F1)")
    parser.add_argument("--voices-dir", default="saved_voices", help="Path to saved_voices directory")
    args = parser.parse_args()

    voices_dir = Path(args.voices_dir)
    if not voices_dir.exists():
        logger.info(f"Directory {voices_dir} does not exist. Nothing to migrate.")
        return

    migrated_count = 0
    failed_count = 0

    for voice_dir in sorted(voices_dir.iterdir()):
        if voice_dir.is_dir():
            success = migrate_voice_directory(voice_dir)
            if success:
                migrated_count += 1
            else:
                failed_count += 1

    logger.info(f"Migration completed. Successful: {migrated_count}, Failed/Unsupported: {failed_count}")


if __name__ == "__main__":
    main()

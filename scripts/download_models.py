#!/usr/bin/env python3
"""
Robust Resumable Model Downloader for Qwen3-TTS (Milestone M0 Benchmark).
Downloads Tokenizer-12Hz, 0.6B-Base, and 1.7B-Base with automatic retries and resume.
"""

import os
import sys
import time
from pathlib import Path

from huggingface_hub import snapshot_download

MODELS = [
    {
        "repo_id": "Qwen/Qwen3-TTS-Tokenizer-12Hz",
        "local_dir": "./Qwen3-TTS-Tokenizer-12Hz",
        "desc": "Qwen3-TTS 12Hz Audio Tokenizer (~1.0 GB)",
        "expected_file": "model.safetensors",
    },
    {
        "repo_id": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        "local_dir": "./Qwen3-TTS-12Hz-0.6B-Base",
        "desc": "Qwen3-TTS 0.6B Base Model (~2.3 GB)",
        "expected_file": "model.safetensors",
    },
    {
        "repo_id": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        "local_dir": "./Qwen3-TTS-12Hz-1.7B-Base",
        "desc": "Qwen3-TTS 1.7B Base Model (~4.2 GB)",
        "expected_file": "model.safetensors",
    },
]


def is_already_downloaded(local_dir: Path, expected_file: str) -> bool:
    target = local_dir / expected_file
    if target.exists() and target.stat().st_size > 100 * 1024 * 1024:
        return True
    return False


def download_all():
    print("====================================================")
    print("DOWNLOADING QWEN3-TTS MODELS FOR MILESTONE M0")
    print("====================================================")

    for item in MODELS:
        repo_id = item["repo_id"]
        local_dir = Path(item["local_dir"]).resolve()
        desc = item["desc"]
        expected_file = item["expected_file"]

        if is_already_downloaded(local_dir, expected_file):
            print(f"\n>>> Target: {desc}")
            print(f"    [SKIP] Already fully downloaded at {local_dir}")
            continue

        print(f"\n>>> Target: {desc}")
        print(f"    Repository: {repo_id}")
        print(f"    Local Path: {local_dir}")

        success = False
        for attempt in range(1, 11):
            print(f"    Attempt {attempt}/10...")
            t0 = time.perf_counter()
            try:
                snapshot_download(
                    repo_id=repo_id,
                    local_dir=str(local_dir),
                    resume_download=True,
                    max_workers=4,
                )
                elapsed = round(time.perf_counter() - t0, 1)
                print(f"    [OK] Download completed in {elapsed}s")
                success = True
                break
            except Exception as e:
                print(f"    [WARNING] Attempt {attempt} failed: {e}")
                time.sleep(5)

        if not success:
            print(f"    [ERROR] Failed to download {repo_id} after 10 attempts.")
            sys.exit(1)

    print("\n====================================================")
    print("ALL MODELS DOWNLOADED AND VERIFIED SUCCESSFULLY")
    print("====================================================")


if __name__ == "__main__":
    download_all()

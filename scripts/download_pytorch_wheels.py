#!/usr/bin/env python3
"""
Resumable PyTorch CUDA Wheel Downloader.
Downloads torch and torchaudio wheels from PyTorch index with Range header and auto-resume.
"""

import sys
import time
from pathlib import Path
import requests

WHEELS = [
    {
        "url": "https://download.pytorch.org/whl/cu124/torch-2.6.0%2Bcu124-cp312-cp312-win_amd64.whl",
        "filename": "torch-2.6.0+cu124-cp312-cp312-win_amd64.whl",
        "expected_bytes": 2532302369,
    },
    {
        "url": "https://download.pytorch.org/whl/cu124/torchaudio-2.6.0%2Bcu124-cp312-cp312-win_amd64.whl",
        "filename": "torchaudio-2.6.0+cu124-cp312-cp312-win_amd64.whl",
        "expected_bytes": 4259840,
    },
]

DEST_DIR = Path("_wheels").resolve()
DEST_DIR.mkdir(parents=True, exist_ok=True)


def download_wheel(url: str, dest_file: Path, expected_size: int):
    print(f"\n>>> Downloading: {dest_file.name}")
    print(f"    Target size: {round(expected_size / (1024**2), 2)} MB")

    while True:
        current_size = dest_file.stat().st_size if dest_file.exists() else 0
        if current_size >= expected_size:
            print(f"    [OK] File already complete: {current_size} bytes")
            return

        headers = {}
        if current_size > 0:
            headers["Range"] = f"bytes={current_size}-"
            print(f"    Resuming from {round(current_size / (1024**2), 2)} MB...")

        try:
            with requests.get(url, headers=headers, stream=True, timeout=30) as r:
                if r.status_code not in (200, 206):
                    print(f"    [WARNING] Unexpected status code {r.status_code}, retrying...")
                    time.sleep(3)
                    continue

                mode = "ab" if current_size > 0 else "wb"
                with open(dest_file, mode) as f:
                    last_log = time.time()
                    downloaded_chunk = 0
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                            current_size += len(chunk)
                            downloaded_chunk += len(chunk)

                            if time.time() - last_log > 5.0:
                                percent = (current_size / expected_size) * 100
                                mb = round(current_size / (1024**2), 1)
                                total_mb = round(expected_size / (1024**2), 1)
                                print(f"    Progress: {mb}/{total_mb} MB ({percent:.1f}%)")
                                last_log = time.time()

            if dest_file.stat().st_size >= expected_size:
                print(f"    [OK] Completed {dest_file.name}")
                return
        except Exception as e:
            print(f"    [RETRY] Connection dropped ({e}). Resuming in 3 seconds...")
            time.sleep(3)


def main():
    print("====================================================")
    print("RESUMING PYTORCH CUDA 12.4 WHEELS DOWNLOAD")
    print("====================================================")

    for w in WHEELS:
        dest = DEST_DIR / w["filename"]
        download_wheel(w["url"], dest, w["expected_bytes"])

    print("\n====================================================")
    print("ALL PYTORCH CUDA WHEELS READY IN _wheels/")
    print("====================================================")


if __name__ == "__main__":
    main()

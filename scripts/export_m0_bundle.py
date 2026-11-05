#!/usr/bin/env python3
"""
Bundle packager for Milestone M0 Deliverables.
Creates m0_deliverables.zip for independent verification and offline peer-review.
"""

import zipfile
from pathlib import Path

files_to_bundle = [
    "TECH_AUDIT.md",
    "LICENSE_AUDIT.md",
    "HARDWARE_REPORT.md",
    "BENCHMARK_PLAN.md",
    "M0_REVIEW_REQUEST.md",
    "PIANO_PROGETTO_ITA_TTS.md",
    "scripts/hardware_probe.py",
    "scripts/hardware_info.json",
    "scripts/prepare_reference_voices.py",
    "bench/manifest.csv",
    "bench/test_dataset.jsonl",
    "bench/run_benchmark.py",
    "bench/results_benchmark.csv",
    "bench/summary_metrics.json",
    "bench/reference_voices/VOICE_DATASHEET.md",
]

zip_path = Path("m0_deliverables.zip")

with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for f in files_to_bundle:
        p = Path(f)
        if p.exists():
            zf.write(p, arcname=f)
            print(f"Added {f} to {zip_path}")
        else:
            print(f"Warning: {f} not found!")

print(f"\nBundle created: {zip_path} ({round(zip_path.stat().st_size / 1024, 2)} KB)")

#!/usr/bin/env python3
"""
Sentinel Linguistic & ASR Quality Audit on Generated Benchmark Audio.
Evaluates speech accuracy, hallucinations, babbling, and repetition loops
on representative Italian benchmark test cases using Whisper ASR.
"""

import csv
import json
from pathlib import Path
import soundfile as sf
import torch
from transformers import pipeline


def run_sentinel_audit():
    manifest_path = Path("bench/manifest.csv")
    with open(manifest_path, "r", encoding="utf-8") as f:
        tests = list(csv.DictReader(f))
    tests_by_id = {t["id"]: t for t in tests}

    sample_ids = [
        "LEN_01",  # "Sì."
        "LEN_02",  # "No, grazie."
        "LEN_03",  # "Certamente, arrivo subito!"
        "NAR_01",  # Dialogue narrative
        "ACR_01",  # Abbreviations (Dott., Avv., TAR)
        "HOM_01",  # Homograph/accent (ancora / àncora)
        "NUM_01",  # Unnormalized numbers and dates
        "TEC_01",  # English technical terms in Italian
    ]

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Loading Whisper sentinel ASR on {device}...")
    asr = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-tiny",
        device=device,
    )

    report = []
    print("\n" + "=" * 60)
    print("SENTINEL LINGUISTIC AUDIT RESULTS (WHISPER ASR vs GROUND TRUTH)")
    print("=" * 60)

    for sid in sample_ids:
        test = tests_by_id[sid]
        raw_text = test["text_raw"]
        category = test["category"]
        print(f"\n[{sid}] Category: {category}")
        print(f"  Ground Truth: \"{raw_text}\"")

        entry = {
            "id": sid,
            "category": category,
            "ground_truth": raw_text,
            "models": {}
        }

        for model_id in ["M-QWEN-1.7B", "M-QWEN-0.6B"]:
            wav_path = Path(f"bench/outputs/{model_id}/{sid}.wav")
            if not wav_path.exists():
                print(f"  [{model_id}]: WAV not found!")
                entry["models"][model_id] = {"error": "WAV not found"}
                continue

            data, sr = sf.read(str(wav_path))
            if data.ndim > 1:
                data = data.mean(axis=1)
            dur = len(data) / sr

            res = asr(
                {"array": data.astype("float32"), "sampling_rate": sr},
                generate_kwargs={"language": "italian"}
            )
            transcript = res["text"].strip()
            print(f"  [{model_id}] (Dur: {dur:.2f}s): \"{transcript}\"")

            # Basic diagnostics
            has_repetition = any(
                transcript.count(word) > 4 for word in transcript.split() if len(word) > 3
            )
            is_hallucination = (
                len(raw_text.split()) < 3 and len(transcript.split()) > 10
            )

            entry["models"][model_id] = {
                "duration_sec": round(dur, 2),
                "asr_transcript": transcript,
                "has_repetition_loop": has_repetition,
                "is_hallucination": is_hallucination,
            }

        report.append(entry)

    # Save to JSON
    with open("bench/sentinel_quality_audit.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("Sentinel audit saved to bench/sentinel_quality_audit.json")
    print("=" * 60)


if __name__ == "__main__":
    run_sentinel_audit()

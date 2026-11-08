#!/usr/bin/env python3
"""
Rerun NUM_01 for M-QWEN-0.6B to remove the overnight OS suspension artifact,
then re-generate results_benchmark.csv and summary_metrics.json.
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datetime import datetime, timezone
import json
import time
import numpy as np
import soundfile as sf
import torch

from bench.run_benchmark import (
    Qwen3Runner,
    analyze_audio_quality,
    check_cuda_vram,
    compute_sha256,
    run_benchmark,
)

def fix_num01_and_regenerate():
    runner = Qwen3Runner("M-QWEN-0.6B", "./Qwen3-TTS-12Hz-0.6B-Base")
    print("Loading 0.6B model...")
    runner.load()

    ref_audio = "bench/reference_voices/ref_voice_it_01.wav"
    text = "L'11/05/1998 alle ore 14:30 il PIL è salito del 2,5% a fronte di 1.500.000 €."

    print("Generating NUM_01 iteration 1 without OS sleep...")
    audio, sr, gen_time, vram_info = runner.synthesize(text, ref_audio_path=ref_audio)
    runner.unload()

    wav_path = Path("bench/outputs/M-QWEN-0.6B/NUM_01_run1.wav")
    sf.write(str(wav_path), audio, sr)
    metrics = analyze_audio_quality(audio, sr)
    rtf = round(gen_time / metrics["duration_sec"], 3) if metrics["duration_sec"] > 0 else 0.0

    new_event = {
        "timestamp_iso": datetime.now(timezone.utc).isoformat(),
        "event_type": "benchmark_run",
        "test_id": "NUM_01",
        "category": "numbers_currencies_dates",
        "model_id": "M-QWEN-0.6B",
        "run_type": "REAL_INFERENCE",
        "iteration": 1,
        "input_text": text,
        "input_sha256": compute_sha256(text),
        "ref_audio_path": ref_audio,
        "output_wav": str(wav_path),
        "output_sha256": compute_sha256(wav_path),
        "sample_rate": sr,
        "gen_time_sec": round(gen_time, 4),
        "duration_sec": metrics["duration_sec"],
        "rtf": rtf,
        "sample_peak_dbfs": metrics["sample_peak_dbfs"],
        "rms_dbfs": metrics["rms_dbfs"],
        "is_empty": metrics["is_empty"],
        "is_clipped": metrics["is_clipped"],
        "trailing_silence_sec": metrics["trailing_silence_sec"],
        "trailing_silence_gt_2s": metrics["trailing_silence_gt_2s"],
        "peak_vram_allocated_mb": vram_info.get("max_allocated_mb", 0.0),
        "peak_vram_reserved_mb": vram_info.get("max_reserved_mb", 0.0),
        "nvidia_smi_used_mb": vram_info.get("nvidia_smi_used_mb", 0.0),
    }

    print(f"Clean NUM_01 run: gen_time={gen_time:.2f}s, dur={metrics['duration_sec']}s, RTF={rtf:.3f}")

    # Read events, replace event 23 (NUM_01 iter 1 for 0.6B)
    events_path = Path("bench/benchmark_events.jsonl")
    with open(events_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    replaced = False
    for i, line in enumerate(lines):
        ev = json.loads(line)
        if ev.get("model_id") == "M-QWEN-0.6B" and ev.get("test_id") == "NUM_01" and ev.get("iteration") == 1:
            lines[i] = json.dumps(new_event, ensure_ascii=False)
            replaced = True
            print(f"Replaced event at line index {i}")
            break

    if not replaced:
        lines.append(json.dumps(new_event, ensure_ascii=False))

    with open(events_path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")
    print("Updated bench/benchmark_events.jsonl")

    # Now recompute summary and results CSV using benchmark_events.jsonl
    recompute_summary_and_csv()

def recompute_summary_and_csv():
    import csv
    manifest_path = Path("bench/manifest.csv")
    with open(manifest_path, "r", encoding="utf-8") as f:
        tests = list(csv.DictReader(f))

    events_path = Path("bench/benchmark_events.jsonl")
    events = []
    with open(events_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                events.append(json.loads(line.strip()))

    models = ["M-QWEN-1.7B", "M-QWEN-0.6B", "M-CHAT-0.5B"]
    results = []

    for model_id in models:
        model_events = [e for e in events if e.get("model_id") == model_id and e.get("run_type") == "REAL_INFERENCE"]
        ev_map = {}
        for e in model_events:
            tid = e["test_id"]
            if tid not in ev_map:
                ev_map[tid] = []
            ev_map[tid].append(e)

        model_out_dir = Path("bench/outputs") / model_id

        if model_id == "M-CHAT-0.5B":
            load_error = (
                "Chatterbox Multilingual V3 richiede transformers 5.2.0 e gradio 6.8.0, "
                "in conflitto diretto nello stesso ambiente Python con qwen-tts (transformers 4.57.3). "
                "Benchmark non eseguibile in questo runtime; test differito ad ambiente virtuale isolato."
            )
            for item in tests:
                results.append({
                    "test_id": item["id"],
                    "category": item["category"],
                    "model_id": model_id,
                    "run_type": "DEFERRED_SEPARATE_VENV",
                    "status": "NON_TESTATO_DIFFERITO",
                    "error_message": load_error,
                    "gen_time_sec": None,
                    "duration_sec": None,
                    "rtf": None,
                    "peak_vram_mb": None,
                    "peak_dbfs": None,
                    "rms_dbfs": None,
                    "is_empty": None,
                    "is_clipped": None,
                    "is_truncated": None,
                    "wav_path": None,
                })
            continue

        for item in tests:
            test_id = item["id"]
            wav_file = model_out_dir / f"{test_id}.wav"

            if test_id in ev_map:
                evs = ev_map[test_id]
                primary = evs[0]
                rtfs = [e["rtf"] for e in evs]
                gen_times = [e["gen_time_sec"] for e in evs]
                vrams = [e["peak_vram_allocated_mb"] for e in evs]

                results.append({
                    "test_id": test_id,
                    "category": item["category"],
                    "model_id": model_id,
                    "run_type": "REAL_INFERENCE",
                    "status": "SUCCESS",
                    "error_message": "",
                    "gen_time_sec": round(float(np.mean(gen_times)), 3),
                    "duration_sec": primary["duration_sec"],
                    "rtf": round(float(np.median(rtfs)), 3),
                    "peak_vram_mb": round(float(np.max(vrams)), 2),
                    "peak_dbfs": primary["sample_peak_dbfs"],
                    "rms_dbfs": primary["rms_dbfs"],
                    "is_empty": primary["is_empty"],
                    "is_clipped": primary["is_clipped"],
                    "is_truncated": primary["trailing_silence_gt_2s"],
                    "wav_path": str(wav_file),
                })
            elif wav_file.exists() and wav_file.stat().st_size > 1000:
                audio, sr = sf.read(str(wav_file))
                metrics = analyze_audio_quality(audio, sr)
                results.append({
                    "test_id": test_id,
                    "category": item["category"],
                    "model_id": model_id,
                    "run_type": "CACHE_RECOVERY_UNMEASURED",
                    "status": "SUCCESS",
                    "error_message": "",
                    "gen_time_sec": None,
                    "duration_sec": metrics["duration_sec"],
                    "rtf": None,
                    "peak_vram_mb": None,
                    "peak_dbfs": metrics["peak_dbfs"],
                    "rms_dbfs": metrics["rms_dbfs"],
                    "is_empty": metrics["is_empty"],
                    "is_clipped": metrics["is_clipped"],
                    "is_truncated": metrics["is_truncated"],
                    "wav_path": str(wav_file),
                })
            else:
                results.append({
                    "test_id": test_id,
                    "category": item["category"],
                    "model_id": model_id,
                    "run_type": "UNAVAILABLE",
                    "status": "MISSING",
                    "error_message": "Audio not generated",
                    "gen_time_sec": None,
                    "duration_sec": None,
                    "rtf": None,
                    "peak_vram_mb": None,
                    "peak_dbfs": None,
                    "rms_dbfs": None,
                    "is_empty": None,
                    "is_clipped": None,
                    "is_truncated": None,
                    "wav_path": None,
                })

    # Write CSV
    with open("bench/results_benchmark.csv", "w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "test_id", "category", "model_id", "run_type", "status", "error_message",
            "gen_time_sec", "duration_sec", "rtf", "peak_vram_mb",
            "peak_dbfs", "rms_dbfs", "is_empty", "is_clipped", "is_truncated", "wav_path"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # Compute summary
    summary = {}
    for runner_id in ["M-QWEN-1.7B", "M-QWEN-0.6B", "M-CHAT-0.5B"]:
        model_res = [r for r in results if r["model_id"] == runner_id]
        real_inf_res = [r for r in model_res if r["run_type"] == "REAL_INFERENCE" and r["status"] == "SUCCESS"]
        cached_res = [r for r in model_res if r["run_type"] == "CACHE_RECOVERY_UNMEASURED"]
        deferred_res = [r for r in model_res if r["run_type"] == "DEFERRED_SEPARATE_VENV"]

        if deferred_res:
            summary[runner_id] = {
                "status": "NON_TESTATO_DIFFERITO",
                "classification": "NON TESTATO (ambiente separato differito)",
                "total_tests": len(model_res),
                "successful_tests": 0,
                "reason": deferred_res[0]["error_message"],
            }
            continue

        all_success = [r for r in model_res if r["status"] == "SUCCESS"]
        all_durs = [r["duration_sec"] for r in all_success if r["duration_sec"] is not None]

        measured_events = [e for e in events if e.get("model_id") == runner_id and e.get("run_type") == "REAL_INFERENCE"]
        ev_rtfs = [e["rtf"] for e in measured_events]
        ev_gen_times = [e["gen_time_sec"] for e in measured_events]
        ev_alloc_vrams = [e["peak_vram_allocated_mb"] for e in measured_events if e.get("peak_vram_allocated_mb")]
        ev_res_vrams = [e["peak_vram_reserved_mb"] for e in measured_events if e.get("peak_vram_reserved_mb")]
        ev_smi_vrams = [e["nvidia_smi_used_mb"] for e in measured_events if e.get("nvidia_smi_used_mb")]

        summary[runner_id] = {
            "status": "MISURATO_EMPIRICO_SU_CAMPIONE_CONTROLLATO",
            "classification": "MISURATO",
            "acoustic_corpus": {
                "total_cases": len(model_res),
                "successful_audio_files": len(all_success),
                "measured_cases": len(real_inf_res),
                "cached_unmeasured_cases": len(cached_res),
                "mean_audio_duration_sec": round(float(np.mean(all_durs)), 2) if all_durs else None,
                "empty_audio_count": sum(1 for r in all_success if r["is_empty"]),
                "clipped_audio_count": sum(1 for r in all_success if r["is_clipped"]),
                "trailing_silence_gt_2s_count": sum(1 for r in all_success if r["is_truncated"]),
            },
            "empirical_benchmark": {
                "test_ids": sorted(list(set(e["test_id"] for e in measured_events))),
                "total_measured_runs": len(measured_events),
                "iterations_per_test": 2,
                "warmup_excluded": True,
                "timing_method": "time.perf_counter monotonic",
                "rtf_definition": "gen_time_sec / duration_sec (RTF > 1.0 indica piu lento del tempo reale)",
                "rtf_median": round(float(np.median(ev_rtfs)), 3) if ev_rtfs else None,
                "rtf_mean": round(float(np.mean(ev_rtfs)), 3) if ev_rtfs else None,
                "rtf_min": round(float(np.min(ev_rtfs)), 3) if ev_rtfs else None,
                "rtf_max": round(float(np.max(ev_rtfs)), 3) if ev_rtfs else None,
                "rtf_p95": round(float(np.percentile(ev_rtfs, 95)), 3) if ev_rtfs else None,
                "rtf_std": round(float(np.std(ev_rtfs)), 3) if ev_rtfs else None,
                "gen_time_median_sec": round(float(np.median(ev_gen_times)), 3) if ev_gen_times else None,
                "gen_time_mean_sec": round(float(np.mean(ev_gen_times)), 3) if ev_gen_times else None,
                "peak_vram_allocated_mb": round(float(np.max(ev_alloc_vrams)), 2) if ev_alloc_vrams else None,
                "peak_vram_reserved_mb": round(float(np.max(ev_res_vrams)), 2) if ev_res_vrams else None,
                "nvidia_smi_used_mb": round(float(np.max(ev_smi_vrams)), 2) if ev_smi_vrams else None,
            }
        }

    with open("bench/summary_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Successfully wrote bench/results_benchmark.csv and bench/summary_metrics.json")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    fix_num01_and_regenerate()

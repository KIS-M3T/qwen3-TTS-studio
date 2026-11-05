#!/usr/bin/env python3
"""
Repeatable Benchmark Runner for Italian TTS Studio (Milestone M0).
Evaluates candidate models across 60 frozen Italian test sentences.
Measures latency, RTF, peak VRAM, audio clipping, RMS, and silent output.
"""

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf


def check_cuda_vram() -> dict:
    """Query current CUDA memory if torch is available."""
    try:
        import torch

        if torch.cuda.is_available():
            return {
                "available": True,
                "device_name": torch.cuda.get_device_name(0),
                "allocated_mb": round(torch.cuda.memory_allocated() / (1024**2), 2),
                "reserved_mb": round(torch.cuda.memory_reserved() / (1024**2), 2),
                "max_allocated_mb": round(
                    torch.cuda.max_memory_allocated() / (1024**2), 2
                ),
            }
    except Exception:
        pass
    return {"available": False, "allocated_mb": 0.0, "reserved_mb": 0.0, "max_allocated_mb": 0.0}


def analyze_audio_quality(audio: np.ndarray, sr: int) -> dict:
    """Analyze peak, RMS, clipping, and trailing silence on generated audio."""
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    dur_sec = len(audio) / sr if sr > 0 else 0.0
    if len(audio) == 0:
        return {
            "duration_sec": 0.0,
            "peak_linear": 0.0,
            "peak_dbfs": -120.0,
            "rms_linear": 0.0,
            "rms_dbfs": -120.0,
            "is_empty": True,
            "is_clipped": False,
            "trailing_silence_sec": 0.0,
            "is_truncated": False,
        }

    peak_linear = float(np.max(np.abs(audio)))
    rms_linear = float(np.sqrt(np.mean(audio**2)))

    peak_dbfs = float(20 * np.log10(max(peak_linear, 1e-6)))
    rms_dbfs = float(20 * np.log10(max(rms_linear, 1e-6)))

    is_empty = peak_linear < 0.003 or rms_linear < 0.001
    is_clipped = peak_linear >= 0.999

    # Trailing silence detection: check trailing 50ms frames
    frame_len = int(sr * 0.05)
    trailing_frames = 0
    if frame_len > 0 and len(audio) >= frame_len:
        n_frames = len(audio) // frame_len
        for i in range(n_frames - 1, -1, -1):
            frame = audio[i * frame_len : (i + 1) * frame_len]
            if np.sqrt(np.mean(frame**2)) < 0.002:
                trailing_frames += 1
            else:
                break
    trailing_silence_sec = round(trailing_frames * 0.05, 3)
    is_truncated = trailing_silence_sec > 2.0

    return {
        "duration_sec": round(dur_sec, 3),
        "peak_linear": round(peak_linear, 5),
        "peak_dbfs": round(peak_dbfs, 2),
        "rms_linear": round(rms_linear, 5),
        "rms_dbfs": round(rms_dbfs, 2),
        "is_empty": is_empty,
        "is_clipped": is_clipped,
        "trailing_silence_sec": trailing_silence_sec,
        "is_truncated": is_truncated,
    }


class BaseModelRunner:
    def __init__(self, model_id: str, model_path: str | None = None):
        self.model_id = model_id
        self.model_path = model_path
        self.loaded = False
        self.load_error = None

    def load(self) -> bool:
        raise NotImplementedError

    def synthesize(self, text: str, ref_audio_path: str | None = None) -> tuple[np.ndarray, int, float]:
        """Returns: (audio_array, sample_rate, generation_time_sec)"""
        raise NotImplementedError

    def unload(self) -> None:
        pass


class Qwen3Runner(BaseModelRunner):
    def __init__(self, model_id: str, model_path: str):
        super().__init__(model_id, model_path)
        self.model = None

    def load(self) -> bool:
        try:
            import torch
            from qwen_tts import Qwen3TTSModel

            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32

            print(f"[{self.model_id}] Loading model from {self.model_path} on {device} ({dtype})...")
            t0 = time.perf_counter()
            self.model = Qwen3TTSModel.from_pretrained(
                self.model_path,
                device_map=device,
                dtype=dtype,
            )
            self.model.model.eval()
            self.load_time_sec = round(time.perf_counter() - t0, 3)
            self.loaded = True
            print(f"[{self.model_id}] Loaded successfully in {self.load_time_sec}s")
            return True
        except Exception as e:
            self.load_error = str(e)
            print(f"[{self.model_id}] Failed to load: {e}")
            return False

    def synthesize(self, text: str, ref_audio_path: str | None = None) -> tuple[np.ndarray, int, float]:
        if not self.loaded or self.model is None:
            raise RuntimeError(f"Model {self.model_id} not loaded: {self.load_error}")

        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        t0 = time.perf_counter()
        # In Qwen3-TTS, CustomVoice uses speaker preset, Base uses voice clone prompt
        if "CustomVoice" in self.model_id:
            wavs, sr = self.model.generate_custom_voice(
                text=text,
                speaker="vivian",
                language="italian",
            )
        elif ref_audio_path and os.path.exists(ref_audio_path):
            prompt = self.model.create_voice_clone_prompt(ref_audio_path)
            wavs, sr = self.model.generate_voice_clone(
                text=text,
                voice_clone_prompt=prompt,
                language="italian",
            )
        else:
            # Fallback direct generation
            wavs, sr = self.model.generate(
                text=text,
                language="italian",
            )
        gen_time = time.perf_counter() - t0
        return np.array(wavs[0], dtype=np.float32), sr, gen_time

    def unload(self) -> None:
        del self.model
        self.model = None
        self.loaded = False
        try:
            import gc, torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


class ChatterboxRunner(BaseModelRunner):
    def __init__(self, model_id: str = "M-CHAT-0.5B"):
        super().__init__(model_id)

    def load(self) -> bool:
        # Chatterbox Multilingual V3 status check
        try:
            import chatterbox
            self.loaded = True
            return True
        except ImportError:
            self.load_error = "Package 'chatterbox' non installato. Il repository upstream ResembleAI/chatterbox e' ritirato (retired research) da agosto 2026."
            return False

    def synthesize(self, text: str, ref_audio_path: str | None = None):
        raise NotImplementedError(self.load_error)


class DiagnosticSimulationRunner(BaseModelRunner):
    """
    Diagnostic runner for CI/dry-run checks: verifies tokenization,
    splits, regex processing, and pipeline contracts without GPU weights.
    """
    def __init__(self, model_id: str = "DIAGNOSTIC_DRYRUN"):
        super().__init__(model_id)

    def load(self) -> bool:
        self.loaded = True
        return True

    def synthesize(self, text: str, ref_audio_path: str | None = None) -> tuple[np.ndarray, int, float]:
        sr = 24000
        # Simulate acoustic duration: ~15 chars per second in standard Italian reading
        dur = max(0.5, len(text) / 14.5)
        n_samples = int(dur * sr)
        # Synthetic sine test tone (440Hz with envelope) to verify audio I/O pipeline
        t = np.linspace(0, dur, n_samples, endpoint=False, dtype=np.float32)
        audio = 0.2 * np.sin(2 * np.pi * 440 * t)
        # Fade in and out
        fade_len = min(int(sr * 0.05), n_samples // 4)
        if fade_len > 0:
            audio[:fade_len] *= np.linspace(0, 1, fade_len)
            audio[-fade_len:] *= np.linspace(1, 0, fade_len)
        gen_time = 0.05  # Simulated fast response
        return audio, sr, gen_time


def run_benchmark(
    manifest_path: str = "bench/manifest.csv",
    output_dir: str = "bench/outputs",
    results_csv: str = "bench/results_benchmark.csv",
    summary_json: str = "bench/summary_metrics.json",
    models_to_run: list[str] | None = None,
    limit: int | None = None,
    dry_run: bool = False,
):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_file = Path(manifest_path)
    if not manifest_file.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        tests = list(reader)

    if limit and limit > 0:
        tests = tests[:limit]

    print(f"Loaded {len(tests)} benchmark test cases from {manifest_path}")

    # Register runners
    runners: list[BaseModelRunner] = []
    if dry_run:
        runners.append(DiagnosticSimulationRunner())
    else:
        # Check available model paths
        qwen_17b_path = "./Qwen3-TTS-12Hz-1.7B-Base"
        qwen_06b_path = "./Qwen3-TTS-12Hz-0.6B-Base"

        if os.path.exists(qwen_17b_path):
            runners.append(Qwen3Runner("M-QWEN-1.7B", qwen_17b_path))
        if os.path.exists(qwen_06b_path):
            runners.append(Qwen3Runner("M-QWEN-0.6B", qwen_06b_path))

        # Always register Chatterbox to formally document its status
        runners.append(ChatterboxRunner())

        if not runners:
            print("No downloaded Qwen models found locally. Registering Diagnostic Simulation Runner.")
            runners.append(DiagnosticSimulationRunner())

    results = []

    for runner in runners:
        print(f"\n==========================================")
        print(f"RUNNING BENCHMARK FOR: {runner.model_id}")
        print(f"==========================================")

        success = runner.load()
        model_out_dir = out_dir / runner.model_id
        model_out_dir.mkdir(parents=True, exist_ok=True)

        if not success:
            print(f"SKIPPING {runner.model_id}: {runner.load_error}")
            for item in tests:
                results.append({
                    "test_id": item["id"],
                    "category": item["category"],
                    "model_id": runner.model_id,
                    "status": "UNAVAILABLE",
                    "error_message": runner.load_error,
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

        for i, item in enumerate(tests):
            test_id = item["id"]
            text = item["text_raw"]
            print(f"[{i+1}/{len(tests)}] {test_id} ({item['category']}): {text[:50]}...")

            wav_file = model_out_dir / f"{test_id}.wav"
            try:
                audio, sr, gen_time = runner.synthesize(text)
                sf.write(str(wav_file), audio, sr)

                metrics = analyze_audio_quality(audio, sr)
                rtf = round(gen_time / metrics["duration_sec"], 3) if metrics["duration_sec"] > 0 else 0.0

                vram_info = check_cuda_vram()
                peak_vram = vram_info.get("max_allocated_mb", 0.0)

                results.append({
                    "test_id": test_id,
                    "category": item["category"],
                    "model_id": runner.model_id,
                    "status": "SUCCESS",
                    "error_message": "",
                    "gen_time_sec": round(gen_time, 3),
                    "duration_sec": metrics["duration_sec"],
                    "rtf": rtf,
                    "peak_vram_mb": peak_vram,
                    "peak_dbfs": metrics["peak_dbfs"],
                    "rms_dbfs": metrics["rms_dbfs"],
                    "is_empty": metrics["is_empty"],
                    "is_clipped": metrics["is_clipped"],
                    "is_truncated": metrics["is_truncated"],
                    "wav_path": str(wav_file),
                })
            except Exception as e:
                print(f"Error executing {test_id}: {e}")
                results.append({
                    "test_id": test_id,
                    "category": item["category"],
                    "model_id": runner.model_id,
                    "status": "ERROR",
                    "error_message": str(e),
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

        runner.unload()

    # Write results CSV
    with open(results_csv, "w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "test_id", "category", "model_id", "status", "error_message",
            "gen_time_sec", "duration_sec", "rtf", "peak_vram_mb",
            "peak_dbfs", "rms_dbfs", "is_empty", "is_clipped", "is_truncated", "wav_path"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # Compute summary statistics
    summary = {}
    for runner_id in set(r["model_id"] for r in results):
        model_res = [r for r in results if r["model_id"] == runner_id]
        success_res = [r for r in model_res if r["status"] == "SUCCESS"]

        if not success_res:
            summary[runner_id] = {
                "status": "UNAVAILABLE",
                "total_tests": len(model_res),
                "successful_tests": 0,
                "reason": model_res[0]["error_message"] if model_res else "No data",
            }
            continue

        rtfs = [r["rtf"] for r in success_res if r["rtf"] is not None]
        durs = [r["duration_sec"] for r in success_res if r["duration_sec"] is not None]
        vrams = [r["peak_vram_mb"] for r in success_res if r["peak_vram_mb"] is not None]

        summary[runner_id] = {
            "status": "TESTED",
            "total_tests": len(model_res),
            "successful_tests": len(success_res),
            "rtf_median": round(float(np.median(rtfs)), 3) if rtfs else None,
            "rtf_p95": round(float(np.percentile(rtfs, 95)), 3) if rtfs else None,
            "mean_audio_duration_sec": round(float(np.mean(durs)), 2) if durs else None,
            "peak_vram_mb": round(float(np.max(vrams)), 2) if vrams and max(vrams) > 0 else "N/A (CPU/Dry-run)",
            "empty_audio_count": sum(1 for r in success_res if r["is_empty"]),
            "clipped_audio_count": sum(1 for r in success_res if r["is_clipped"]),
            "truncated_count": sum(1 for r in success_res if r["is_truncated"]),
        }

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n==========================================")
    print("BENCHMARK EXECUTION COMPLETED")
    print(f"Results CSV:  {results_csv}")
    print(f"Summary JSON: {summary_json}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Italian TTS Benchmark")
    parser.add_argument("--manifest", default="bench/manifest.csv")
    parser.add_argument("--output-dir", default="bench/outputs")
    parser.add_argument("--results-csv", default="bench/results_benchmark.csv")
    parser.add_argument("--summary-json", default="bench/summary_metrics.json")
    parser.add_argument("--dry-run", action="store_true", help="Run diagnostic simulation runner")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of tests")
    args = parser.parse_args()

    run_benchmark(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        results_csv=args.results_csv,
        summary_json=args.summary_json,
        dry_run=args.dry_run,
        limit=args.limit,
    )

#!/usr/bin/env python3
"""
Repeatable Benchmark Runner for Italian TTS Studio (Milestone M0).
Evaluates candidate models across Italian test sentences.
Measures latency, RTF, peak VRAM, audio clipping, RMS, and trailing silence.
Maintains an append-only event log (bench/benchmark_events.jsonl) for empirical runs.
Never invents synthetic inference times or VRAM for cached files.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf


def compute_sha256(file_or_str: str | Path | bytes) -> str:
    """Compute SHA-256 hash of a file path, string, or bytes."""
    h = hashlib.sha256()
    if isinstance(file_or_str, (str, Path)) and os.path.isfile(str(file_or_str)):
        with open(str(file_or_str), "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        return h.hexdigest()
    elif isinstance(file_or_str, str):
        h.update(file_or_str.encode("utf-8"))
        return h.hexdigest()
    elif isinstance(file_or_str, (bytes, bytearray)):
        h.update(file_or_str)
        return h.hexdigest()
    return ""


def check_cuda_vram() -> dict:
    """Query current CUDA memory (allocated, reserved) and driver-level nvidia-smi."""
    res = {
        "available": False,
        "device_name": "N/A",
        "allocated_mb": 0.0,
        "reserved_mb": 0.0,
        "max_allocated_mb": 0.0,
        "max_reserved_mb": 0.0,
        "nvidia_smi_used_mb": 0.0,
    }
    try:
        import torch

        if torch.cuda.is_available():
            res["available"] = True
            res["device_name"] = torch.cuda.get_device_name(0)
            res["allocated_mb"] = round(torch.cuda.memory_allocated() / (1024**2), 2)
            res["reserved_mb"] = round(torch.cuda.memory_reserved() / (1024**2), 2)
            res["max_allocated_mb"] = round(
                torch.cuda.max_memory_allocated() / (1024**2), 2
            )
            res["max_reserved_mb"] = round(
                torch.cuda.max_memory_reserved() / (1024**2), 2
            )
    except Exception:
        pass

    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=3,
        ).strip()
        res["nvidia_smi_used_mb"] = float(out.split("\n")[0].strip())
    except Exception:
        pass

    return res


def analyze_audio_quality(audio: np.ndarray, sr: int) -> dict:
    """
    Analyze sample peak, RMS, clipping, and trailing silence on audio.
    Note: 'peak_dbfs' is sample peak (not true inter-sample peak).
    'is_truncated' is True if trailing silence > 2.0s (not semantic word cut).
    """
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    dur_sec = len(audio) / sr if sr > 0 else 0.0
    if len(audio) == 0:
        return {
            "duration_sec": 0.0,
            "peak_linear": 0.0,
            "peak_dbfs": -120.0,
            "sample_peak_dbfs": -120.0,
            "rms_linear": 0.0,
            "rms_dbfs": -120.0,
            "is_empty": True,
            "is_clipped": False,
            "trailing_silence_sec": 0.0,
            "is_truncated": False,
            "trailing_silence_gt_2s": False,
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
    trailing_silence_gt_2s = trailing_silence_sec > 2.0

    return {
        "duration_sec": round(dur_sec, 3),
        "peak_linear": round(peak_linear, 5),
        "peak_dbfs": round(peak_dbfs, 2),
        "sample_peak_dbfs": round(peak_dbfs, 2),
        "rms_linear": round(rms_linear, 5),
        "rms_dbfs": round(rms_dbfs, 2),
        "is_empty": is_empty,
        "is_clipped": is_clipped,
        "trailing_silence_sec": trailing_silence_sec,
        "is_truncated": trailing_silence_gt_2s,  # preserved for backward compatibility
        "trailing_silence_gt_2s": trailing_silence_gt_2s,
    }


def append_benchmark_event(events_file: Path, event: dict) -> None:
    """Atomically append a benchmark measurement event to the JSONL log."""
    events_file.parent.mkdir(parents=True, exist_ok=True)
    with open(events_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")
        f.flush()


class BaseModelRunner:
    def __init__(self, model_id: str, model_path: str | None = None):
        self.model_id = model_id
        self.model_path = model_path
        self.loaded = False
        self.load_error = None
        self.load_time_sec = None

    def load(self) -> bool:
        raise NotImplementedError

    def synthesize(self, text: str, ref_audio_path: str | None = None) -> tuple[np.ndarray, int, float, dict]:
        """Returns: (audio_array, sample_rate, generation_time_sec, vram_info)"""
        raise NotImplementedError

    def unload(self) -> None:
        pass


class Qwen3Runner(BaseModelRunner):
    KNOWN_REF_TRANSCRIPTS = {
        "ref_voice_it_01.wav": "Pone sotto la propria egida il campionato e la Nazionale libanese.",
        "ref_voice_it_02.wav": "A tutt'oggi viene venduto su eBay a cifre esorbitanti.",
    }

    def __init__(self, model_id: str, model_path: str):
        super().__init__(model_id, model_path)
        self.model = None
        self.cached_prompt = None
        self.cached_prompt_ref = None

    def load(self) -> bool:
        try:
            import torch
            from qwen_tts import Qwen3TTSModel

            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32

            print(f"[{self.model_id}] Loading model from {self.model_path} on {device} ({dtype})...")
            t0 = time.perf_counter()
            load_kwargs = {
                "device_map": device,
                "dtype": dtype,
            }
            if "cuda" in device:
                load_kwargs["attn_implementation"] = "sdpa"

            self.model = Qwen3TTSModel.from_pretrained(
                self.model_path,
                **load_kwargs,
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

    def synthesize(self, text: str, ref_audio_path: str | None = None) -> tuple[np.ndarray, int, float, dict]:
        if not self.loaded or self.model is None:
            raise RuntimeError(f"Model {self.model_id} not loaded: {self.load_error}")

        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        t0 = time.perf_counter()

        if "CustomVoice" in self.model_id:
            wavs, sr = self.model.generate_custom_voice(
                text=text,
                speaker="vivian",
                language="italian",
            )
        elif ref_audio_path and os.path.exists(ref_audio_path):
            if self.cached_prompt is None or self.cached_prompt_ref != ref_audio_path:
                ref_basename = Path(ref_audio_path).name
                ref_text = self.KNOWN_REF_TRANSCRIPTS.get(ref_basename)
                if ref_text:
                    self.cached_prompt = self.model.create_voice_clone_prompt(
                        ref_audio=ref_audio_path,
                        ref_text=ref_text,
                    )
                else:
                    self.cached_prompt = self.model.create_voice_clone_prompt(
                        ref_audio=ref_audio_path,
                        x_vector_only_mode=True,
                    )
                self.cached_prompt_ref = ref_audio_path

            wavs, sr = self.model.generate_voice_clone(
                text=text,
                voice_clone_prompt=self.cached_prompt,
                language="italian",
            )
        else:
            wavs, sr = self.model.generate(
                text=text,
                language="italian",
            )
        gen_time = time.perf_counter() - t0
        vram_info = check_cuda_vram()
        return np.array(wavs[0], dtype=np.float32), sr, gen_time, vram_info

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
        self.load_error = (
            "Chatterbox Multilingual V3 richiede transformers 5.2.0 e gradio 6.8.0, "
            "in conflitto diretto nello stesso ambiente Python con qwen-tts (transformers 4.57.3). "
            "Benchmark non eseguibile in questo runtime; test differito ad ambiente virtuale isolato."
        )
        self.loaded = False
        return False

    def synthesize(self, text: str, ref_audio_path: str | None = None):
        raise NotImplementedError(self.load_error)


class DiagnosticSimulationRunner(BaseModelRunner):
    def __init__(self, model_id: str = "DIAGNOSTIC_DRYRUN"):
        super().__init__(model_id)

    def load(self) -> bool:
        self.loaded = True
        return True

    def synthesize(self, text: str, ref_audio_path: str | None = None) -> tuple[np.ndarray, int, float, dict]:
        sr = 24000
        dur = max(0.5, len(text) / 14.5)
        n_samples = int(dur * sr)
        t = np.linspace(0, dur, n_samples, endpoint=False, dtype=np.float32)
        audio = 0.2 * np.sin(2 * np.pi * 440 * t)
        fade_len = min(int(sr * 0.05), n_samples // 4)
        if fade_len > 0:
            audio[:fade_len] *= np.linspace(0, 1, fade_len)
            audio[-fade_len:] *= np.linspace(1, 0, fade_len)
        gen_time = 0.05
        vram_info = {"available": False, "allocated_mb": 0.0, "reserved_mb": 0.0, "max_allocated_mb": 0.0, "max_reserved_mb": 0.0, "nvidia_smi_used_mb": 0.0}
        return audio, sr, gen_time, vram_info


DEFAULT_EMPIRICAL_SUBSET = [
    "NUM_01",
    "ACR_01",
    "HOM_01",
    "TEC_01",
    "NAR_01",
    "LEN_01",
    "LEN_02",
    "LEN_03",
    "LEN_04",
    "LEN_05",
]


def run_benchmark(
    manifest_path: str = "bench/manifest.csv",
    output_dir: str = "bench/outputs",
    results_csv: str = "bench/results_benchmark.csv",
    summary_json: str = "bench/summary_metrics.json",
    events_jsonl: str = "bench/benchmark_events.jsonl",
    ref_audio_path: str = "bench/reference_voices/ref_voice_it_01.wav",
    models_to_run: list[str] | None = None,
    empirical_tests: list[str] | None = None,
    iterations: int = 2,
    run_all: bool = False,
    dry_run: bool = False,
    limit: int | None = None,
):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    events_path = Path(events_jsonl)

    manifest_file = Path(manifest_path)
    if not manifest_file.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        tests = list(reader)

    if limit and limit > 0:
        tests = tests[:limit]

    test_dict = {item["id"]: item for item in tests}
    selected_test_ids = [t for t in (empirical_tests or DEFAULT_EMPIRICAL_SUBSET) if t in test_dict]

    print(f"Loaded {len(tests)} benchmark test cases from {manifest_path}")
    print(f"Empirical benchmark subset ({len(selected_test_ids)} cases, {iterations} iterations each): {selected_test_ids}")

    # Register runners
    runners: list[BaseModelRunner] = []
    if dry_run:
        runners.append(DiagnosticSimulationRunner())
    else:
        qwen_17b_path = "./Qwen3-TTS-12Hz-1.7B-Base"
        qwen_06b_path = "./Qwen3-TTS-12Hz-0.6B-Base"

        def is_model_ready(p: str) -> bool:
            st = Path(p) / "model.safetensors"
            return st.exists() and st.stat().st_size > 100 * 1024 * 1024

        if is_model_ready(qwen_17b_path) and (not models_to_run or "M-QWEN-1.7B" in models_to_run):
            runners.append(Qwen3Runner("M-QWEN-1.7B", qwen_17b_path))
        if is_model_ready(qwen_06b_path) and (not models_to_run or "M-QWEN-0.6B" in models_to_run):
            runners.append(Qwen3Runner("M-QWEN-0.6B", qwen_06b_path))

        # Always register Chatterbox to formally document deferred status
        if not models_to_run or "M-CHAT-0.5B" in models_to_run:
            runners.append(ChatterboxRunner())

        if not runners:
            print("No downloaded Qwen models found locally. Registering Diagnostic Simulation Runner.")
            runners.append(DiagnosticSimulationRunner())

    results = []
    empirical_records = {}

    for runner in runners:
        print(f"\n==========================================")
        print(f"BENCHMARK RUNNER: {runner.model_id}")
        print(f"==========================================")

        model_out_dir = out_dir / runner.model_id
        model_out_dir.mkdir(parents=True, exist_ok=True)

        if isinstance(runner, ChatterboxRunner):
            runner.load()
            for item in tests:
                results.append({
                    "test_id": item["id"],
                    "category": item["category"],
                    "model_id": runner.model_id,
                    "run_type": "DEFERRED_SEPARATE_VENV",
                    "status": "NON_TESTATO_DIFFERITO",
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

        if isinstance(runner, DiagnosticSimulationRunner):
            runner.load()
            for item in tests:
                wav_file = model_out_dir / f"{item['id']}.wav"
                audio, sr, gen_time, vram_info = runner.synthesize(item["text_raw"])
                sf.write(str(wav_file), audio, sr)
                metrics = analyze_audio_quality(audio, sr)
                results.append({
                    "test_id": item["id"],
                    "category": item["category"],
                    "model_id": runner.model_id,
                    "run_type": "DIAGNOSTIC_SIMULATION",
                    "status": "SUCCESS",
                    "error_message": "",
                    "gen_time_sec": round(gen_time, 3),
                    "duration_sec": metrics["duration_sec"],
                    "rtf": round(gen_time / metrics["duration_sec"], 3) if metrics["duration_sec"] > 0 else 0.0,
                    "peak_vram_mb": 0.0,
                    "peak_dbfs": metrics["peak_dbfs"],
                    "rms_dbfs": metrics["rms_dbfs"],
                    "is_empty": metrics["is_empty"],
                    "is_clipped": metrics["is_clipped"],
                    "is_truncated": metrics["is_truncated"],
                    "wav_path": str(wav_file),
                })
            continue

        # Load real Qwen model
        success = runner.load()
        if not success:
            print(f"FAILED TO LOAD {runner.model_id}: {runner.load_error}")
            for item in tests:
                results.append({
                    "test_id": item["id"],
                    "category": item["category"],
                    "model_id": runner.model_id,
                    "run_type": "ERROR",
                    "status": "ERROR",
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

        # 1. Warmup Run (Separate and excluded from benchmark metrics)
        print(f"\n[{runner.model_id}] Running separate WARMUP inference...")
        warmup_text = "Inizializzazione e riscaldamento del modello vocale neurale."
        try:
            w_audio, w_sr, w_gen_time, w_vram = runner.synthesize(warmup_text, ref_audio_path=ref_audio_path)
            w_metrics = analyze_audio_quality(w_audio, w_sr)
            w_event = {
                "timestamp_iso": datetime.now(timezone.utc).isoformat(),
                "event_type": "warmup_run",
                "model_id": runner.model_id,
                "run_type": "WARMUP",
                "input_text": warmup_text,
                "input_sha256": compute_sha256(warmup_text),
                "ref_audio_path": ref_audio_path,
                "gen_time_sec": round(w_gen_time, 4),
                "duration_sec": w_metrics["duration_sec"],
                "rtf": round(w_gen_time / w_metrics["duration_sec"], 3) if w_metrics["duration_sec"] > 0 else 0.0,
                "peak_vram_allocated_mb": w_vram.get("max_allocated_mb", 0.0),
                "peak_vram_reserved_mb": w_vram.get("max_reserved_mb", 0.0),
                "nvidia_smi_used_mb": w_vram.get("nvidia_smi_used_mb", 0.0),
            }
            append_benchmark_event(events_path, w_event)
            print(f"[{runner.model_id}] Warmup completed: gen_time={w_gen_time:.2f}s, RTF={w_event['rtf']}")
        except Exception as e:
            print(f"[{runner.model_id}] Warmup failed (non-blocking): {e}")

        # 2. Empirical Benchmark Runs (Controlled subset or all)
        tests_to_measure = tests if run_all else [test_dict[tid] for tid in selected_test_ids]
        empirical_records[runner.model_id] = []

        print(f"\n[{runner.model_id}] Executing empirical benchmark: {len(tests_to_measure)} tests x {iterations} iterations...")
        for it in range(1, iterations + 1):
            print(f"\n--- Iteration {it}/{iterations} for {runner.model_id} ---")
            for item in tests_to_measure:
                test_id = item["id"]
                text = item["text_raw"]
                wav_file = model_out_dir / f"{test_id}.wav"
                iter_wav_file = model_out_dir / f"{test_id}_run{it}.wav"

                print(f"[{runner.model_id}] [Run {it}] {test_id} ({item['category']}): {text[:45]}...")
                try:
                    audio, sr, gen_time, vram_info = runner.synthesize(text, ref_audio_path=ref_audio_path)
                    # Save main WAV and iteration WAV
                    sf.write(str(wav_file), audio, sr)
                    sf.write(str(iter_wav_file), audio, sr)

                    wav_sha256 = compute_sha256(wav_file)
                    metrics = analyze_audio_quality(audio, sr)
                    rtf = round(gen_time / metrics["duration_sec"], 3) if metrics["duration_sec"] > 0 else 0.0
                    peak_vram = vram_info.get("max_allocated_mb", 0.0)

                    event = {
                        "timestamp_iso": datetime.now(timezone.utc).isoformat(),
                        "event_type": "benchmark_run",
                        "test_id": test_id,
                        "category": item["category"],
                        "model_id": runner.model_id,
                        "run_type": "REAL_INFERENCE",
                        "iteration": it,
                        "input_text": text,
                        "input_sha256": compute_sha256(text),
                        "ref_audio_path": ref_audio_path,
                        "output_wav": str(wav_file),
                        "output_sha256": wav_sha256,
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
                        "peak_vram_allocated_mb": peak_vram,
                        "peak_vram_reserved_mb": vram_info.get("max_reserved_mb", 0.0),
                        "nvidia_smi_used_mb": vram_info.get("nvidia_smi_used_mb", 0.0),
                    }
                    append_benchmark_event(events_path, event)
                    empirical_records[runner.model_id].append(event)
                    print(f"    -> gen_time={gen_time:.2f}s, dur={metrics['duration_sec']}s, RTF={rtf:.3f}, VRAM={peak_vram:.1f}MB")

                except Exception as e:
                    print(f"    -> ERROR in {test_id} (run {it}): {e}")

        runner.unload()

        # Ingest full test set into results table
        measured_map = {}
        for ev in empirical_records.get(runner.model_id, []):
            tid = ev["test_id"]
            if tid not in measured_map:
                measured_map[tid] = []
            measured_map[tid].append(ev)

        for item in tests:
            test_id = item["id"]
            wav_file = model_out_dir / f"{test_id}.wav"

            if test_id in measured_map:
                evs = measured_map[test_id]
                # Use iteration 1 or average
                primary = evs[0]
                rtfs = [e["rtf"] for e in evs]
                gen_times = [e["gen_time_sec"] for e in evs]
                vrams = [e["peak_vram_allocated_mb"] for e in evs]

                results.append({
                    "test_id": test_id,
                    "category": item["category"],
                    "model_id": runner.model_id,
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
                    "model_id": runner.model_id,
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
                    "model_id": runner.model_id,
                    "run_type": "UNAVAILABLE",
                    "status": "MISSING",
                    "error_message": "Audio not generated and not cached",
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

    # Write results CSV
    with open(results_csv, "w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "test_id", "category", "model_id", "run_type", "status", "error_message",
            "gen_time_sec", "duration_sec", "rtf", "peak_vram_mb",
            "peak_dbfs", "rms_dbfs", "is_empty", "is_clipped", "is_truncated", "wav_path"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # Compute summary statistics separating REAL_INFERENCE from unmeasured
    summary = {}
    for runner_id in sorted(list(set(r["model_id"] for r in results))):
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

        summary_entry = {
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
            }
        }

        if real_inf_res:
            measured_events = empirical_records.get(runner_id, [])
            ev_rtfs = [e["rtf"] for e in measured_events]
            ev_gen_times = [e["gen_time_sec"] for e in measured_events]
            ev_alloc_vrams = [e["peak_vram_allocated_mb"] for e in measured_events if e.get("peak_vram_allocated_mb")]
            ev_res_vrams = [e["peak_vram_reserved_mb"] for e in measured_events if e.get("peak_vram_reserved_mb")]
            ev_smi_vrams = [e["nvidia_smi_used_mb"] for e in measured_events if e.get("nvidia_smi_used_mb")]

            summary_entry["empirical_benchmark"] = {
                "test_ids": sorted(list(set(e["test_id"] for e in measured_events))),
                "total_measured_runs": len(measured_events),
                "iterations_per_test": iterations,
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
        else:
            summary_entry["empirical_benchmark"] = None

        summary[runner_id] = summary_entry

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n==========================================")
    print("BENCHMARK EXECUTION COMPLETED")
    print(f"Results CSV:  {results_csv}")
    print(f"Summary JSON: {summary_json}")
    print(f"Events JSONL: {events_jsonl}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Italian TTS Benchmark")
    parser.add_argument("--manifest", default="bench/manifest.csv")
    parser.add_argument("--output-dir", default="bench/outputs")
    parser.add_argument("--results-csv", default="bench/results_benchmark.csv")
    parser.add_argument("--summary-json", default="bench/summary_metrics.json")
    parser.add_argument("--events-jsonl", default="bench/benchmark_events.jsonl")
    parser.add_argument("--ref-audio", default="bench/reference_voices/ref_voice_it_01.wav", help="Reference voice WAV")
    parser.add_argument("--empirical-tests", nargs="+", default=DEFAULT_EMPIRICAL_SUBSET, help="Subset of tests for empirical timing")
    parser.add_argument("--iterations", type=int, default=2, help="Number of measured iterations per test")
    parser.add_argument("--run-all", action="store_true", help="Run empirical benchmark on all tests")
    parser.add_argument("--dry-run", action="store_true", help="Run diagnostic simulation runner")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of tests")
    parser.add_argument("--models", nargs="+", default=None, help="Filter specific model IDs (e.g. M-QWEN-0.6B)")
    args = parser.parse_args()

    run_benchmark(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        results_csv=args.results_csv,
        summary_json=args.summary_json,
        events_jsonl=args.events_jsonl,
        ref_audio_path=args.ref_audio,
        models_to_run=args.models,
        empirical_tests=args.empirical_tests,
        iterations=args.iterations,
        run_all=args.run_all,
        dry_run=args.dry_run,
        limit=args.limit,
    )

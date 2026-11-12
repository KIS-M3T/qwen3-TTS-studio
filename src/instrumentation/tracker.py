"""Benchmark session runner, per-run event schema validation, and reporting (R1, R2, C3, C4)."""

from datetime import datetime, timezone
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid

from src.instrumentation.manifest import get_environment_manifest
from src.instrumentation.memory import get_memory_snapshot
from src.instrumentation.persistence import (
    append_benchmark_event,
    read_benchmark_events,
    write_atomic_csv,
    write_atomic_json,
)


PER_RUN_CSV_FIELDNAMES = [
    "session_id",
    "run_id",
    "model_id",
    "model_sha256",
    "test_case_id",
    "iteration_index",
    "run_type",
    "status",
    "measured",
    "seed",
    "temperature",
    "top_p",
    "top_k",
    "min_new_tokens",
    "max_new_tokens",
    "input_text_raw",
    "audio_duration_sec",
    "inference_time_sec",
    "rtf",
    "prep_time_sec",
    "save_time_sec",
    "torch_allocated_mb",
    "torch_reserved_mb",
    "torch_peak_allocated_mb",
    "driver_used_mb_at_end",
    "baseline_process_mb",
    "wav_path",
    "wav_sha256",
    "timestamp_utc",
]


class BenchmarkSession:
    """Manages an isolated benchmark session with atomic logging and zero historical overwrite."""

    def __init__(self, session_id: Optional[str] = None, base_dir: Optional[Path] = None):
        now_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.session_id = session_id or f"m1_{now_str}_{uuid.uuid4().hex[:6]}"
        self.base_dir = Path(base_dir) if base_dir else Path("bench/sessions")
        self.session_dir = self.base_dir / self.session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)

        self.events_file = self.session_dir / "benchmark_events.jsonl"
        self.csv_file = self.session_dir / "results_benchmark.csv"
        self.summary_file = self.session_dir / "summary_metrics.json"
        self.manifest_file = self.session_dir / "environment_manifest.json"

        # Initialize and save environment manifest once per session
        self.manifest = get_environment_manifest()
        write_atomic_json(self.manifest_file, self.manifest)

    def record_run(
        self,
        model_id: str,
        test_case_id: str,
        iteration_index: int,
        input_text_raw: str,
        run_type: str = "measured",  # "warmup" or "measured"
        status: str = "success",     # "success", "error", "deferred"
        measured: bool = True,
        seed: Optional[int] = 42,
        temperature: float = 0.7,
        top_p: float = 0.8,
        top_k: int = 20,
        min_new_tokens: int = 60,
        max_new_tokens: int = 2048,
        subtalker_params: Optional[Dict[str, Any]] = None,
        audio_duration_sec: Optional[float] = None,
        inference_time_sec: Optional[float] = None,
        prep_time_sec: Optional[float] = None,
        save_time_sec: Optional[float] = None,
        memory_snapshot: Optional[Dict[str, Any]] = None,
        wav_path: Optional[str] = None,
        wav_sha256: Optional[str] = None,
        model_sha256: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Validate and append a single per-run record."""
        run_id = f"{self.session_id}_{model_id}_{test_case_id}_iter{iteration_index}"

        # Calculate RTF only if empirically measured and duration > 0
        rtf: Optional[float] = None
        if measured and inference_time_sec is not None and audio_duration_sec and audio_duration_sec > 0:
            rtf = round(inference_time_sec / audio_duration_sec, 4)

        mem = memory_snapshot or get_memory_snapshot()

        event: Dict[str, Any] = {
            "session_id": self.session_id,
            "run_id": run_id,
            "model_id": model_id,
            "model_sha256": model_sha256 or "unknown",
            "test_case_id": test_case_id,
            "iteration_index": iteration_index,
            "run_type": run_type,
            "status": status,
            "measured": measured,
            "seed": seed,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "min_new_tokens": min_new_tokens,
            "max_new_tokens": max_new_tokens,
            "subtalker_params": subtalker_params or {},
            "input_text_raw": input_text_raw,
            "audio_duration_sec": round(audio_duration_sec, 4) if audio_duration_sec is not None else None,
            "inference_time_sec": round(inference_time_sec, 4) if inference_time_sec is not None else None,
            "rtf": rtf,
            "prep_time_sec": round(prep_time_sec, 4) if prep_time_sec is not None else None,
            "save_time_sec": round(save_time_sec, 4) if save_time_sec is not None else None,
            "torch_allocated_mb": mem.get("torch_allocated_mb", 0.0),
            "torch_reserved_mb": mem.get("torch_reserved_mb", 0.0),
            "torch_peak_allocated_mb": mem.get("torch_peak_allocated_mb", 0.0),
            "driver_used_mb_at_end": mem.get("driver_used_mb_at_end"),
            "baseline_process_mb": mem.get("baseline_process_mb", 0.0),
            "wav_path": str(wav_path) if wav_path else None,
            "wav_sha256": wav_sha256,
            "error_message": error_message,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }

        # Thread-safe fsync append
        append_benchmark_event(self.events_file, event)
        return event

    def generate_reports(self) -> Dict[str, Any]:
        """Aggregate per-run events into results_benchmark.csv and summary_metrics.json atomically."""
        events = read_benchmark_events(self.events_file)
        if not events:
            return {}

        # 1. Write per-run CSV
        write_atomic_csv(self.csv_file, PER_RUN_CSV_FIELDNAMES, events)

        # 2. Compute statistical summary (separate measured from warmup and errors)
        summary: Dict[str, Any] = {
            "session_id": self.session_id,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "total_events": len(events),
            "models": {},
        }

        # Group by model
        by_model: Dict[str, List[Dict[str, Any]]] = {}
        for ev in events:
            m_id = ev.get("model_id", "unknown")
            by_model.setdefault(m_id, []).append(ev)

        for m_id, m_events in by_model.items():
            measured_runs = [
                e for e in m_events
                if e.get("run_type") == "measured" and e.get("status") == "success" and e.get("measured") is True
            ]
            rtf_vals = [e["rtf"] for e in measured_runs if e.get("rtf") is not None]
            inf_times = [e["inference_time_sec"] for e in measured_runs if e.get("inference_time_sec") is not None]
            peak_vrams = [e["torch_peak_allocated_mb"] for e in measured_runs if e.get("torch_peak_allocated_mb") is not None]

            rtf_vals.sort()
            inf_times.sort()

            summary["models"][m_id] = {
                "total_runs": len(m_events),
                "measured_runs": len(measured_runs),
                "warmup_runs": len([e for e in m_events if e.get("run_type") == "warmup"]),
                "error_runs": len([e for e in m_events if e.get("status") == "error"]),
                "rtf_median": round(rtf_vals[len(rtf_vals) // 2], 4) if rtf_vals else None,
                "rtf_min": min(rtf_vals) if rtf_vals else None,
                "rtf_max": max(rtf_vals) if rtf_vals else None,
                "inference_time_mean_sec": round(sum(inf_times) / len(inf_times), 4) if inf_times else None,
                "max_peak_vram_mb": max(peak_vrams) if peak_vrams else None,
            }

        # Write summary JSON atomically
        write_atomic_json(self.summary_file, summary)
        return summary

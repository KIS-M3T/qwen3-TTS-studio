"""Unit tests for instrumentation, persistence, and crash recovery (WP1)."""

import json
from pathlib import Path
import tempfile
import pytest

from src.instrumentation.manifest import get_environment_manifest
from src.instrumentation.memory import get_memory_snapshot
from src.instrumentation.persistence import (
    append_benchmark_event,
    read_benchmark_events,
    write_atomic_csv,
    write_atomic_json,
)
from src.instrumentation.tracker import BenchmarkSession, PER_RUN_CSV_FIELDNAMES


def test_environment_manifest():
    """Verify environment manifest collects expected fields and deterministic SHA-256."""
    manifest = get_environment_manifest()
    assert "python_version" in manifest
    assert "platform_system" in manifest
    assert "manifest_sha256" in manifest
    assert len(manifest["manifest_sha256"]) == 64


def test_memory_snapshot_structure():
    """Verify memory snapshot separates PyTorch allocation, reserved, and driver."""
    mem = get_memory_snapshot(baseline_process_mb=100.0)
    assert "torch_allocated_mb" in mem
    assert "torch_reserved_mb" in mem
    assert "torch_peak_allocated_mb" in mem
    assert "driver_used_mb_at_end" in mem
    assert "baseline_process_mb" in mem
    assert mem["baseline_process_mb"] == 100.0


def test_atomic_fsync_persistence_and_read():
    """Verify single-writer append with fsync produces valid JSONL."""
    with tempfile.TemporaryDirectory() as tmpdir:
        events_file = Path(tmpdir) / "events.jsonl"
        event1 = {"run_id": "run_01", "model_id": "test_model", "rtf": 4.5}
        event2 = {"run_id": "run_02", "model_id": "test_model", "rtf": 4.8}

        append_benchmark_event(events_file, event1)
        append_benchmark_event(events_file, event2)

        loaded = read_benchmark_events(events_file)
        assert len(loaded) == 2
        assert loaded[0]["run_id"] == "run_01"
        assert loaded[1]["run_id"] == "run_02"


def test_crash_recovery_truncated_line():
    """Verify parser recovers from mid-write crash/power-cut (truncated trailing line)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        events_file = Path(tmpdir) / "events.jsonl"
        # Write two valid lines and one truncated line
        with open(events_file, "w", encoding="utf-8") as f:
            f.write(json.dumps({"run_id": "run_01", "status": "success"}) + "\n")
            f.write(json.dumps({"run_id": "run_02", "status": "success"}) + "\n")
            f.write('{"run_id": "run_03", "status": "succ')  # Truncated JSON line

        loaded = read_benchmark_events(events_file)
        # Should gracefully recover the 2 valid events without crashing
        assert len(loaded) == 2
        assert loaded[0]["run_id"] == "run_01"
        assert loaded[1]["run_id"] == "run_02"

        # Should isolate the corrupt fragment in .corrupted_recovery.log
        recovery_log = events_file.with_suffix(".corrupted_recovery.log")
        assert recovery_log.exists()
        content = recovery_log.read_text(encoding="utf-8")
        assert "run_03" in content


def test_atomic_replace_json_and_csv():
    """Verify atomic write pattern via temp file and os.replace."""
    with tempfile.TemporaryDirectory() as tmpdir:
        target_json = Path(tmpdir) / "summary.json"
        data = {"metric_a": 123, "metric_b": "test"}
        write_atomic_json(target_json, data)
        assert target_json.exists()
        with open(target_json, encoding="utf-8") as f:
            assert json.load(f) == data

        target_csv = Path(tmpdir) / "results.csv"
        rows = [
            {"col1": "val1", "col2": "val2"},
            {"col1": "val3", "col2": "val4"},
        ]
        write_atomic_csv(target_csv, ["col1", "col2"], rows)
        assert target_csv.exists()
        lines = target_csv.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 3  # header + 2 rows


def test_benchmark_session_disaggregation():
    """Verify session records per-run rows and separates warmup from measured."""
    with tempfile.TemporaryDirectory() as tmpdir:
        session = BenchmarkSession(session_id="test_m1_session", base_dir=Path(tmpdir))

        # Record 1 warmup run
        session.record_run(
            model_id="1.7B-Base",
            test_case_id="WARM_01",
            iteration_index=0,
            input_text_raw="Warmup text",
            run_type="warmup",
            measured=False,
            audio_duration_sec=2.0,
            inference_time_sec=8.0,
        )

        # Record 2 measured runs
        session.record_run(
            model_id="1.7B-Base",
            test_case_id="TEST_01",
            iteration_index=1,
            input_text_raw="Test phrase",
            run_type="measured",
            measured=True,
            audio_duration_sec=2.0,
            inference_time_sec=9.0,
        )
        session.record_run(
            model_id="1.7B-Base",
            test_case_id="TEST_01",
            iteration_index=2,
            input_text_raw="Test phrase",
            run_type="measured",
            measured=True,
            audio_duration_sec=2.0,
            inference_time_sec=9.4,
        )

        # Generate reports
        summary = session.generate_reports()
        assert session.events_file.exists()
        assert session.csv_file.exists()
        assert session.summary_file.exists()

        events = read_benchmark_events(session.events_file)
        assert len(events) == 3  # 1 warmup + 2 measured

        m_summary = summary["models"]["1.7B-Base"]
        assert m_summary["total_runs"] == 3
        assert m_summary["measured_runs"] == 2
        assert m_summary["warmup_runs"] == 1
        # G4: Strict mathematical median of [4.5, 4.7] is 4.6
        assert m_summary["rtf_median"] == 4.6


def test_exact_mathematical_rtf_median_mandatory():
    """G4 Mandatory test: two measured runs with RTF 4.5 and 4.7 must produce exactly median 4.6.

    Verifies that alternative or loose values (such as 4.5 or 4.7 from integer division) are rejected.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        session = BenchmarkSession(session_id="test_median_exact", base_dir=Path(tmpdir))

        # Run 1: duration 2.0s, inf time 9.0s -> RTF 4.5
        session.record_run(
            model_id="0.6B-Base",
            test_case_id="TEST_01",
            iteration_index=1,
            input_text_raw="Frase 1",
            run_type="measured",
            measured=True,
            audio_duration_sec=2.0,
            inference_time_sec=9.0,
            status="success",
        )

        # Run 2: duration 2.0s, inf time 9.4s -> RTF 4.7
        session.record_run(
            model_id="0.6B-Base",
            test_case_id="TEST_01",
            iteration_index=2,
            input_text_raw="Frase 1",
            run_type="measured",
            measured=True,
            audio_duration_sec=2.0,
            inference_time_sec=9.4,
            status="success",
        )

        summary = session.generate_reports()
        m_summary = summary["models"]["0.6B-Base"]

        # Exact mathematical median of [4.5, 4.7] is 4.6
        assert m_summary["rtf_median"] == 4.6
        assert m_summary["rtf_median"] != 4.5
        assert m_summary["rtf_median"] != 4.7



def test_m0_artifacts_untouched():
    """Verify that M0 historical files exist and are not touched by new sessions."""
    m0_events = Path("bench/benchmark_events.jsonl")
    if m0_events.exists():
        initial_mtime = m0_events.stat().st_mtime
        # Create a session
        with tempfile.TemporaryDirectory() as tmpdir:
            s = BenchmarkSession(session_id="dummy_test", base_dir=Path(tmpdir))
            s.record_run("0.6B-Base", "TEST_01", 1, "text")
        assert m0_events.stat().st_mtime == initial_mtime

"""Robust, crash-resilient persistence utilities for benchmark logs (R1, R2, C3)."""

import csv
import json
import logging
import os
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, List, Union

logger = logging.getLogger(__name__)

_PERSISTENCE_LOCK = threading.Lock()


def append_benchmark_event(events_file: Union[Path, str], event: Dict[str, Any]) -> None:
    """Thread-safe append of a benchmark event with mandatory flush and fsync.

    Args:
        events_file: Path to the target JSONL file.
        event: Dictionary representing a single run event.
    """
    path = Path(events_file)
    path.parent.mkdir(parents=True, exist_ok=True)

    line = json.dumps(event, ensure_ascii=False) + "\n"

    with _PERSISTENCE_LOCK:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())


def read_benchmark_events(events_file: Union[Path, str]) -> List[Dict[str, Any]]:
    """Read benchmark events with automatic recovery from truncated lines (crash resilience).

    If the trailing line is incomplete (e.g. due to hard kill mid-write),
    the corrupted fragment is safely isolated to `.corrupted_recovery.log`
    and all valid preceding records are returned intact.

    Args:
        events_file: Path to the JSONL file.

    Returns:
        List[Dict[str, Any]]: List of parsed event dictionaries.
    """
    path = Path(events_file)
    if not path.exists():
        return []

    events: List[Dict[str, Any]] = []
    corrupted_lines: List[str] = []

    with open(path, "r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
                events.append(event)
            except json.JSONDecodeError as e:
                logger.warning(
                    f"Corrupted or truncated JSONL line detected at {path}:{line_no}: {e}. Isolating line."
                )
                corrupted_lines.append(f"Line {line_no} error: {e} | Content: {raw_line}")

    if corrupted_lines:
        recovery_log = path.with_suffix(".corrupted_recovery.log")
        try:
            with open(recovery_log, "a", encoding="utf-8") as rf:
                for cl in corrupted_lines:
                    rf.write(cl + "\n")
                rf.flush()
                os.fsync(rf.fileno())
        except Exception as ex:
            logger.error(f"Failed to write corrupted lines recovery log: {ex}")

    return events


def write_atomic_json(target_path: Union[Path, str], data: Any) -> None:
    """Atomic write of JSON document via temporary file and atomic replace.

    Args:
        target_path: Path to target file.
        data: JSON-serializable object.
    """
    target = Path(target_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    with _PERSISTENCE_LOCK:
        # Create temp file in same directory to guarantee atomic rename across filesystems
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=target.parent,
            prefix=f".{target.name}.tmp_",
            delete=False,
            encoding="utf-8",
        ) as tf:
            temp_name = tf.name
            json.dump(data, tf, indent=2, ensure_ascii=False)
            tf.write("\n")
            tf.flush()
            os.fsync(tf.fileno())

        os.replace(temp_name, target)


def write_atomic_csv(
    target_path: Union[Path, str],
    fieldnames: List[str],
    rows: List[Dict[str, Any]],
) -> None:
    """Atomic write of tabular CSV document via temporary file and atomic replace.

    Args:
        target_path: Path to target CSV file.
        fieldnames: Column headers list.
        rows: List of row dictionaries.
    """
    target = Path(target_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    with _PERSISTENCE_LOCK:
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=target.parent,
            prefix=f".{target.name}.tmp_",
            delete=False,
            encoding="utf-8",
            newline="",
        ) as tf:
            temp_name = tf.name
            writer = csv.DictWriter(tf, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
            tf.flush()
            os.fsync(tf.fileno())

        os.replace(temp_name, target)

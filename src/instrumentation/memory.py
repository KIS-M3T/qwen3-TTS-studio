"""GPU and process memory telemetry for accurate benchmark tracking (R3, C6, F3)."""

import subprocess
from typing import Any, Dict, Optional


def get_driver_vram_mb() -> Optional[float]:
    """Query NVIDIA driver for overall GPU memory used (via nvidia-smi).

    Returns:
        Optional[float]: Total used VRAM in MiB reported by driver, or None.
    """
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ]
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5,
            check=True,
        )
        lines = result.stdout.strip().splitlines()
        if lines:
            return float(lines[0].strip())
    except Exception:
        pass
    return None


def get_memory_snapshot(baseline_process_mb: float = 0.0) -> Dict[str, Any]:
    """Capture multi-tier memory metrics.

    Distinguishes:
      - Active PyTorch tensor allocations (memory_allocated)
      - PyTorch caching allocator reserved pool (memory_reserved)
      - Peak PyTorch allocation (max_memory_allocated)
      - Driver-level total GPU usage snapshot at end (nvidia-smi / NVML)

    Args:
        baseline_process_mb: Process memory baseline measured after CUDA initialization.

    Returns:
        Dict[str, Any]: Memory metrics dictionary in MiB.
    """
    snapshot: Dict[str, Any] = {
        "cuda_available": False,
        "torch_allocated_mb": 0.0,
        "torch_reserved_mb": 0.0,
        "torch_peak_allocated_mb": 0.0,
        "driver_used_mb_at_end": None,
        "baseline_process_mb": round(baseline_process_mb, 2),
        "delta_alloc_mb": 0.0,
    }

    try:
        import torch

        if torch.cuda.is_available():
            snapshot["cuda_available"] = True
            alloc_mb = torch.cuda.memory_allocated() / (1024.0**2)
            res_mb = torch.cuda.memory_reserved() / (1024.0**2)
            peak_mb = torch.cuda.max_memory_allocated() / (1024.0**2)

            snapshot["torch_allocated_mb"] = round(alloc_mb, 2)
            snapshot["torch_reserved_mb"] = round(res_mb, 2)
            snapshot["torch_peak_allocated_mb"] = round(peak_mb, 2)
            snapshot["delta_alloc_mb"] = round(max(0.0, alloc_mb - baseline_process_mb), 2)
            snapshot["driver_used_mb_at_end"] = get_driver_vram_mb()
    except Exception:
        pass

    return snapshot

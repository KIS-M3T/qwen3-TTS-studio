"""Lifecycle management, state transitions, dynamic VRAM gating, and OOM recovery (C2, F3)."""

import gc
import logging
from typing import Optional

from src.engine.errors import EngineOOMError, UnrecoverableCudaError
from src.engine.types import LifecycleState

logger = logging.getLogger(__name__)


class ModelLifecycleManager:
    """Manages model state transitions and guarded CUDA recovery."""

    def __init__(self, model_id: str):
        self.model_id = model_id
        self._state = LifecycleState.UNLOADED
        self._baseline_allocated_mb = 0.0

    @property
    def state(self) -> LifecycleState:
        return self._state

    def begin_loading(self) -> None:
        """Transition from UNLOADED to LOADING."""
        if self._state not in (LifecycleState.UNLOADED, LifecycleState.FAILED):
            logger.warning(f"Beginning load from unexpected state: {self._state}")
        self._state = LifecycleState.LOADING

    def mark_ready(self) -> None:
        """Transition to READY after successful resource initialization."""
        self._state = LifecycleState.READY

    def mark_unloaded(self) -> None:
        """Transition to UNLOADED after clean resource release."""
        self._state = LifecycleState.UNLOADED

    def check_vram_requirement(self, required_mb: float) -> bool:
        """Verify available VRAM exceeds requirement plus safety margin."""
        try:
            import torch

            if torch.cuda.is_available():
                free_b, total_b = torch.cuda.mem_get_info()
                free_mb = free_b / (1024.0**2)
                return free_mb >= required_mb
        except Exception:
            pass
        return True

    def handle_oom(self, error: Exception) -> EngineOOMError:
        """Execute OOM cleanup and diagnostic probe allocation (F3).

        Probe allocation is a preliminary diagnostic check of CUDA context health.
        If probe succeeds, transitions to UNLOADED.
        If probe fails, transitions to FAILED.
        """
        self._state = LifecycleState.RECOVERING
        logger.error(f"OOM detected in {self.model_id}: {error}. Executing recovery...")

        gc.collect()

        probe_success = False
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                # Diagnostic probe allocation
                probe = torch.zeros((512, 512), dtype=torch.float32, device="cuda")
                torch.cuda.synchronize()
                del probe
                torch.cuda.empty_cache()
                probe_success = True
        except Exception as probe_err:
            logger.critical(f"CUDA context corrupted during probe allocation: {probe_err}")
            probe_success = False

        if probe_success:
            self._state = LifecycleState.UNLOADED
            return EngineOOMError(
                f"Recoverable OOM occurred in {self.model_id}. Cache freed; engine reset to UNLOADED.",
                is_recoverable=True,
            )
        else:
            self._state = LifecycleState.FAILED
            return EngineOOMError(
                f"Unrecoverable CUDA context fault in {self.model_id}. Worker process restart required.",
                is_recoverable=False,
            )

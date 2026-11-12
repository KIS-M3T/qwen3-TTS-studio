"""Lifecycle management, state transitions, dynamic VRAM gating, residency coordination, and OOM recovery (C2, G3)."""

import gc
import logging
import threading
from typing import Callable, Optional

from src.engine.errors import EngineOOMError
from src.engine.types import LifecycleState

logger = logging.getLogger(__name__)


class ModelLifecycleManager:
    """Manages model state transitions, single-resident concurrency, and guarded CUDA recovery."""

    # Class-level residency lock and tracker across adapter instances (G3)
    _residency_lock = threading.Lock()
    _active_resident_id: Optional[str] = None

    def __init__(self, model_id: str):
        self.model_id = model_id
        self._state = LifecycleState.UNLOADED
        self._baseline_allocated_mb: Optional[float] = None

    @property
    def state(self) -> LifecycleState:
        return self._state

    @classmethod
    def get_active_resident_id(cls) -> Optional[str]:
        """Return the model_id of the currently resident model, or None."""
        with cls._residency_lock:
            return cls._active_resident_id

    @classmethod
    def acquire_residency(cls, model_id: str) -> None:
        """Enforce single resident model policy across adapters (G3)."""
        with cls._residency_lock:
            if cls._active_resident_id is not None and cls._active_resident_id != model_id:
                raise RuntimeError(
                    f"Residency conflict: model '{cls._active_resident_id}' is currently resident in GPU memory. "
                    f"It must be unloaded before loading '{model_id}'."
                )
            cls._active_resident_id = model_id

    @classmethod
    def release_residency(cls, model_id: str) -> None:
        """Release residency if held by model_id."""
        with cls._residency_lock:
            if cls._active_resident_id == model_id:
                cls._active_resident_id = None

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
        self.release_residency(self.model_id)

    def mark_failed(self) -> None:
        """Transition to FAILED upon unrecoverable error."""
        self._state = LifecycleState.FAILED
        self.release_residency(self.model_id)

    def check_vram_requirement(self, required_mb: float) -> bool:
        """Verify available VRAM exceeds requirement before loading (G3)."""
        try:
            import torch

            if torch.cuda.is_available():
                free_b, total_b = torch.cuda.mem_get_info()
                free_mb = free_b / (1024.0**2)
                return free_mb >= required_mb
        except Exception as e:
            logger.warning(f"Unable to query VRAM via mem_get_info: {e}")
        return True

    def handle_oom(
        self,
        error: Exception,
        release_callback: Optional[Callable[[], None]] = None,
    ) -> EngineOOMError:
        """Execute OOM cleanup with guaranteed reference severance and diagnostic probe (G3).

        Note: Probe allocation is strictly a preliminary diagnostic check of CUDA context
        integrity; it is NEVER treated as sufficient proof of full operational recovery.
        """
        self._state = LifecycleState.RECOVERING
        logger.error(f"OOM detected in {self.model_id}: {error}. Executing recovery...")

        # 1. Execute release callback to sever references before garbage collection
        if release_callback is not None:
            try:
                release_callback()
            except Exception as cb_err:
                logger.warning(f"Error during OOM release callback: {cb_err}")

        # 2. Release residency
        self.release_residency(self.model_id)

        # 3. Garbage collection
        gc.collect()

        probe_success = False
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                # Diagnostic probe allocation (512x512 float32 = 1MB)
                probe = torch.zeros((512, 512), dtype=torch.float32, device="cuda")
                torch.cuda.synchronize()
                del probe
                torch.cuda.empty_cache()
                probe_success = True
        except Exception as probe_err:
            logger.critical(f"CUDA context corrupted during diagnostic probe: {probe_err}")
            probe_success = False

        if probe_success:
            self._state = LifecycleState.UNLOADED
            return EngineOOMError(
                f"Recoverable OOM in {self.model_id}. Model references severed, cache emptied; "
                "state reset to UNLOADED. Note: CUDA probe is preliminary diagnostic only.",
                is_recoverable=True,
            )
        else:
            self._state = LifecycleState.FAILED
            return EngineOOMError(
                f"Unrecoverable CUDA context fault in {self.model_id}. Worker process restart required.",
                is_recoverable=False,
            )

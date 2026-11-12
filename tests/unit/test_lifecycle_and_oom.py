"""Unit tests for ModelLifecycleManager, single-residency coordination, rollback, and OOM recovery (C2, F3, G3)."""

from unittest.mock import patch
import pytest

from src.engine.adapters.qwen3_adapter import Qwen3EngineAdapter
from src.engine.errors import ModelLoadError
from src.engine.lifecycle import ModelLifecycleManager
from src.engine.types import LifecycleState


def test_lifecycle_state_transitions():
    """Verify clean state flow UNLOADED -> LOADING -> READY -> UNLOADED."""
    mgr = ModelLifecycleManager("1.7B-Base")
    assert mgr.state == LifecycleState.UNLOADED

    mgr.begin_loading()
    assert mgr.state == LifecycleState.LOADING

    mgr.mark_ready()
    assert mgr.state == LifecycleState.READY

    mgr.mark_unloaded()
    assert mgr.state == LifecycleState.UNLOADED


def test_vram_requirement_gating():
    """Verify VRAM check function handles available memory check."""
    mgr = ModelLifecycleManager("1.7B-Base")
    # Low requirement should succeed
    assert mgr.check_vram_requirement(10.0) is True


def test_oom_recovery_probe_success():
    """F3: Verify OOM recovery transitions to UNLOADED on successful probe."""
    mgr = ModelLifecycleManager("1.7B-Base")
    mgr.mark_ready()

    dummy_oom = RuntimeError("CUDA out of memory")
    err = mgr.handle_oom(dummy_oom)

    assert err.is_recoverable is True
    assert mgr.state == LifecycleState.UNLOADED


def test_oom_release_callback_executed():
    """G3: Verify that OOM release callback is executed to sever references."""
    mgr = ModelLifecycleManager("0.6B-Base")
    mgr.mark_ready()

    callback_called = False

    def dummy_release():
        nonlocal callback_called
        callback_called = True

    dummy_oom = RuntimeError("CUDA out of memory")
    err = mgr.handle_oom(dummy_oom, release_callback=dummy_release)

    assert callback_called is True
    assert err.is_recoverable is True
    assert mgr.state == LifecycleState.UNLOADED


def test_residency_coordination_and_conflict():
    """G3: Verify single resident model policy across adapters."""
    # Ensure clean starting state
    ModelLifecycleManager.release_residency("model_A")
    ModelLifecycleManager.release_residency("model_B")

    ModelLifecycleManager.acquire_residency("model_A")
    assert ModelLifecycleManager.get_active_resident_id() == "model_A"

    # Same model re-acquiring is allowed
    ModelLifecycleManager.acquire_residency("model_A")

    # Different model acquiring must raise RuntimeError
    with pytest.raises(RuntimeError, match="Residency conflict"):
        ModelLifecycleManager.acquire_residency("model_B")

    # Release residency
    ModelLifecycleManager.release_residency("model_A")
    assert ModelLifecycleManager.get_active_resident_id() is None

    # Now model_B can acquire
    ModelLifecycleManager.acquire_residency("model_B")
    assert ModelLifecycleManager.get_active_resident_id() == "model_B"
    ModelLifecycleManager.release_residency("model_B")


def test_adapter_load_rollback_on_failure():
    """G3: Verify guaranteed rollback when load() encounters an exception."""
    adapter = Qwen3EngineAdapter("test-model", model_path="/non/existent/path/for/rollback")

    with pytest.raises(ModelLoadError):
        adapter.load()

    assert adapter.state == LifecycleState.UNLOADED
    assert adapter.model is None
    # Residency should have been released
    assert ModelLifecycleManager.get_active_resident_id() != "test-model"

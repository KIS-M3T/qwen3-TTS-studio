"""Unit tests for ModelLifecycleManager and OOM recovery protocol (C2, F3)."""

import pytest
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

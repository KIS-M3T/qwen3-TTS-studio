"""Unit tests for ModelLifecycleManager, single-residency coordination, rollback, and OOM recovery (C2, F3, G3, H1, H5, H6)."""

from unittest.mock import MagicMock, patch
import pytest
import torch

from src.engine.adapters.qwen3_adapter import Qwen3EngineAdapter
from src.engine.errors import EngineOOMError, ModelLoadError
from src.engine.lifecycle import ModelLifecycleManager
from src.engine.types import DecodingParams, LifecycleState, VoicePrompt


def test_lifecycle_state_transitions():
    """Verify clean state flow UNLOADED -> LOADING -> READY -> UNLOADED."""
    mgr = ModelLifecycleManager("1.7B-Base", owner_id="test_owner_1")
    assert mgr.state == LifecycleState.UNLOADED

    mgr.begin_loading()
    assert mgr.state == LifecycleState.LOADING

    mgr.mark_ready()
    assert mgr.state == LifecycleState.READY

    mgr.mark_unloaded()
    assert mgr.state == LifecycleState.UNLOADED


def test_vram_requirement_gating():
    """Verify VRAM check function handles available memory check."""
    mgr = ModelLifecycleManager("1.7B-Base", owner_id="test_owner_2")
    # Low requirement should succeed
    assert mgr.check_vram_requirement(10.0) is True


def test_oom_recovery_probe_success():
    """F3: Verify OOM recovery transitions to UNLOADED on successful probe."""
    mgr = ModelLifecycleManager("1.7B-Base", owner_id="test_owner_3")
    mgr.mark_ready()

    dummy_oom = RuntimeError("CUDA out of memory")
    err = mgr.handle_oom(dummy_oom)

    assert err.is_recoverable is True
    assert mgr.state == LifecycleState.UNLOADED


def test_oom_release_callback_executed():
    """G3, H1: Verify that OOM release callback is executed to sever references."""
    mgr = ModelLifecycleManager("0.6B-Base", owner_id="test_owner_4")
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


def test_residency_bound_to_instance_identity():
    """H5: Verify residency is bound to adapter instance owner identity, preventing duplicate residency even for same model_id."""
    # Ensure clean starting state
    owner_a = "adapter_inst_A"
    owner_b = "adapter_inst_B"
    ModelLifecycleManager.release_residency(owner_a)
    ModelLifecycleManager.release_residency(owner_b)

    ModelLifecycleManager.acquire_residency(owner_a, "0.6B-Base")
    assert ModelLifecycleManager.get_active_resident_owner_id() == owner_a
    assert ModelLifecycleManager.get_active_resident_id() == "0.6B-Base"

    # Same owner re-acquiring is idempotent
    ModelLifecycleManager.acquire_residency(owner_a, "0.6B-Base")

    # Distinct adapter instance with same or different model_id must be rejected
    with pytest.raises(RuntimeError, match="Residency conflict"):
        ModelLifecycleManager.acquire_residency(owner_b, "0.6B-Base")

    # Release by wrong owner must not release the active residency
    ModelLifecycleManager.release_residency(owner_b)
    assert ModelLifecycleManager.get_active_resident_owner_id() == owner_a

    # Proper owner release
    ModelLifecycleManager.release_residency(owner_a)
    assert ModelLifecycleManager.get_active_resident_owner_id() is None
    assert ModelLifecycleManager.get_active_resident_id() is None


def test_inference_oom_handling_voice_clone():
    """H1: Verify OOM during synthesize_voice_clone triggers handle_oom and resets state to UNLOADED."""
    adapter = Qwen3EngineAdapter("0.6B-Base", model_path="/fake/path")
    mock_model = MagicMock()
    mock_model.generate_voice_clone.side_effect = torch.cuda.OutOfMemoryError("CUDA out of memory during voice clone")
    adapter.model = mock_model
    adapter.lifecycle.mark_ready()
    ModelLifecycleManager.acquire_residency(adapter.instance_id, adapter.model_id)

    fake_prompt = VoicePrompt(voice_id="v", ref_audio_path="/fake.wav", raw_prompt_payload={"p": 1})

    with pytest.raises(EngineOOMError) as exc_info:
        adapter.synthesize_voice_clone("Test OOM text", voice_prompt=fake_prompt)

    assert exc_info.value.is_recoverable is True
    assert adapter.state == LifecycleState.UNLOADED
    assert adapter.model is None  # References severed (H1)
    assert ModelLifecycleManager.get_active_resident_owner_id() is None  # Residency released


def test_inference_oom_handling_custom_voice():
    """H1: Verify OOM during synthesize_custom_voice triggers handle_oom and resets state to UNLOADED."""
    adapter = Qwen3EngineAdapter("0.6B-CustomVoice", model_path="/fake/path")
    mock_model = MagicMock()
    mock_model.generate_custom_voice.side_effect = torch.cuda.OutOfMemoryError("CUDA out of memory during custom voice")
    adapter.model = mock_model
    adapter.lifecycle.mark_ready()
    ModelLifecycleManager.acquire_residency(adapter.instance_id, adapter.model_id)

    with pytest.raises(EngineOOMError) as exc_info:
        adapter.synthesize_custom_voice("Test OOM text", speaker_id="alessandro")

    assert exc_info.value.is_recoverable is True
    assert adapter.state == LifecycleState.UNLOADED
    assert adapter.model is None
    assert ModelLifecycleManager.get_active_resident_owner_id() is None


def test_inference_oom_handling_base():
    """H1: Verify OOM during synthesize_base triggers handle_oom and resets state to UNLOADED."""
    adapter = Qwen3EngineAdapter("0.6B-Base", model_path="/fake/path")
    mock_model = MagicMock()
    mock_model.generate_voice_clone.side_effect = torch.cuda.OutOfMemoryError("CUDA out of memory during base generation")
    adapter.model = mock_model
    adapter.cached_prompt = {"mock_prompt": True}
    adapter.lifecycle.mark_ready()
    ModelLifecycleManager.acquire_residency(adapter.instance_id, adapter.model_id)

    with pytest.raises(EngineOOMError) as exc_info:
        adapter.synthesize_base("Test OOM text")

    assert exc_info.value.is_recoverable is True
    assert adapter.state == LifecycleState.UNLOADED
    assert adapter.model is None
    assert ModelLifecycleManager.get_active_resident_owner_id() is None


def test_transactional_load_rollback_points():
    """H6: Verify that errors at any point in load() roll back state to UNLOADED without orphan residency."""
    # 1. Missing model path failure
    adapter1 = Qwen3EngineAdapter("model-missing-path", model_path="/completely/invalid/path/12345")
    with pytest.raises(ModelLoadError):
        adapter1.load()
    assert adapter1.state == LifecycleState.UNLOADED
    assert adapter1.model is None
    assert ModelLifecycleManager.get_active_resident_owner_id() is None

    # 2. Insufficient VRAM failure
    adapter2 = Qwen3EngineAdapter("0.6B-Base", model_path="/fake/path")
    with patch.object(adapter2.lifecycle, "check_vram_requirement", return_value=False):
        with pytest.raises(EngineOOMError):
            adapter2.load()
    assert adapter2.state == LifecycleState.UNLOADED
    assert adapter2.model is None
    assert ModelLifecycleManager.get_active_resident_owner_id() is None

    # 3. Residency conflict failure
    ModelLifecycleManager.acquire_residency("other_owner", "other_model")
    adapter3 = Qwen3EngineAdapter("0.6B-Base", model_path="/fake/path")
    with pytest.raises(ModelLoadError, match="Residency conflict"):
        adapter3.load()
    assert adapter3.state == LifecycleState.UNLOADED
    assert adapter3.model is None
    ModelLifecycleManager.release_residency("other_owner")

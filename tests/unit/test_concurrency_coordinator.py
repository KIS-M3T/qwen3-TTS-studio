"""Unit tests for Concurrency Coordinator and thread safety (J2).

Verifies that model lifecycle operations (load, synthesize, unload) are strictly
coordinated across threads to prevent unloading a resident model during inference,
and guarantees deadlock-free operation between TTSEngineFacade and legacy model caches.
"""

import threading
import time
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from src.engine.facade import TTSEngineFacade
from src.engine.types import (
    AudioBuffer,
    DecodingParams,
    GenerationResult,
    LifecycleState,
    MemorySnapshot,
)


class MockAdapter:
    """Mock engine adapter tracking its lifecycle and synthesis execution."""

    def __init__(self, model_id: str):
        self.model_id = model_id
        self.state = LifecycleState.UNLOADED
        self.unloaded = False
        self.synthesizing = False

    def load(self, device="cuda:0", dtype_str="bfloat16"):
        self.state = LifecycleState.READY
        self.unloaded = False

    def unload(self):
        self.state = LifecycleState.UNLOADED
        self.unloaded = True

    def synthesize_base(self, text: str, language: str = "italian", params=None):
        self.synthesizing = True
        try:
            return GenerationResult(
                audio=AudioBuffer(samples=np.zeros(24000, dtype=np.float32), sample_rate=24000),
                inference_time_sec=0.1,
                rtf=0.1,
                memory=MemorySnapshot(0, 0, 0, None),
                model_version=self.model_id,
                decoding_params=params or DecodingParams(),
                environment_manifest_sha256="test_sha",
            )
        finally:
            self.synthesizing = False


@patch("src.engine.facade.Qwen3EngineAdapter")
def test_concurrency_coordinator_synchronized_two_threads(mock_adapter_class):
    """J2 Mandatory test: Two threads with synchronized events.

    Thread 1 generates with model X.
    Thread 2 requests model Y.
    Model X must NOT be unloaded until Thread 1 finishes.
    Model Y is loaded strictly after.
    No deadlock occurs.
    """
    mock_x = MockAdapter(model_id="0.6B-Base")
    mock_y = MockAdapter(model_id="1.7B-Base")

    def adapter_factory(model_id: str):
        if model_id == "0.6B-Base":
            return mock_x
        elif model_id == "1.7B-Base":
            return mock_y
        return MockAdapter(model_id)

    mock_adapter_class.side_effect = adapter_factory
    facade = TTSEngineFacade(default_device="cpu")

    t1_started_synthesis = threading.Event()
    t2_attempted_load = threading.Event()
    t1_completed_synthesis = threading.Event()
    t2_completed_synthesis = threading.Event()

    t1_unloaded_during_inference = False
    execution_order = []

    def mock_synthesize_t1(text: str, language: str = "italian", params=None):
        nonlocal t1_unloaded_during_inference
        execution_order.append("t1_start_synthesis")
        t1_started_synthesis.set()

        # Wait until Thread 2 has initiated its request
        assert t2_attempted_load.wait(timeout=5.0), "Thread 2 did not attempt load in time"
        time.sleep(0.1)  # Simulate GPU computation time while Thread 2 is blocked

        # Critical invariant check: while Thread 1 was generating, mock_x was NOT unloaded!
        if mock_x.unloaded:
            t1_unloaded_during_inference = True

        execution_order.append("t1_finish_synthesis")
        t1_completed_synthesis.set()
        return GenerationResult(
            audio=AudioBuffer(samples=np.zeros(24000, dtype=np.float32), sample_rate=24000),
            inference_time_sec=0.1,
            rtf=0.1,
            memory=MemorySnapshot(0, 0, 0, None),
            model_version="0.6B-Base",
            decoding_params=params or DecodingParams(),
            environment_manifest_sha256="test_sha",
        )

    mock_x.synthesize_base = mock_synthesize_t1

    def run_thread1():
        res = facade.synthesize(text="Thread 1 phrase", model_id="0.6B-Base")
        assert res.model_version == "0.6B-Base"

    def run_thread2():
        # Wait until Thread 1 is inside synthesis
        assert t1_started_synthesis.wait(timeout=5.0), "Thread 1 did not start in time"
        t2_attempted_load.set()
        execution_order.append("t2_request_adapter")

        # This call should block until Thread 1 finishes releasing the coordinator lock
        res = facade.synthesize(text="Thread 2 phrase", model_id="1.7B-Base")
        execution_order.append("t2_finish_synthesis")
        assert res.model_version == "1.7B-Base"
        t2_completed_synthesis.set()

    th1 = threading.Thread(target=run_thread1, name="InferenceWorker-1")
    th2 = threading.Thread(target=run_thread2, name="InferenceWorker-2")

    th1.start()
    th2.start()

    th1.join(timeout=10.0)
    th2.join(timeout=10.0)

    assert not th1.is_alive(), "Thread 1 timed out or deadlocked"
    assert not th2.is_alive(), "Thread 2 timed out or deadlocked"

    assert t1_completed_synthesis.is_set()
    assert t2_completed_synthesis.is_set()
    assert not t1_unloaded_during_inference, "Violation: Model X was unloaded while Thread 1 was generating!"

    # Verify execution order: Thread 1 starts, Thread 2 requests, Thread 1 finishes, Thread 2 finishes
    assert execution_order == [
        "t1_start_synthesis",
        "t2_request_adapter",
        "t1_finish_synthesis",
        "t2_finish_synthesis",
    ]


def test_concurrency_ui_singleton_threadsafe():
    """Verify that get_engine_facade() initializes a single instance safely across concurrent calls."""
    from qwen_tts_ui import get_engine_facade
    import qwen_tts_ui

    instances = []
    errors = []

    def fetch_facade():
        try:
            f = get_engine_facade()
            instances.append(id(f))
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=fetch_facade) for _ in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5.0)

    assert not errors, f"Errors during singleton resolution: {errors}"
    assert len(instances) == 16
    assert len(set(instances)) == 1, "Expected exactly one singleton instance ID across all threads"


def test_concurrency_legacy_model_loader_coordination():
    """Verify coordination and deadlock-free unloads between Facade and legacy model loader."""
    import audio.model_loader as ml
    from src.engine.facade import TTSEngineFacade

    facade = TTSEngineFacade(default_device="cpu")

    # 1. Simulate legacy model loaded in loaded_models
    with ml._model_lock:
        ml.loaded_models["1.7B-VoiceDesign"] = "dummy_legacy_model"

    assert "1.7B-VoiceDesign" in ml.loaded_models

    # 2. Facade unload_legacy_models should cleanly purge legacy cache
    facade.unload_legacy_models()
    assert len(ml.loaded_models) == 0

    # 3. Simulate Facade adapter loaded, then legacy get_model calls hook
    mock_adapter = MockAdapter("0.6B-Base")
    mock_adapter.load()
    facade._active_adapter = mock_adapter

    assert facade._active_adapter is not None

    # Triggering the unload callback registered with ml must unload facade adapter
    if ml._facade_unload_callback:
        ml._facade_unload_callback()

    assert facade._active_adapter is None
    assert mock_adapter.unloaded

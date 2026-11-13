"""Unit tests for Concurrency Coordinator and thread safety (J2).

Verifies that model lifecycle operations (load, synthesize, unload) are strictly
coordinated across threads to prevent unloading a resident model during inference,
and guarantees deadlock-free operation between TTSEngineFacade and legacy model caches.
"""

import subprocess
import sys
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
    """Verify coordination and deadlock-free unloads between Facade and legacy model loader (J2, K2)."""
    import audio.model_loader as ml
    from src.engine.facade import TTSEngineFacade

    facade = TTSEngineFacade(default_device="cpu")

    # 1. Simulate legacy model loaded in loaded_models
    with ml._model_lock:
        ml.loaded_models["1.7B-VoiceDesign"] = "dummy_legacy_model"

    assert "1.7B-VoiceDesign" in ml.loaded_models

    # 2. Facade unload_legacy_models should cleanly purge legacy cache when no lease is active
    facade.unload_legacy_models()
    assert len(ml.loaded_models) == 0

    # 3. Simulate Facade adapter loaded, then legacy trigger_facade_unloads calls hook
    mock_adapter = MockAdapter("0.6B-Base")
    mock_adapter.load()
    facade._active_adapter = mock_adapter

    assert facade._active_adapter is not None

    # Triggering the unload callbacks registered with ml must unload facade adapter
    ml.trigger_facade_unloads()

    assert facade._active_adapter is None
    assert mock_adapter.unloaded


def test_k2_multiple_facades_safe_callback_registry():
    """K2: Verify that multiple facade instances register without overwriting each other."""
    import audio.model_loader as ml
    from src.engine.facade import TTSEngineFacade

    f1 = TTSEngineFacade(default_device="cpu")
    f2 = TTSEngineFacade(default_device="cpu")

    a1 = MockAdapter("0.6B-Base")
    a1.load()
    f1._active_adapter = a1

    a2 = MockAdapter("1.7B-Base")
    a2.load()
    f2._active_adapter = a2

    assert f1._active_adapter is not None
    assert f2._active_adapter is not None

    # Trigger unloads across all registered facades safely
    ml.trigger_facade_unloads()

    assert a1.unloaded
    assert a2.unloaded
    assert f1._active_adapter is None
    assert f2._active_adapter is None

    # Unregister f1
    f1.close()
    f2.close()


def test_k2_concurrent_legacy_podcast_and_facade_synthesis():
    """K2 Mandatory test: Two threads simulating concurrent legacy podcast and new engine facade synthesis.

    Thread 1 runs legacy podcast dialogue generation holding a lease.
    Thread 2 requests a model via new TTSEngineFacade.
    Verify: Thread 2 does NOT evict the legacy model prematurely, does NOT load concurrently,
    and both threads complete without deadlock.
    """
    import audio.model_loader as ml
    from src.engine.facade import TTSEngineFacade

    facade = TTSEngineFacade(default_device="cpu")

    legacy_model_unloaded_during_use = False
    execution_order = []

    legacy_started = threading.Event()
    facade_waiting = threading.Event()
    legacy_finished = threading.Event()
    facade_finished = threading.Event()

    mock_legacy_model = MagicMock()
    mock_facade_adapter = MockAdapter("0.6B-Base")

    def mock_get_model(name):
        return mock_legacy_model

    def run_legacy_podcast_thread():
        nonlocal legacy_model_unloaded_during_use
        execution_order.append("legacy_enter_lease")

        with patch("audio.model_loader.get_model", side_effect=mock_get_model):
            with ml.lease_model("1.7B-CustomVoice") as model:
                with ml._model_lock:
                    ml.loaded_models["1.7B-CustomVoice"] = model

                execution_order.append("legacy_start_generation")
                legacy_started.set()

                # Wait until Thread 2 attempts to use facade
                assert facade_waiting.wait(timeout=5.0), "Thread 2 did not attempt synthesis in time"
                time.sleep(0.1)  # Simulate generation duration while Thread 2 is blocked

                # Invariant: model MUST NOT be evicted from loaded_models while lease is active
                if "1.7B-CustomVoice" not in ml.loaded_models:
                    legacy_model_unloaded_during_use = True

                execution_order.append("legacy_finish_generation")
                legacy_finished.set()

    def run_facade_thread():
        # Wait until legacy generation is in progress
        assert legacy_started.wait(timeout=5.0), "Legacy thread did not start in time"
        facade_waiting.set()
        execution_order.append("facade_request_synthesis")

        def adapter_factory(model_id):
            return mock_facade_adapter

        with patch("src.engine.facade.Qwen3EngineAdapter", side_effect=adapter_factory):
            # Must block until Thread 1's lease_model exits
            res = facade.synthesize(text="Facade phrase", model_id="0.6B-Base")
            execution_order.append("facade_finish_synthesis")
            assert res.model_version == "0.6B-Base"
            facade_finished.set()

    th_legacy = threading.Thread(target=run_legacy_podcast_thread, name="LegacyPodcastWorker")
    th_facade = threading.Thread(target=run_facade_thread, name="FacadeWorker")

    th_legacy.start()
    th_facade.start()

    th_legacy.join(timeout=10.0)
    th_facade.join(timeout=10.0)

    assert not th_legacy.is_alive(), "Legacy podcast thread timed out or deadlocked"
    assert not th_facade.is_alive(), "Facade thread timed out or deadlocked"

    assert legacy_finished.is_set()
    assert facade_finished.is_set()
    assert not legacy_model_unloaded_during_use, "Violation: Legacy model was evicted while podcast generation was in progress!"

    # Invariant: Thread 1 finishes generation BEFORE Thread 2 finishes synthesis
    assert execution_order == [
        "legacy_enter_lease",
        "legacy_start_generation",
        "facade_request_synthesis",
        "legacy_finish_generation",
        "facade_finish_synthesis",
    ]


def test_k2_get_model_direct_call_synchronized_with_active_lease():
    """K2 Mandatory test: Two threads, NO facade callbacks registered.

    Thread 1 holds an active lease on Model A.
    Thread 2 calls get_model('B') directly.
    Verify:
    1. Model A is NOT unloaded during Thread 1's inference.
    2. Thread 2 is blocked by _global_execution_lock and only acquires Model B after Thread 1 releases its lease.
    3. Both threads complete without deadlock.
    """
    import audio.model_loader as ml

    # Ensure clean isolated state
    ml._facade_unload_callbacks.clear()
    ml.loaded_models.clear()
    ml._active_model_leases.clear()
    ml.MAX_LOADED_MODELS = 1

    t1_in_lease = threading.Event()
    t2_calling_get_model = threading.Event()
    t1_done = threading.Event()
    t2_done = threading.Event()

    execution_order = []
    model_a_unloaded_during_use = False

    mock_model_a = MagicMock(name="ModelA")
    mock_model_a.model.eval = MagicMock()
    mock_model_b = MagicMock(name="ModelB")
    mock_model_b.model.eval = MagicMock()

    def mock_from_pretrained(path, **kwargs):
        if "0.6B-Base" in str(path):
            return mock_model_a
        return mock_model_b

    def run_thread1():
        nonlocal model_a_unloaded_during_use
        with ml.lease_model("0.6B-Base") as model:
            execution_order.append("t1_in_lease")
            t1_in_lease.set()

            # Wait until Thread 2 attempts direct get_model call
            assert t2_calling_get_model.wait(timeout=5.0), "Thread 2 did not call get_model in time"
            time.sleep(0.1)  # Simulate active inference time

            # Verify invariant: Model A has NOT been evicted
            if "0.6B-Base" not in ml.loaded_models:
                model_a_unloaded_during_use = True
            if ml.get_active_lease_count("0.6B-Base") != 1:
                model_a_unloaded_during_use = True

            execution_order.append("t1_finish_inference")
            t1_done.set()

    def run_thread2():
        # Wait until Thread 1 has acquired lease
        assert t1_in_lease.wait(timeout=5.0), "Thread 1 did not enter lease in time"
        t2_calling_get_model.set()
        execution_order.append("t2_call_get_model")

        # Direct get_model call without lease: MUST block on _global_execution_lock until Thread 1 exits
        m_b = ml.get_model("1.7B-Base")
        execution_order.append("t2_got_model")
        assert m_b is mock_model_b
        t2_done.set()

    with patch("qwen_tts.Qwen3TTSModel.from_pretrained", side_effect=mock_from_pretrained):
        th1 = threading.Thread(target=run_thread1, name="LeaseWorker")
        th2 = threading.Thread(target=run_thread2, name="DirectCaller")

        th1.start()
        th2.start()

        th1.join(timeout=5.0)
        th2.join(timeout=5.0)

        assert not th1.is_alive(), "Thread 1 timed out or deadlocked"
        assert not th2.is_alive(), "Thread 2 timed out or deadlocked"

        assert t1_done.is_set(), "Thread 1 did not complete"
        assert t2_done.is_set(), "Thread 2 did not complete"
        assert not model_a_unloaded_during_use, "Violation: Model A was evicted while lease was active!"

        # Invariant: Thread 1 finishes inference BEFORE Thread 2 gets Model B
        assert execution_order == [
            "t1_in_lease",
            "t2_call_get_model",
            "t1_finish_inference",
            "t2_got_model",
        ]


def test_k2_get_model_deadlock_subprocess_regression():
    """K2 Regression test: Run the concurrent direct get_model scenario in an isolated subprocess with a strict timeout.

    Guarantees that any potential regression in locking or eviction cannot hang the test runner.
    """
    script = '''
import threading
import time
from unittest.mock import MagicMock, patch
import audio.model_loader as ml

ml._facade_unload_callbacks.clear()
ml.loaded_models.clear()
ml._active_model_leases.clear()
ml.MAX_LOADED_MODELS = 1

t1_in_lease = threading.Event()
t2_called = threading.Event()
t1_done = threading.Event()
t2_done = threading.Event()

mock_a = MagicMock()
mock_a.model.eval = MagicMock()
mock_b = MagicMock()
mock_b.model.eval = MagicMock()

def mock_from_pretrained(path, **kwargs):
    return mock_a if "0.6B-Base" in str(path) else mock_b

def run_t1():
    with ml.lease_model("0.6B-Base"):
        t1_in_lease.set()
        assert t2_called.wait(timeout=3.0)
        time.sleep(0.05)
        t1_done.set()

def run_t2():
    assert t1_in_lease.wait(timeout=3.0)
    t2_called.set()
    m = ml.get_model("1.7B-Base")
    assert m is mock_b
    t2_done.set()

with patch("qwen_tts.Qwen3TTSModel.from_pretrained", side_effect=mock_from_pretrained):
    th1 = threading.Thread(target=run_t1)
    th2 = threading.Thread(target=run_t2)
    th1.start()
    th2.start()
    th1.join(timeout=4.0)
    th2.join(timeout=4.0)

assert not th1.is_alive() and not th2.is_alive()
assert t1_done.is_set() and t2_done.is_set()
print("SUBPROCESS_CONCURRENCY_OK")
'''
    res = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert res.returncode == 0, f"Subprocess failed with stderr:\n{res.stderr}\nstdout:\n{res.stdout}"
    assert "SUBPROCESS_CONCURRENCY_OK" in res.stdout


def test_k2_deterministic_mutual_exclusion_legacy_and_facade_residency():
    """Verify indivisibility of trigger_facade_unloads() + get_model() under _global_execution_lock.

    Demonstrates that interleaved requests between legacy get_model() and facade adapter loading
    cannot result in dual residency of both a legacy model and a facade adapter in memory.
    """
    import audio.model_loader as ml
    from src.engine.facade import TTSEngineFacade

    # Ensure clean state
    ml.loaded_models.clear()
    ml._active_model_leases.clear()

    facade = TTSEngineFacade(default_device="cpu")

    mock_facade_adapter = MockAdapter("0.6B-Base")
    mock_legacy_model = MagicMock(name="LegacyModel")
    mock_legacy_model.model.eval = MagicMock()

    residency_violations = []
    lock = threading.Lock()

    def record_and_verify_residency(step_name: str):
        with lock:
            legacy_count = len(ml.loaded_models)
            has_facade = facade._active_adapter is not None
            if legacy_count > 0 and has_facade:
                residency_violations.append(
                    f"Violation at {step_name}: {legacy_count} legacy model(s) and facade adapter '{facade._active_adapter.model_id}' both resident!"
                )

    # 1. Start with an active facade adapter resident
    mock_facade_adapter.load()
    facade._active_adapter = mock_facade_adapter
    record_and_verify_residency("initial_facade_loaded")
    assert facade._active_adapter is not None
    assert len(ml.loaded_models) == 0

    barrier = threading.Barrier(2)
    thread1_done = threading.Event()
    thread2_done = threading.Event()

    def run_legacy_caller():
        # Both threads synchronize at barrier to start simultaneously
        barrier.wait()
        with patch("qwen_tts.Qwen3TTSModel.from_pretrained", return_value=mock_legacy_model):
            m = ml.get_model("0.6B-Base")
            assert m is mock_legacy_model
            record_and_verify_residency("after_legacy_get_model")
            thread1_done.set()

    def run_facade_caller():
        # Both threads synchronize at barrier to start simultaneously
        barrier.wait()
        def adapter_factory(model_id=None, **kwargs):
            a = MockAdapter(model_id or "default")
            a.load()
            return a

        with patch("src.engine.facade.Qwen3EngineAdapter", side_effect=adapter_factory):
            adapter = facade.get_or_load_adapter("1.7B-Base")
            assert adapter.model_id == "1.7B-Base"
            record_and_verify_residency("after_facade_get_or_load_adapter")
            thread2_done.set()

    th1 = threading.Thread(target=run_legacy_caller, name="LegacyCaller")
    th2 = threading.Thread(target=run_facade_caller, name="FacadeCaller")

    th1.start()
    th2.start()

    th1.join(timeout=5.0)
    th2.join(timeout=5.0)

    assert not th1.is_alive(), "Legacy thread deadlocked"
    assert not th2.is_alive(), "Facade thread deadlocked"
    assert thread1_done.is_set()
    assert thread2_done.is_set()

    # Final post-condition verification
    record_and_verify_residency("final_state")
    assert len(residency_violations) == 0, f"Residency violations detected:\n" + "\n".join(residency_violations)

    # Invariant: exactly one subsystem is resident, never both
    legacy_resident = len(ml.loaded_models) > 0
    facade_resident = facade._active_adapter is not None
    assert legacy_resident != facade_resident, f"Expected exactly one system resident, got legacy={legacy_resident}, facade={facade_resident}"

    facade.close()


def test_k2_deterministic_sequential_interleaved_transitions():
    """Verify that multiple consecutive alternating transitions between legacy and facade preserve single residency."""
    import audio.model_loader as ml
    from src.engine.facade import TTSEngineFacade

    ml.loaded_models.clear()
    ml._active_model_leases.clear()

    facade = TTSEngineFacade(default_device="cpu")
    mock_legacy_model = MagicMock(name="LegacyModel")
    mock_legacy_model.model.eval = MagicMock()

    def adapter_factory(model_id=None, **kwargs):
        a = MockAdapter(model_id or "default")
        a.load()
        return a

    with patch("qwen_tts.Qwen3TTSModel.from_pretrained", return_value=mock_legacy_model), \
         patch("src.engine.facade.Qwen3EngineAdapter", side_effect=adapter_factory):

        for i in range(5):
            # 1. Call legacy get_model
            m = ml.get_model("0.6B-Base")
            assert m is mock_legacy_model
            assert len(ml.loaded_models) == 1
            assert facade._active_adapter is None, f"Iteration {i}: Facade adapter was not unloaded when legacy loaded!"

            # 2. Call facade get_or_load_adapter
            ad = facade.get_or_load_adapter("1.7B-Base")
            assert ad.model_id == "1.7B-Base"
            assert facade._active_adapter is not None
            assert len(ml.loaded_models) == 0, f"Iteration {i}: Legacy models were not purged when facade loaded!"

    facade.close()

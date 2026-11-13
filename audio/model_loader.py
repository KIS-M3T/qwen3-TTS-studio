"""Model loading utilities for Qwen3-TTS - isolated from Gradio UI."""

import os
import shutil
import gc
import functools
import threading
import warnings
from contextlib import contextmanager
from pathlib import Path
from collections import OrderedDict
from typing import Optional

import torch

# Minimum required qwen-tts version for critical tokenizer bugfixes
# (padding bugs in 12Hz tokenizer decode: commits 5f8581d0, 6cafe558)
QWEN_TTS_MIN_VERSION = "0.1.1"

MODEL_PATHS = {
    "1.7B-CustomVoice": "./Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "0.6B-CustomVoice": "./Qwen3-TTS-12Hz-0.6B-CustomVoice",
    "1.7B-Base": "./Qwen3-TTS-12Hz-1.7B-Base",
    "0.6B-Base": "./Qwen3-TTS-12Hz-0.6B-Base",
    "1.7B-VoiceDesign": "./Qwen3-TTS-12Hz-1.7B-VoiceDesign",
}

MAX_LOADED_MODELS = 1

MIN_NEW_TOKENS_DEFAULT = int(os.environ.get("QWEN_TTS_MIN_NEW_TOKENS", "60"))


@functools.lru_cache(maxsize=1)
def _check_qwen_tts_version() -> None:
    """Verify qwen-tts package meets minimum version requirement.

    Raises RuntimeError if the installed version is too old (known tokenizer
    decode bugs that corrupt 12Hz audio output). Can be bypassed with
    QWEN_TTS_ALLOW_OLD=1 environment variable.

    Cached so the check only runs once per process (env-based bypass is
    locked in at first call).
    """
    if os.environ.get("QWEN_TTS_ALLOW_OLD", "").strip() == "1":
        return

    try:
        from importlib.metadata import version as pkg_version
        from packaging.version import Version

        installed = pkg_version("qwen-tts")
        if Version(installed) < Version(QWEN_TTS_MIN_VERSION):
            raise RuntimeError(
                f"qwen-tts {installed} is installed, but >={QWEN_TTS_MIN_VERSION} is required. "
                f"Older versions have known tokenizer decode bugs that corrupt 12Hz audio. "
                f"Run: pip install -U 'qwen-tts>={QWEN_TTS_MIN_VERSION}' "
                f"(set QWEN_TTS_ALLOW_OLD=1 to bypass this check)"
            )
    except ImportError:
        # packaging not available — skip version check but warn
        warnings.warn(
            "Cannot verify qwen-tts version (missing 'packaging' library). "
            f"Ensure qwen-tts >= {QWEN_TTS_MIN_VERSION} is installed.",
            RuntimeWarning,
            stacklevel=2,
        )


def _detect_device() -> str:
    """Auto-detect optimal device with environment variable override.

    Priority: QWEN_TTS_DEVICE env var > MPS > CUDA > CPU.

    Returns:
        Device string for ``device_map`` (e.g. ``"mps"``, ``"cuda:0"``, ``"cpu"``).
    """
    override = os.environ.get("QWEN_TTS_DEVICE", "").strip()
    if override:
        return override

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def _get_attn_implementation(device: str) -> str | None:
    """Return ``"flash_attention_2"`` when available on CUDA, else ``None``.

    FlashAttention is CUDA-only; MPS/CPU devices use default attention.
    Catches any import failure (``ImportError``, ``OSError``, ``RuntimeError``
    from mismatched CUDA toolkits) and silently falls back.
    """
    if not device.startswith("cuda"):
        return None
    try:
        import flash_attn  # noqa: F401

        print(f"[MODEL] FlashAttention 2 available — will attempt on {device}")
        return "flash_attention_2"
    except Exception:
        return None


def _patch_generate_min_tokens(model, min_new_tokens: int = MIN_NEW_TOKENS_DEFAULT):
    """
    Patch model.model.generate to enforce min_new_tokens.

    Prevents premature EOS token generation that causes audio truncation
    (especially for certain voices like 'ryan').
    """
    if not hasattr(model, "model") or not hasattr(model.model, "generate"):
        print(f"[PATCH] Warning: Cannot patch model - no model.model.generate found")
        return model

    original_generate = model.model.generate

    @functools.wraps(original_generate)
    def patched_generate(*args, **kwargs):
        if (
            "min_new_tokens" not in kwargs
            or kwargs.get("min_new_tokens", 0) < min_new_tokens
        ):
            kwargs["min_new_tokens"] = min_new_tokens
        return original_generate(*args, **kwargs)

    model.model.generate = patched_generate
    print(
        f"[PATCH] Applied min_new_tokens={min_new_tokens} to prevent audio truncation"
    )
    return model


loaded_models: OrderedDict = OrderedDict()

# Lock hierarchy (K2):
# Level 1: _global_execution_lock (RLock) — Acquired FIRST before inference or lifecycle transitions.
# Level 2: _model_lock (RLock) — Acquired SECOND for dictionary mutations and lease counts.
# Level 3: _callback_lock (Lock) — Acquired independently for facade callback registry.
# Rule: A thread holding Level 2 or Level 3 must NEVER attempt to acquire Level 1.
_global_execution_lock = threading.RLock()
_model_lock = threading.RLock()
_callback_lock = threading.Lock()
_facade_unload_callbacks: set = set()
_active_model_leases: dict[str, int] = {}


def get_global_execution_lock() -> threading.RLock:
    """Return the shared global execution lock used to coordinate legacy and engine inference (K2)."""
    return _global_execution_lock


def register_facade_unload_callback(callback) -> None:
    """Register callback from Engine Facade to unload resident adapter when legacy model loads (J2, K2)."""
    with _callback_lock:
        _facade_unload_callbacks.add(callback)


def unregister_facade_unload_callback(callback) -> None:
    """Unregister callback from Engine Facade upon shutdown or garbage collection (K2)."""
    with _callback_lock:
        _facade_unload_callbacks.discard(callback)


def trigger_facade_unloads() -> None:
    """Trigger all registered facade unload callbacks safely without holding registry lock (K2)."""
    with _callback_lock:
        cbs = list(_facade_unload_callbacks)
    for cb in cbs:
        try:
            cb()
        except Exception as e:
            print(f"[MODEL] Warning: Failed to trigger facade unload: {e}")


def get_active_lease_count(model_name: Optional[str] = None) -> int:
    """Return number of active leases on a model or across all legacy models (K2)."""
    with _model_lock:
        if model_name is not None:
            return _active_model_leases.get(model_name, 0)
        return sum(_active_model_leases.values())


def _gpu_cleanup():
    gc.collect()
    if torch.backends.mps.is_available() and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _unload_model(model_name: str) -> bool:
    """Unload a model from loaded_models cache, unless an active lease is in progress (K2)."""
    with _model_lock:
        if _active_model_leases.get(model_name, 0) > 0:
            print(f"[MODEL] Skipping unload of '{model_name}': active lease in progress ({_active_model_leases[model_name]} active)")
            return False
        if model_name in loaded_models:
            del loaded_models[model_name]
            _gpu_cleanup()
            print(f"Unloaded {model_name}")
            return True
        return False


@contextmanager
def lease_model(model_name: str):
    """Context manager acquiring a lease on a legacy model during generation (K2).

    Protects the entire sequence 'get_model -> inference -> release':
    1. Acquires _global_execution_lock to prevent conflicting inference or adapter loading.
    2. Triggers registered Facade unloads to free resident adapter VRAM.
    3. Retrieves or loads the model via get_model().
    4. Increments the model lease counter to prevent cache eviction while in use.
    5. Yields the model for inference.
    6. Decrements the lease counter upon context exit and releases the lock.
    """
    with _global_execution_lock:
        trigger_facade_unloads()
        model = get_model(model_name)
        with _model_lock:
            _active_model_leases[model_name] = _active_model_leases.get(model_name, 0) + 1
        try:
            yield model
        finally:
            with _model_lock:
                _active_model_leases[model_name] = max(0, _active_model_leases.get(model_name, 1) - 1)


def get_model(model_name: str):
    """
    Get or load a Qwen3-TTS model.

    Args:
        model_name: Name of the model to load (e.g., "1.7B-CustomVoice")

    Returns:
        Loaded Qwen3TTSModel instance

    Raises:
        ValueError: If model path doesn't exist
        RuntimeError: If model loading fails
    """
    trigger_facade_unloads()

    with _model_lock:
        if model_name in loaded_models:
            loaded_models.move_to_end(model_name)
            return loaded_models[model_name]

        while len(loaded_models) >= MAX_LOADED_MODELS:
            old_name, _ = next(iter(loaded_models.items()))
            _unload_model(old_name)

        _check_qwen_tts_version()

        print(f"Loading {model_name}...")
        from qwen_tts import Qwen3TTSModel

        model_path = MODEL_PATHS.get(model_name)
        if not model_path:
            raise ValueError(
                f"Unknown model: {model_name}. Available: {list(MODEL_PATHS.keys())}"
            )

        if not os.path.exists(model_path):
            raise ValueError(f"Model not found: {model_path}")

        tokenizer_src = Path("Qwen3-TTS-Tokenizer-12Hz/model.safetensors")
        speech_tokenizer_dst = (
            Path(model_path) / "speech_tokenizer" / "model.safetensors"
        )
        if tokenizer_src.exists() and not speech_tokenizer_dst.exists():
            try:
                speech_tokenizer_dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(tokenizer_src, speech_tokenizer_dst)
            except PermissionError as e:
                raise RuntimeError(
                    "Speech tokenizer weights are missing in the model directory, but the app cannot write them. "
                    "This often happens in Docker when model folders are bind-mounted read-only or owned by a "
                    "different UID/GID.\n\n"
                    "Fix: copy the tokenizer weights on the host so the model directory contains:\n"
                    f"  {speech_tokenizer_dst}\n\n"
                    "Source tokenizer weights expected at:\n"
                    f"  {tokenizer_src}\n"
                ) from e

        device = _detect_device()
        attn_impl = _get_attn_implementation(device)
        preferred_dtypes = [torch.bfloat16, torch.float16, torch.float32]
        last_err = None

        for tdtype in preferred_dtypes:
            load_kwargs: dict = {
                "device_map": device,
                "dtype": tdtype,
            }
            if attn_impl is not None:
                load_kwargs["attn_implementation"] = attn_impl

            try:
                m = Qwen3TTSModel.from_pretrained(model_path, **load_kwargs)
            except Exception as e:
                if "attn_implementation" not in load_kwargs:
                    last_err = e
                    _gpu_cleanup()
                    continue
                # FA2 kwarg rejected or FA2 failed at init — retry without it
                print(
                    f"[MODEL] FlashAttention failed ({type(e).__name__}), retrying without it"
                )
                load_kwargs.pop("attn_implementation")
                attn_impl = None
                try:
                    m = Qwen3TTSModel.from_pretrained(model_path, **load_kwargs)
                except Exception as e2:
                    last_err = e2
                    _gpu_cleanup()
                    continue

            try:
                m.model.eval()
            except Exception:
                pass
            _patch_generate_min_tokens(m)
            loaded_models[model_name] = m
            print(f"{model_name} loaded on {device} with {tdtype}!")
            return m

        raise RuntimeError(f"Failed to load {model_name}: {last_err}")


if __name__ == "__main__":
    # Test model loading
    print("Testing model loader...")
    try:
        model = get_model("1.7B-CustomVoice")
        print(f"Model loaded successfully!")
        print(f"Supported speakers: {model.get_supported_speakers()}")
    except Exception as e:
        print(f"Error: {e}")

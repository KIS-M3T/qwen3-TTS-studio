"""Environment manifest collector for experimental repeatability."""

import hashlib
import json
import platform
import sys
from typing import Any, Dict


def get_environment_manifest() -> Dict[str, Any]:
    """Collect hardware, driver, framework, and package versions.

    Returns:
        Dict[str, Any]: Canonical environment dictionary including SHA-256 hash.
    """
    manifest: Dict[str, Any] = {
        "python_version": sys.version.split()[0],
        "python_build": sys.version,
        "platform_system": platform.system(),
        "platform_release": platform.release(),
        "platform_machine": platform.machine(),
    }

    # PyTorch and CUDA details
    try:
        import torch

        manifest["torch_version"] = torch.__version__
        manifest["cuda_available"] = bool(torch.cuda.is_available())
        manifest["cuda_version"] = torch.version.cuda if torch.cuda.is_available() else None
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            manifest["cuda_device_name"] = torch.cuda.get_device_name(0)
            manifest["cuda_device_count"] = torch.cuda.device_count()
            manifest["cuda_arch_list"] = torch.cuda.get_arch_list()
        else:
            manifest["cuda_device_name"] = None
            manifest["cuda_device_count"] = 0
            manifest["cuda_arch_list"] = []
    except ImportError:
        manifest["torch_version"] = None
        manifest["cuda_available"] = False
        manifest["cuda_version"] = None
        manifest["cuda_device_name"] = None

    # Vendor package version
    try:
        from importlib.metadata import version as pkg_version

        manifest["qwen_tts_version"] = pkg_version("qwen-tts")
    except Exception:
        manifest["qwen_tts_version"] = None

    # Compute deterministic SHA-256 of canonical JSON representation
    canonical_json = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    manifest["manifest_sha256"] = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()

    return manifest

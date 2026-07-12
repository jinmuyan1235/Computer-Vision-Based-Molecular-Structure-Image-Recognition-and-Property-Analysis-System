"""CUDA shared-library path helpers for pip-installed NVIDIA wheels."""

from __future__ import annotations

import os
from pathlib import Path
import sys


def nvidia_library_paths() -> list[str]:
    """Return existing NVIDIA library directories under the active Python prefix."""
    version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    root = Path(sys.prefix) / "lib" / version / "site-packages" / "nvidia"
    paths: list[str] = []
    wsl_driver_path = Path("/usr/lib/wsl/lib")
    if wsl_driver_path.is_dir():
        paths.append(str(wsl_driver_path))
    for relative in (
        "cudnn/lib",
        "cublas/lib",
        "cuda_runtime/lib",
        "cuda_cupti/lib",
        "cufft/lib",
        "curand/lib",
        "cusolver/lib",
        "cusparse/lib",
        "nccl/lib",
    ):
        path = root / relative
        if path.is_dir():
            paths.append(str(path))
    return paths


def ensure_cuda_library_path(reexec: bool = False) -> None:
    """Prepend CUDA library paths, optionally restarting Python so dlopen can see them."""
    paths = nvidia_library_paths()
    if not paths:
        return
    existing = [part for part in os.environ.get("LD_LIBRARY_PATH", "").split(":") if part]
    missing = [path for path in paths if path not in existing]
    if not missing:
        return
    os.environ["LD_LIBRARY_PATH"] = ":".join([*missing, *existing])
    if reexec and os.name == "posix" and os.environ.get("MOLECULE_VISION_CUDA_REEXEC") != "1":
        os.environ["MOLECULE_VISION_CUDA_REEXEC"] = "1"
        os.execvpe(sys.executable, [sys.executable, *sys.argv], os.environ)

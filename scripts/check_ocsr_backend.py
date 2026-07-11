"""Print diagnostics for optional OCSR backends without crashing on missing deps."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from typing import Any

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ocsr.demo_adapter import DemoOCSRAdapter
from src.ocsr.decimer_adapter import DECIMERAdapter
from src.ocsr.ensemble import EnsembleOCSRAdapter
from src.ocsr.molscribe_adapter import MolScribeAdapter


def _cuda_available() -> tuple[bool, str | None]:
    try:
        import torch

        return bool(torch.cuda.is_available()), None
    except Exception as exc:
        return False, str(exc)


def check_backend(backend: str) -> dict[str, Any]:
    if backend == "demo":
        cuda_available, torch_error = _cuda_available()
        adapter = DemoOCSRAdapter()
        status = adapter.status()
        status.update(
            {
                "python_version": platform.python_version(),
                "package_installed": True,
                "package_version": "built-in",
                "model_path": None,
                "model_exists": True,
                "model_loaded": True,
                "device": "cpu",
                "cuda_available": cuda_available,
            }
        )
        if torch_error:
            status["torch_error"] = torch_error
        return status
    if backend == "decimer":
        adapter = DECIMERAdapter()
        status = adapter.diagnose(load_model=True)
        status["python_version"] = platform.python_version()
        return status
    if backend == "ensemble":
        adapter = EnsembleOCSRAdapter()
        status = adapter.status()
        status["python_version"] = platform.python_version()
        return status
    adapter = MolScribeAdapter()
    status = adapter.diagnose(load_model=True)
    status["python_version"] = platform.python_version()
    return status


def main() -> int:
    parser = argparse.ArgumentParser(description="Check OCSR backend readiness.")
    parser.add_argument("--backend", choices=["molscribe", "demo", "decimer", "ensemble"], default="molscribe")
    args = parser.parse_args()
    status = check_backend(args.backend)
    print(json.dumps(status, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Backend selection and unified recognition error handling."""

from __future__ import annotations

from typing import Any

import config

from .base import BaseOCSRAdapter, OCSRResult
from .decimer_adapter import DECIMERAdapter
from .demo_adapter import DemoOCSRAdapter
from .ensemble import EnsembleOCSRAdapter
from .molscribe_adapter import MolScribeAdapter


class MoleculeRecognizer:
    """Select and execute one of the configured OCSR adapters."""

    ADAPTERS: dict[str, type[BaseOCSRAdapter]] = {
        "demo": DemoOCSRAdapter,
        "molscribe": MolScribeAdapter,
        "decimer": DECIMERAdapter,
        "ensemble": EnsembleOCSRAdapter,
    }

    def __init__(self, backend: str | None = None) -> None:
        self.backend = (backend or config.OCSR_BACKEND).strip().lower()
        if self.backend not in self.ADAPTERS:
            raise ValueError(f"不支持的 OCSR 后端：{self.backend}。可选值：demo/molscribe/decimer/ensemble。")
        self.adapter = self.ADAPTERS[self.backend]()

    def recognize(self, image_path_or_array: Any) -> OCSRResult:
        """Recognize an image and convert unexpected exceptions to failed results."""
        try:
            return self.adapter.recognize(image_path_or_array)
        except Exception as exc:
            return OCSRResult(None, None, self.backend, "failed", f"OCSR 识别发生未预期错误：{exc}")

    def status(self) -> dict[str, Any]:
        """Return readiness information for the selected adapter."""
        return self.adapter.status()

    @property
    def preferred_image_stage(self) -> str:
        """Return which preprocessing stage should be sent to the selected backend."""
        return getattr(self.adapter, "preferred_image_stage", "preprocessed")

    @property
    def is_demo(self) -> bool:
        """Return whether the recognizer is using the demonstration backend."""
        return self.backend == "demo"

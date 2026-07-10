"""Safe, optional MolScribe backend adapter."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .base import BaseOCSRAdapter, OCSRResult


class MolScribeAdapter(BaseOCSRAdapter):
    """Wrap MolScribe without making it a mandatory project dependency."""

    backend_name = "molscribe"

    def __init__(self, model_path: str | None = None, device: str | None = None) -> None:
        self.model_path = model_path or os.getenv("MOLSCRIBE_MODEL_PATH")
        self.device = device or os.getenv("OCSR_DEVICE", "cpu")
        self.model = None
        self.import_error: str | None = None
        try:
            from molscribe import MolScribe  # type: ignore

            if self.model_path:
                self.model = MolScribe(self.model_path, device=self.device)
            else:
                self.import_error = "已检测到 MolScribe，但未设置 MOLSCRIBE_MODEL_PATH。"
        except (ImportError, ModuleNotFoundError) as exc:
            self.import_error = f"未安装 MolScribe：{exc}"
        except Exception as exc:
            self.import_error = f"MolScribe 模型加载失败：{exc}"

    def recognize(self, image_path_or_array: Any) -> OCSRResult:
        """Run MolScribe inference when its package and checkpoint are available."""
        if self.model is None:
            return OCSRResult(None, None, self.backend_name, "failed", self.import_error or "MolScribe 不可用。")
        try:
            image_path = str(Path(image_path_or_array))
            try:
                prediction = self.model.predict_image_file(image_path, return_confidence=True)
            except TypeError:
                # Older releases may not expose the return_confidence keyword.
                prediction = self.model.predict_image_file(image_path)
            if isinstance(prediction, dict):
                smiles = prediction.get("smiles") or prediction.get("predicted_smiles")
                confidence = prediction.get("confidence")
            else:
                smiles, confidence = str(prediction), None
            if not smiles:
                return OCSRResult(None, confidence, self.backend_name, "failed", "MolScribe 未返回 SMILES。")
            return OCSRResult(str(smiles), confidence, self.backend_name, "success", "MolScribe 识别完成。")
        except Exception as exc:
            return OCSRResult(None, None, self.backend_name, "failed", f"MolScribe 推理失败：{exc}")

    @property
    def is_available(self) -> bool:
        """Return whether a MolScribe checkpoint was loaded successfully."""
        return self.model is not None

    @property
    def availability_message(self) -> str:
        """Describe the current MolScribe configuration state."""
        return self.import_error or "MolScribe 模型已加载。"

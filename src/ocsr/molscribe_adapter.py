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
            # MolScribe releases may expose slightly different return fields.
            prediction = self.model.predict_image_file(str(Path(image_path_or_array)))
            if isinstance(prediction, dict):
                smiles = prediction.get("smiles") or prediction.get("predicted_smiles")
                confidence = prediction.get("confidence")
            else:
                smiles, confidence = str(prediction), None
            if not smiles:
                return OCSRResult(None, confidence, self.backend_name, "failed", "MolScribe 未返回 SMILES。")
            return OCSRResult(str(smiles), confidence, self.backend_name, "success", "MolScribe 识别完成。")
        except Exception as exc:
            # TODO: If a selected MolScribe release changes its inference API,
            # adapt the call above according to that release's documentation.
            return OCSRResult(None, None, self.backend_name, "failed", f"MolScribe 推理失败：{exc}")

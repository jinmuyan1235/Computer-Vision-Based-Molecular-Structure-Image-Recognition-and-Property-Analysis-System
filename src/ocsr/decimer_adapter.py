"""Safe, optional DECIMER backend adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from .base import BaseOCSRAdapter, OCSRResult


class DECIMERAdapter(BaseOCSRAdapter):
    """Wrap DECIMER while allowing the rest of the project to run without it."""

    backend_name = "decimer"

    def __init__(self) -> None:
        self.predictor: Callable[..., Any] | None = None
        self.import_error: str | None = None
        try:
            from DECIMER import predict_SMILES  # type: ignore

            self.predictor = predict_SMILES
        except (ImportError, ModuleNotFoundError) as exc:
            self.import_error = f"未安装 DECIMER：{exc}"
        except Exception as exc:
            self.import_error = f"DECIMER 初始化失败：{exc}"

    def recognize(self, image_path_or_array: Any) -> OCSRResult:
        """Run DECIMER inference when its optional dependency is available."""
        if self.predictor is None:
            return OCSRResult(None, None, self.backend_name, "failed", self.import_error or "DECIMER 不可用。")
        try:
            image_path = str(Path(image_path_or_array))
            try:
                prediction = self.predictor(image_path, confidence=True)
            except TypeError:
                # Compatibility with DECIMER releases without the confidence flag.
                prediction = self.predictor(image_path)
            if isinstance(prediction, dict):
                smiles = prediction.get("smiles") or prediction.get("SMILES")
                confidence = prediction.get("confidence")
            elif isinstance(prediction, (tuple, list)):
                smiles = prediction[0] if prediction else None
                confidence = prediction[1] if len(prediction) > 1 else None
            else:
                smiles, confidence = prediction, None
            if not smiles:
                return OCSRResult(None, None, self.backend_name, "failed", "DECIMER 未返回 SMILES。")
            numeric_confidence = float(confidence) if confidence is not None else None
            return OCSRResult(str(smiles), numeric_confidence, self.backend_name, "success", "DECIMER 识别完成。")
        except Exception as exc:
            return OCSRResult(None, None, self.backend_name, "failed", f"DECIMER 推理失败：{exc}")

    @property
    def is_available(self) -> bool:
        """Return whether DECIMER exposed its prediction function."""
        return self.predictor is not None

    @property
    def availability_message(self) -> str:
        """Describe the current DECIMER import state."""
        return self.import_error or "DECIMER 后端已加载。"

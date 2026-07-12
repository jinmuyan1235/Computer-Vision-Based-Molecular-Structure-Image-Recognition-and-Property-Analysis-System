"""Production-oriented, optional DECIMER backend adapter."""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
from PIL import Image

import config
from src.chem.smiles_validator import validate_smiles
from .base import BaseOCSRAdapter, OCSRResult

ImageStrategy = Literal["original", "grayscale", "normalized", "binary"]


class DECIMERAdapterError(RuntimeError):
    """Base class for classified DECIMER adapter failures."""


class DECIMERDependencyError(DECIMERAdapterError):
    """Raised when DECIMER or TensorFlow cannot be imported."""


class DECIMERConfigurationError(DECIMERAdapterError):
    """Raised when device or input configuration is invalid."""


class DECIMERInitializationError(DECIMERAdapterError):
    """Raised when DECIMER predictor initialization fails."""


class DECIMERInferenceError(DECIMERAdapterError):
    """Raised when DECIMER inference fails or returns no SMILES."""


class DECIMERAdapter(BaseOCSRAdapter):
    """Wrap DECIMER while allowing the rest of the project to run without it."""

    backend_name = "decimer"
    preferred_image_stage = "original"

    def __init__(
        self,
        device: str | None = None,
        timeout_seconds: float | None = None,
        image_strategy: ImageStrategy | None = None,
        strict_mode: bool | None = None,
        model_name: str | None = None,
        model_version: str | None = None,
        hand_drawn: bool = False,
    ) -> None:
        self.requested_device = (device or config.DECIMER_DEVICE or "auto").strip().lower()
        if self.requested_device not in {"cpu", "gpu", "auto"}:
            self.requested_device = "auto"
        self.device = self.requested_device
        self.timeout_seconds = float(timeout_seconds or config.DECIMER_TIMEOUT_SECONDS)
        self.image_strategy: ImageStrategy = image_strategy or config.DECIMER_IMAGE_STRATEGY
        self.strict_mode = config.DECIMER_STRICT_MODE if strict_mode is None else strict_mode
        self.model_name = model_name or config.DECIMER_MODEL_NAME
        self.model_version = model_version or config.DECIMER_MODEL_VERSION
        self.hand_drawn = hand_drawn
        self.package_version = self._detect_package_version()
        self.tensorflow_version: str | None = None
        self.detected_gpus: list[str] = []
        self.predictor: Callable[..., Any] | None = None
        self._load_error: str | None = None
        self.last_inference_time_ms: float | None = None
        self.max_smiles_length = 1000

    @staticmethod
    def _package_installed() -> bool:
        return importlib.util.find_spec("DECIMER") is not None

    @staticmethod
    def _detect_package_version() -> str | None:
        for package_name in ("decimer", "DECIMER"):
            try:
                return importlib.metadata.version(package_name)
            except importlib.metadata.PackageNotFoundError:
                continue
        return None

    def _tensorflow_status(self) -> dict[str, Any]:
        try:
            tensorflow = importlib.import_module("tensorflow")
            gpus = tensorflow.config.list_physical_devices("GPU")
            return {
                "tensorflow_installed": True,
                "tensorflow_version": getattr(tensorflow, "__version__", None),
                "gpu_available": bool(gpus),
                "detected_gpus": [str(gpu) for gpu in gpus],
                "tensorflow": tensorflow,
            }
        except Exception as exc:
            return {
                "tensorflow_installed": False,
                "tensorflow_version": None,
                "gpu_available": False,
                "detected_gpus": [],
                "tensorflow_error": str(exc),
                "tensorflow": None,
            }

    def _resolve_device(self) -> None:
        status = self._tensorflow_status()
        self.tensorflow_version = status.get("tensorflow_version")
        self.detected_gpus = list(status.get("detected_gpus") or [])
        tensorflow = status.get("tensorflow")
        if not status.get("tensorflow_installed"):
            if self.requested_device == "gpu" and self.strict_mode:
                raise DECIMERConfigurationError(f"请求 GPU，但 TensorFlow 不可用：{status.get('tensorflow_error')}")
            self.device = "cpu"
            return
        gpu_available = bool(status.get("gpu_available"))
        if self.requested_device == "gpu" and not gpu_available:
            if self.strict_mode:
                raise DECIMERConfigurationError("请求 GPU，但 TensorFlow 未检测到可用 GPU。")
            self.device = "cpu"
        elif self.requested_device == "auto":
            self.device = "gpu" if gpu_available else "cpu"
        else:
            self.device = self.requested_device
        if self.device == "cpu" and tensorflow is not None:
            try:
                tensorflow.config.set_visible_devices([], "GPU")
            except Exception:
                # TensorFlow may already be initialized; keep reporting actual device.
                pass
        if self.device == "gpu" and tensorflow is not None:
            try:
                for gpu in tensorflow.config.list_physical_devices("GPU"):
                    tensorflow.config.experimental.set_memory_growth(gpu, True)
            except Exception:
                pass

    def _import_predictor(self) -> Callable[..., Any]:
        if not self._package_installed():
            raise DECIMERDependencyError("未安装 DECIMER。请先按 README 安装可选真实 OCSR 后端：pip install decimer。")
        try:
            module = importlib.import_module("DECIMER")
            return getattr(module, "predict_SMILES")
        except (ImportError, AttributeError) as exc:
            raise DECIMERDependencyError(f"DECIMER 包已发现，但无法导入 predict_SMILES：{exc}") from exc

    def _load_predictor(self) -> Callable[..., Any]:
        if self.predictor is not None:
            return self.predictor
        try:
            self._resolve_device()
            self.predictor = self._import_predictor()
            self._load_error = None
            return self.predictor
        except DECIMERAdapterError as exc:
            self._load_error = str(exc)
            raise
        except Exception as exc:
            self._load_error = str(exc)
            raise DECIMERInitializationError(f"DECIMER 初始化失败：{exc}") from exc

    @staticmethod
    def _normalize_array(image: Any) -> np.ndarray:
        array = np.asarray(image)
        if array.ndim not in {2, 3}:
            raise DECIMERInferenceError(f"不支持的图像数组维度：{array.ndim}")
        if array.dtype != np.uint8:
            clipped = np.clip(array, 0, 255)
            if clipped.max(initial=0) <= 1:
                clipped = clipped * 255
            array = clipped.astype(np.uint8)
        if array.ndim == 2:
            array = np.stack([array] * 3, axis=-1)
        if array.shape[-1] == 4:
            array = array[..., :3]
        return array

    @staticmethod
    def _load_array_from_path(path: Path) -> np.ndarray:
        with Image.open(path) as image:
            return np.asarray(image.convert("RGB"))

    def _prepare_input(self, image_path_or_array: Any) -> str | np.ndarray:
        if isinstance(image_path_or_array, (str, Path)):
            path = Path(image_path_or_array).expanduser().resolve()
            if not path.is_file():
                raise DECIMERInferenceError(f"输入图片不存在：{path}")
            if self.image_strategy == "original":
                return str(path)
            array = self._load_array_from_path(path)
        else:
            array = self._normalize_array(image_path_or_array)
            if self.image_strategy == "original":
                return array
        normalized = self._normalize_array(array)
        gray = np.asarray(Image.fromarray(normalized).convert("L"))
        if self.image_strategy == "grayscale":
            return np.stack([gray] * 3, axis=-1)
        if self.image_strategy == "binary":
            threshold = int(gray.mean())
            binary = np.where(gray > threshold, 255, 0).astype(np.uint8)
            return np.stack([binary] * 3, axis=-1)
        minimum = float(gray.min(initial=0))
        maximum = float(gray.max(initial=255))
        scaled = gray.astype(np.float32)
        if maximum > minimum:
            scaled = (scaled - minimum) * (255.0 / (maximum - minimum))
        return np.stack([scaled.astype(np.uint8)] * 3, axis=-1)

    @staticmethod
    def _normalize_confidence(value: Any) -> float | None:
        if value is None:
            return None
        if isinstance(value, (list, tuple, dict)):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _normalize_prediction(self, prediction: Any) -> tuple[str | None, float | None]:
        if isinstance(prediction, dict):
            smiles = (
                prediction.get("smiles")
                or prediction.get("SMILES")
                or prediction.get("predicted_smiles")
                or prediction.get("prediction")
            )
            confidence = prediction.get("confidence") or prediction.get("score") or prediction.get("probability")
            return (str(smiles).strip() if smiles else None, self._normalize_confidence(confidence))
        if isinstance(prediction, (tuple, list)):
            if prediction and isinstance(prediction[0], dict):
                return self._normalize_prediction(prediction[0])
            smiles = prediction[0] if prediction else None
            confidence = prediction[1] if len(prediction) > 1 else None
            return (str(smiles).strip() if smiles else None, self._normalize_confidence(confidence))
        if isinstance(prediction, str):
            return prediction.strip(), None
        smiles = getattr(prediction, "smiles", None) or getattr(prediction, "SMILES", None)
        confidence = getattr(prediction, "confidence", None)
        return (str(smiles).strip() if smiles else None, self._normalize_confidence(confidence))

    def _predict(self, predictor: Callable[..., Any], image_input: str | np.ndarray) -> Any:
        try:
            return predictor(image_input, confidence=True, hand_drawn=self.hand_drawn)
        except TypeError:
            try:
                return predictor(image_input, confidence=True)
            except TypeError:
                return predictor(image_input)

    def _run_with_timeout(self, function: Callable[[], Any]) -> Any:
        if self.timeout_seconds <= 0:
            return function()
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(function)
        try:
            return future.result(timeout=self.timeout_seconds)
        except TimeoutError as exc:
            future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
            raise DECIMERInferenceError(f"DECIMER 推理超过 {self.timeout_seconds:.1f} 秒超时。") from exc
        finally:
            if future.done():
                executor.shutdown(wait=True)

    def _result(
        self,
        smiles: str | None,
        confidence: float | None,
        status: Literal["success", "failed"],
        message: str,
        inference_time_ms: float | None,
        raw_output: str | None = None,
    ) -> OCSRResult:
        return OCSRResult(
            smiles=smiles,
            confidence=confidence,
            backend=self.backend_name,
            status=status,
            message=message,
            inference_time_ms=inference_time_ms,
            model_name=self.model_name,
            model_version=self.model_version,
            device=self.device,
            package_version=self.package_version,
            raw_output=raw_output,
        )

    def recognize(self, image_path_or_array: Any) -> OCSRResult:
        """Run DECIMER inference and normalize the result."""
        start = time.perf_counter()
        try:
            predictor = self._load_predictor()
            image_input = self._prepare_input(image_path_or_array)
            prediction = self._run_with_timeout(lambda: self._predict(predictor, image_input))
            smiles, confidence = self._normalize_prediction(prediction)
            elapsed_ms = round((time.perf_counter() - start) * 1000, 3)
            self.last_inference_time_ms = elapsed_ms
            raw_output = smiles.strip() if isinstance(smiles, str) else None
            if raw_output:
                if len(raw_output) > self.max_smiles_length:
                    return self._result(
                        None,
                        confidence,
                        "failed",
                        "模型返回的结构字符串异常过长，已拒绝作为有效 SMILES。",
                        elapsed_ms,
                        raw_output=raw_output[: self.max_smiles_length],
                    )
                validation = validate_smiles(raw_output)
                if not validation["valid"]:
                    return self._result(
                        None,
                        confidence,
                        "failed",
                        "模型返回了无法解析的结构字符串，请调整区域或使用人工修正。",
                        elapsed_ms,
                        raw_output=raw_output,
                    )
                return self._result(raw_output, confidence, "success", "DECIMER 识别完成。", elapsed_ms, raw_output=raw_output)
            if not smiles:
                raise DECIMERInferenceError("DECIMER 未返回 SMILES。")
            return self._result(smiles, confidence, "success", "DECIMER 识别完成。", elapsed_ms)
        except DECIMERAdapterError as exc:
            elapsed_ms = round((time.perf_counter() - start) * 1000, 3)
            self.last_inference_time_ms = elapsed_ms
            return self._result(None, None, "failed", str(exc), elapsed_ms)
        except Exception as exc:
            elapsed_ms = round((time.perf_counter() - start) * 1000, 3)
            self.last_inference_time_ms = elapsed_ms
            return self._result(None, None, "failed", f"DECIMER 推理失败：{exc}", elapsed_ms)

    @property
    def is_available(self) -> bool:
        tf_status = self._tensorflow_status()
        return self._package_installed() and bool(tf_status.get("tensorflow_installed")) and self._load_error is None

    @property
    def availability_message(self) -> str:
        if not self._package_installed():
            return "未安装 DECIMER。demo、MolScribe、手动 SMILES 和 RDKit 分析仍可正常使用。"
        tf_status = self._tensorflow_status()
        if not tf_status.get("tensorflow_installed"):
            return f"DECIMER 需要 TensorFlow，但当前不可用：{tf_status.get('tensorflow_error')}"
        if self._load_error:
            return f"DECIMER 初始化失败：{self._load_error}"
        if self.predictor is None:
            return "DECIMER 已安装；预测器将在第一次真实识别时延迟初始化。"
        return "DECIMER 预测器已初始化。"

    def status(self) -> dict[str, Any]:
        tf_status = self._tensorflow_status()
        return {
            "backend": self.backend_name,
            "available": self.is_available,
            "message": self.availability_message,
            "package_installed": self._package_installed(),
            "package_version": self.package_version,
            "tensorflow_installed": tf_status.get("tensorflow_installed"),
            "tensorflow_version": self.tensorflow_version or tf_status.get("tensorflow_version"),
            "gpu_available": tf_status.get("gpu_available"),
            "detected_gpus": self.detected_gpus or tf_status.get("detected_gpus") or [],
            "requested_device": self.requested_device,
            "device": self.device,
            "image_strategy": self.image_strategy,
            "timeout_seconds": self.timeout_seconds,
            "strict_mode": self.strict_mode,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "model_loaded": self.predictor is not None,
            "last_inference_time_ms": self.last_inference_time_ms,
        }

    def diagnose(self, load_model: bool = False) -> dict[str, Any]:
        diagnostics = self.status()
        if load_model:
            try:
                self._load_predictor()
                diagnostics.update(self.status())
                diagnostics["initialization_success"] = True
            except DECIMERAdapterError as exc:
                diagnostics.update(self.status())
                diagnostics["initialization_success"] = False
                diagnostics["load_error"] = str(exc)
        return diagnostics

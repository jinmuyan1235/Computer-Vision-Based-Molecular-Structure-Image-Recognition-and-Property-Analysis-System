"""Lightweight user-controlled preprocessing for single-image recognition."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

from config import DEFAULT_IMAGE_SIZE
from src.preprocess.image_loader import load_image


OUTPUT_STAGES = ("original", "grayscale", "normalized", "binary")
DEFAULT_USER_ADJUSTMENTS = {
    "crop_bbox": [],
    "rotation": 0.0,
    "invert": False,
    "contrast": 1.0,
    "trim_whitespace": False,
    "output_stage": "original",
}


def image_dimensions(source: Any) -> dict[str, int]:
    """Return image width and height for UI bounds."""
    image = load_image(source)
    return {"width": int(image.shape[1]), "height": int(image.shape[0])}


def normalize_user_adjustments(
    adjustments: Mapping[str, Any] | None,
    image_shape: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    """Normalize user adjustment values and clamp crop boxes to image bounds."""
    raw = dict(DEFAULT_USER_ADJUSTMENTS)
    raw.update(dict(adjustments or {}))
    normalized = {
        "crop_bbox": [],
        "rotation": _float(raw.get("rotation"), 0.0),
        "invert": bool(raw.get("invert")),
        "contrast": max(0.1, min(4.0, _float(raw.get("contrast"), 1.0))),
        "trim_whitespace": bool(raw.get("trim_whitespace")),
        "output_stage": str(raw.get("output_stage") or "original").lower(),
    }
    if normalized["output_stage"] not in OUTPUT_STAGES:
        normalized["output_stage"] = "original"
    bbox = raw.get("crop_bbox") or []
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        x1, y1, x2, y2 = [_int(value, 0) for value in bbox]
        if image_shape is not None:
            height, width = int(image_shape[0]), int(image_shape[1])
            x1, x2 = sorted((max(0, min(width, x1)), max(0, min(width, x2))))
            y1, y2 = sorted((max(0, min(height, y1)), max(0, min(height, y2))))
        else:
            x1, x2 = sorted((x1, x2))
            y1, y2 = sorted((y1, y2))
        if x2 > x1 and y2 > y1:
            normalized["crop_bbox"] = [x1, y1, x2, y2]
    normalized["rotation"] = round(normalized["rotation"], 3)
    normalized["contrast"] = round(normalized["contrast"], 3)
    return normalized


def has_user_adjustments(adjustments: Mapping[str, Any] | None) -> bool:
    """Return whether adjustments differ from the no-op defaults."""
    normalized = normalize_user_adjustments(adjustments)
    return bool(
        normalized["crop_bbox"]
        or abs(float(normalized["rotation"])) > 0.001
        or normalized["invert"]
        or abs(float(normalized["contrast"]) - 1.0) > 0.001
        or normalized["trim_whitespace"]
        or normalized["output_stage"] != "original"
    )


def apply_user_adjustments(
    source: Any,
    adjustments: Mapping[str, Any] | None,
    default_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
) -> np.ndarray:
    """Apply lightweight user adjustments and return a uint8 image."""
    image = load_image(source)
    normalized = normalize_user_adjustments(adjustments, image.shape)
    working = image.copy()
    bbox = normalized["crop_bbox"]
    if bbox:
        x1, y1, x2, y2 = bbox
        working = working[y1:y2, x1:x2].copy()
    if abs(float(normalized["rotation"])) > 0.001:
        working = _rotate_image(working, float(normalized["rotation"]))
    if normalized["trim_whitespace"]:
        working = _trim_whitespace(working)
    if abs(float(normalized["contrast"]) - 1.0) > 0.001:
        alpha = float(normalized["contrast"])
        working = cv2.convertScaleAbs(working, alpha=alpha, beta=128.0 * (1.0 - alpha))
    if normalized["invert"]:
        working = cv2.bitwise_not(working)

    stage = normalized["output_stage"]
    if stage == "grayscale":
        return _to_grayscale(working)
    if stage == "normalized":
        return _resize_normalize(working, default_size)
    if stage == "binary":
        return _binarize(working)
    return np.clip(working, 0, 255).astype(np.uint8)


def save_user_adjusted_image(
    source: Any,
    adjustments: Mapping[str, Any] | None,
    output_path: str | Path,
    default_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
) -> str:
    """Apply adjustments, save a PNG, and return its absolute path."""
    adjusted = apply_user_adjustments(source, adjustments, default_size=default_size)
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    success, encoded = cv2.imencode(".png", adjusted)
    if not success:
        raise RuntimeError("无法编码人工预处理图片。")
    encoded.tofile(destination)
    return str(destination)


def encode_png(image: np.ndarray) -> bytes:
    """Encode an adjusted image array as PNG bytes."""
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise RuntimeError("无法编码预览图片。")
    return encoded.tobytes()


def _to_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image.copy()
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _binarize(image: np.ndarray) -> np.ndarray:
    gray = _to_grayscale(image)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if float(np.mean(binary)) < 127:
        binary = cv2.bitwise_not(binary)
    return binary


def _resize_normalize(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    gray = _to_grayscale(image)
    target_width, target_height = int(size[0]), int(size[1])
    height, width = gray.shape[:2]
    scale = min(target_width / max(width, 1), target_height / max(height, 1))
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    resized = cv2.resize(gray, (resized_width, resized_height), interpolation=interpolation)
    canvas = np.full((target_height, target_width), 255, dtype=np.uint8)
    x = (target_width - resized_width) // 2
    y = (target_height - resized_height) // 2
    canvas[y : y + resized_height, x : x + resized_width] = resized
    return canvas


def _trim_whitespace(image: np.ndarray, padding: int = 8) -> np.ndarray:
    gray = _to_grayscale(image)
    coordinates = cv2.findNonZero((gray < 245).astype(np.uint8))
    if coordinates is None:
        return image.copy()
    x, y, width, height = cv2.boundingRect(coordinates)
    x0, y0 = max(0, x - padding), max(0, y - padding)
    x1 = min(image.shape[1], x + width + padding)
    y1 = min(image.shape[0], y + height + padding)
    return image[y0:y1, x0:x1].copy()


def _rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    height, width = image.shape[:2]
    normalized_angle = angle % 360
    if abs(normalized_angle) < 0.001:
        return image.copy()
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])
    new_width = max(1, int((height * sin) + (width * cos)))
    new_height = max(1, int((height * cos) + (width * sin)))
    matrix[0, 2] += (new_width / 2.0) - center[0]
    matrix[1, 2] += (new_height / 2.0) - center[1]
    return cv2.warpAffine(image, matrix, (new_width, new_height), flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255))


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default

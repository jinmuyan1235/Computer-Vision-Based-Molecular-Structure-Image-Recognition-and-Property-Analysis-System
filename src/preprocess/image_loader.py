"""Robust image loading for paths, bytes, PIL images and NumPy arrays."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from config import SUPPORTED_IMAGE_EXTENSIONS


def load_image(source: Any) -> np.ndarray:
    """Load an image as a BGR NumPy array and raise readable errors."""
    if isinstance(source, np.ndarray):
        if source.size == 0:
            raise ValueError("图片数组为空。")
        if source.ndim == 2:
            return cv2.cvtColor(source.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        return source.astype(np.uint8).copy()
    if isinstance(source, Image.Image):
        rgb = np.asarray(source.convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if isinstance(source, (bytes, bytearray)):
        image = cv2.imdecode(np.frombuffer(source, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("无法解码上传的图片数据。")
        return image
    path = Path(source).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"图片不存在：{path}")
    if path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
        raise ValueError(f"不支持的图片格式：{path.suffix}。仅支持 PNG/JPG/JPEG。")
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"图片损坏或无法读取：{path}")
    return image


class ImageLoader:
    """Object-oriented facade for image loading and format checks."""

    @staticmethod
    def load(source: Any) -> np.ndarray:
        """Load a supported source as a BGR image."""
        return load_image(source)

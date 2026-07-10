"""Filesystem helpers used by analysis and export modules."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

from config import SUPPORTED_IMAGE_EXTENSIONS


def ensure_directory(path: str | Path) -> Path:
    """Create *path* when needed and return it as an absolute Path."""
    directory = Path(path).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def safe_stem(value: str, fallback: str = "molecule") -> str:
    """Convert an arbitrary filename stem to a safe cross-platform name."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return cleaned or fallback


def iter_image_files(folder: str | Path) -> Iterable[Path]:
    """Yield supported images in deterministic filename order."""
    root = Path(folder).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"输入文件夹不存在或不是目录：{root}")
    yield from sorted(
        (item for item in root.iterdir() if item.is_file() and item.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS),
        key=lambda item: item.name.lower(),
    )

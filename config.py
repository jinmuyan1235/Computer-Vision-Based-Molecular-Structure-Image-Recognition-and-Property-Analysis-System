"""Project-wide configuration loaded from environment variables."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
SAMPLE_DIR = DATA_DIR / "samples"
BATCH_INPUT_DIR = DATA_DIR / "batch_input"
OUTPUT_DIR = DATA_DIR / "outputs"
MODEL_DIR = PROJECT_ROOT / "models"

DEFAULT_IMAGE_SIZE = (512, 512)
OCSR_BACKEND = os.getenv("OCSR_BACKEND", "demo").strip().lower()
OCSR_DEVICE = os.getenv("OCSR_DEVICE", "cpu").strip().lower()
OCSR_TIMEOUT_SECONDS = float(os.getenv("OCSR_TIMEOUT_SECONDS", "120").strip() or "120")
OCSR_USE_PREPROCESSED_IMAGE = os.getenv("OCSR_USE_PREPROCESSED_IMAGE", "false").lower() in {"1", "true", "yes"}
OCSR_STRICT_MODE = os.getenv("OCSR_STRICT_MODE", "false").lower() in {"1", "true", "yes"}
ENABLE_ADMET_MODEL = os.getenv("ENABLE_ADMET_MODEL", "false").lower() in {"1", "true", "yes"}


def _env_path(name: str, default: str | Path) -> Path:
    configured = Path(os.getenv(name, str(default))).expanduser()
    return (configured if configured.is_absolute() else PROJECT_ROOT / configured).resolve()


_configured_admet_path = Path(os.getenv("ADMET_MODEL_PATH", str(MODEL_DIR / "admet_baseline.joblib"))).expanduser()
ADMET_MODEL_PATH = (
    _configured_admet_path if _configured_admet_path.is_absolute() else PROJECT_ROOT / _configured_admet_path
).resolve()
MOLSCRIBE_MODEL_PATH = _env_path("MOLSCRIBE_MODEL_PATH", MODEL_DIR / "molscribe_model.pth")
MOLSCRIBE_MODEL_NAME = os.getenv("MOLSCRIBE_MODEL_NAME", MOLSCRIBE_MODEL_PATH.name).strip()
MOLSCRIBE_MODEL_VERSION = os.getenv("MOLSCRIBE_MODEL_VERSION", "").strip() or None
MOLSCRIBE_IMAGE_STRATEGY: Literal["original", "grayscale", "normalized", "binary"] = os.getenv(
    "MOLSCRIBE_IMAGE_STRATEGY", "original"
).strip().lower()  # type: ignore[assignment]
if MOLSCRIBE_IMAGE_STRATEGY not in {"original", "grayscale", "normalized", "binary"}:
    MOLSCRIBE_IMAGE_STRATEGY = "original"
DECIMER_DEVICE = os.getenv("DECIMER_DEVICE", os.getenv("OCSR_DEVICE", "auto")).strip().lower()
if DECIMER_DEVICE not in {"cpu", "gpu", "auto"}:
    DECIMER_DEVICE = "auto"
DECIMER_TIMEOUT_SECONDS = float(os.getenv("DECIMER_TIMEOUT_SECONDS", os.getenv("OCSR_TIMEOUT_SECONDS", "120")).strip() or "120")
DECIMER_IMAGE_STRATEGY: Literal["original", "grayscale", "normalized", "binary"] = os.getenv(
    "DECIMER_IMAGE_STRATEGY", "original"
).strip().lower()  # type: ignore[assignment]
if DECIMER_IMAGE_STRATEGY not in {"original", "grayscale", "normalized", "binary"}:
    DECIMER_IMAGE_STRATEGY = "original"
DECIMER_MODEL_NAME = os.getenv("DECIMER_MODEL_NAME", "DECIMER Image Transformer").strip()
DECIMER_MODEL_VERSION = os.getenv("DECIMER_MODEL_VERSION", "").strip() or None
DECIMER_STRICT_MODE = os.getenv("DECIMER_STRICT_MODE", os.getenv("OCSR_STRICT_MODE", "false")).lower() in {
    "1",
    "true",
    "yes",
}
SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}

for directory in (DATA_DIR, SAMPLE_DIR, BATCH_INPUT_DIR, OUTPUT_DIR, MODEL_DIR):
    directory.mkdir(parents=True, exist_ok=True)

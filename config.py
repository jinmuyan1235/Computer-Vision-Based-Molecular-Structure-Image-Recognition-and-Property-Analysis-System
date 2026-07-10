"""Project-wide configuration loaded from environment variables."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
SAMPLE_DIR = DATA_DIR / "samples"
BATCH_INPUT_DIR = DATA_DIR / "batch_input"
OUTPUT_DIR = DATA_DIR / "outputs"
MODEL_DIR = PROJECT_ROOT / "models"

DEFAULT_IMAGE_SIZE = (512, 512)
OCSR_BACKEND = os.getenv("OCSR_BACKEND", "demo").strip().lower()
OCSR_DEVICE = os.getenv("OCSR_DEVICE", "cpu").strip().lower()
ENABLE_ADMET_MODEL = os.getenv("ENABLE_ADMET_MODEL", "false").lower() in {"1", "true", "yes"}
_configured_admet_path = Path(os.getenv("ADMET_MODEL_PATH", str(MODEL_DIR / "admet_baseline.joblib"))).expanduser()
ADMET_MODEL_PATH = (
    _configured_admet_path if _configured_admet_path.is_absolute() else PROJECT_ROOT / _configured_admet_path
).resolve()
SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}

for directory in (DATA_DIR, SAMPLE_DIR, BATCH_INPUT_DIR, OUTPUT_DIR, MODEL_DIR):
    directory.mkdir(parents=True, exist_ok=True)

"""Project-wide configuration loaded from environment variables."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
SAMPLE_DIR = DATA_DIR / "samples"
BATCH_INPUT_DIR = DATA_DIR / "batch_input"
OUTPUT_DIR = DATA_DIR / "outputs"

DEFAULT_IMAGE_SIZE = (512, 512)
OCSR_BACKEND = os.getenv("OCSR_BACKEND", "demo").strip().lower()
ENABLE_ADMET_MODEL = os.getenv("ENABLE_ADMET_MODEL", "false").lower() in {"1", "true", "yes"}
SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}

for directory in (DATA_DIR, SAMPLE_DIR, BATCH_INPUT_DIR, OUTPUT_DIR):
    directory.mkdir(parents=True, exist_ok=True)

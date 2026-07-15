"""Configuration loading should be validated and side-effect-light."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import config


def test_load_settings_falls_back_for_bad_numeric_env(monkeypatch) -> None:
    monkeypatch.setenv("OCSR_TIMEOUT_SECONDS", "not-a-number")
    monkeypatch.setenv("DOCUMENT_MAX_PAGES", "bad-int")
    monkeypatch.setenv("OCSR_GPU_MAX_CONCURRENT_INFERENCE", "0")
    monkeypatch.setenv("RUN_RETENTION_DAYS", "0")
    monkeypatch.setenv("RUN_MAX_STORAGE_GB", "bad-float")

    settings = config.load_settings()

    assert settings.ocsr_timeout_seconds == 120.0
    assert settings.document_max_pages == 25
    assert settings.ocsr_gpu_max_concurrent_inference == 1
    assert settings.run_retention_days == 1
    assert settings.run_max_storage_gb == 10.0


def test_app_mode_loads_demo_or_production(monkeypatch) -> None:
    monkeypatch.setenv("APP_MODE", "production")
    assert config.load_settings().app_mode == "production"
    monkeypatch.setenv("APP_MODE", "unexpected")
    assert config.load_settings().app_mode == "demo"


def test_import_config_does_not_create_document_output_dir(tmp_path: Path) -> None:
    target = tmp_path / "not_created_on_import"
    code = "import config, os; print(config.DOCUMENT_OUTPUT_DIR)"
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        env={**dict(os.environ), "DOCUMENT_OUTPUT_DIR": str(target)},
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr
    assert not target.exists()


def test_initialize_directories_is_explicit(monkeypatch, tmp_path: Path) -> None:
    target = tmp_path / "documents"
    runs = tmp_path / "runs"
    monkeypatch.setenv("DOCUMENT_OUTPUT_DIR", str(target))
    monkeypatch.setenv("RUNS_DIR", str(runs))
    settings = config.load_settings()

    config.initialize_directories(settings)

    assert target.is_dir()
    assert runs.is_dir()

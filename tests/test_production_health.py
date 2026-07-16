"""Tests for production startup health checks."""

from pathlib import Path

import pytest

from src.runtime import health as health_module


def _patch_health_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(health_module.config, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(health_module.config, "OUTPUT_DIR", tmp_path / "outputs")
    monkeypatch.setattr(health_module.config, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(health_module.config, "DOCUMENT_OUTPUT_DIR", tmp_path / "documents")


def test_production_health_blocks_demo_backend(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch_health_paths(monkeypatch, tmp_path)

    result = health_module.run_production_health_check(
        "demo",
        production=True,
        warmup=False,
        load_model=False,
        force=True,
        use_cache=False,
    )

    assert result["ready"] is False
    assert result["capabilities"]["smiles_manual"] is True
    assert result["capabilities"]["image_recognition"] is False
    assert any(check["name"] == "backend.policy" and check["status"] == "fail" for check in result["checks"])
    assert any("生产模式" in suggestion for suggestion in result["repair_suggestions"])


def test_health_cache_reuses_successful_warmup(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch_health_paths(monkeypatch, tmp_path)
    warmup_image = tmp_path / "warmup.png"
    warmup_image.write_bytes(b"not-used-by-mock")
    calls = {"warmup": 0}

    monkeypatch.setattr(
        health_module,
        "_backend_status",
        lambda backend, runtime_config, load_model: {
            "backend": backend,
            "available": True,
            "message": "ok",
            "package_installed": True,
            "package_version": "test",
            "model_loaded": bool(load_model),
            "device": "cpu",
        },
    )

    def fake_warmup(backend: str, runtime_config: dict, warmup_input: Path) -> dict:
        calls["warmup"] += 1
        return {"name": "warmup", "status": "pass", "message": "mock warm-up", "details": {"input": str(warmup_input)}}

    monkeypatch.setattr(health_module, "_warmup_check", fake_warmup)

    first = health_module.run_production_health_check(
        "decimer",
        production=True,
        warmup=True,
        load_model=True,
        warmup_input=warmup_image,
        force=False,
        use_cache=True,
        cache_ttl_seconds=3600,
    )
    second = health_module.run_production_health_check(
        "decimer",
        production=True,
        warmup=True,
        load_model=True,
        warmup_input=warmup_image,
        force=False,
        use_cache=True,
        cache_ttl_seconds=3600,
    )

    assert first["ready"] is True
    assert second["ready"] is True
    assert second["cached"] is True
    assert calls["warmup"] == 1


def test_health_failure_disables_only_real_ocsr_workflows(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch_health_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(
        health_module,
        "_backend_status",
        lambda backend, runtime_config, load_model: {
            "backend": backend,
            "available": False,
            "message": "missing model",
            "package_installed": True,
            "model_loaded": False,
            "device": "cpu",
        },
    )

    result = health_module.run_production_health_check(
        "molscribe",
        production=True,
        warmup=False,
        load_model=False,
        force=True,
        use_cache=False,
    )

    assert result["ready"] is False
    assert health_module.image_workflows_enabled(result) is False
    assert result["capabilities"]["smiles_manual"] is True
    assert result["capabilities"]["history"] is True
    assert any(check["name"] == "backend.available" for check in result["checks"])

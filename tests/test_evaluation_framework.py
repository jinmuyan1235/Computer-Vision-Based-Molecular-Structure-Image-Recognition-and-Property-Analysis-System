"""Tests for the standalone OCSR benchmark framework."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from PIL import Image

from src.evaluation.dataset import ManifestValidationError, load_manifest
from src.evaluation.evaluator import OCSREvaluator
from src.evaluation.metrics import compute_metrics, enrich_prediction, tanimoto_similarity
from src.evaluation.report_writer import create_run_directory, write_report_bundle
from src.ocsr.base import BaseOCSRAdapter, OCSRResult
from src.ocsr.recognizer import MoleculeRecognizer


def _image(path: Path) -> Path:
    Image.new("RGB", (24, 24), "white").save(path)
    return path


def _manifest(path: Path, rows: list[dict[str, str]]) -> Path:
    fieldnames = ["sample_id", "image_path", "ground_truth_smiles", "category", "source", "notes"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_manifest_loads_valid_rows(tmp_path: Path) -> None:
    _image(tmp_path / "aspirin.png")
    manifest = _manifest(
        tmp_path / "manifest.csv",
        [{
            "sample_id": "aspirin_001",
            "image_path": "aspirin.png",
            "ground_truth_smiles": "CC(=O)Oc1ccccc1C(=O)O",
            "category": "clean",
            "source": "unit",
            "notes": "ok",
        }],
    )
    samples = load_manifest(manifest, tmp_path)
    assert len(samples) == 1
    assert samples[0].ground_truth_canonical_smiles


def test_manifest_detects_missing_image_invalid_smiles_and_duplicate_ids(tmp_path: Path) -> None:
    _image(tmp_path / "one.png")
    manifest = _manifest(
        tmp_path / "manifest.csv",
        [
            {
                "sample_id": "dup",
                "image_path": "missing.png",
                "ground_truth_smiles": "CCO",
                "category": "clean",
                "source": "unit",
                "notes": "",
            },
            {
                "sample_id": "dup",
                "image_path": "one.png",
                "ground_truth_smiles": "not-a-smiles",
                "category": "clean",
                "source": "unit",
                "notes": "",
            },
        ],
    )
    try:
        load_manifest(manifest, tmp_path)
    except ManifestValidationError as exc:
        message = str(exc)
    else:
        raise AssertionError("Expected manifest validation failure")
    assert "image file does not exist" in message
    assert "duplicate sample_id" in message
    assert "invalid ground_truth_smiles" in message


def test_manifest_detects_path_escape(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside.png"
    _image(outside)
    manifest = _manifest(
        tmp_path / "manifest.csv",
        [{
            "sample_id": "escape",
            "image_path": "../outside.png",
            "ground_truth_smiles": "CCO",
            "category": "clean",
            "source": "unit",
            "notes": "",
        }],
    )
    try:
        load_manifest(manifest, tmp_path)
    except ManifestValidationError as exc:
        assert "escapes dataset root" in str(exc)
    else:
        raise AssertionError("Expected path escape validation failure")


def test_canonical_exact_and_equivalent_smiles() -> None:
    row = enrich_prediction(
        {
            "ground_truth_smiles": "CCO",
            "predicted_smiles": "OCC",
            "recognition_success": True,
            "failure_reason": "",
        },
        similarity_threshold=0.95,
    )
    assert row["rdkit_valid"] is True
    assert row["canonical_exact_match"] is True
    assert row["molecule_equivalent"] is True


def test_invalid_prediction_and_similarity() -> None:
    row = enrich_prediction(
        {
            "ground_truth_smiles": "CCO",
            "predicted_smiles": "not-a-smiles",
            "recognition_success": True,
            "failure_reason": "",
        },
        similarity_threshold=0.95,
    )
    assert row["rdkit_valid"] is False
    assert row["failure_reason"] == "invalid_predicted_smiles"
    assert tanimoto_similarity("CCO", "CCO") == 1.0


class FakeSuccessAdapter(BaseOCSRAdapter):
    backend_name = "fake_success"
    preferred_image_stage = "original"

    def recognize(self, image_path_or_array: Any) -> OCSRResult:
        return OCSRResult("OCC", 0.9, self.backend_name, "success", "ok", inference_time_ms=12.5)


class FakeErrorAdapter(BaseOCSRAdapter):
    backend_name = "fake_error"

    def recognize(self, image_path_or_array: Any) -> OCSRResult:
        raise RuntimeError("backend boom")


def test_evaluator_mock_adapter_success_and_latency(monkeypatch, tmp_path: Path) -> None:
    image = _image(tmp_path / "ethanol.png")
    manifest = _manifest(
        tmp_path / "manifest.csv",
        [{
            "sample_id": "ethanol",
            "image_path": "ethanol.png",
            "ground_truth_smiles": "CCO",
            "category": "clean",
            "source": "unit",
            "notes": "",
        }],
    )
    monkeypatch.setitem(MoleculeRecognizer.ADAPTERS, "fake_success", FakeSuccessAdapter)
    sample = load_manifest(manifest, tmp_path)[0]
    result = OCSREvaluator("fake_success").run([sample])
    row = result["rows"][0]
    assert row["image_path"] == str(image.resolve())
    assert row["canonical_exact_match"] is True
    assert result["metrics"]["overall"]["mean_latency_ms"] == 12.5


def test_evaluator_backend_exception_continues(monkeypatch, tmp_path: Path) -> None:
    _image(tmp_path / "ethanol.png")
    manifest = _manifest(
        tmp_path / "manifest.csv",
        [{
            "sample_id": "ethanol",
            "image_path": "ethanol.png",
            "ground_truth_smiles": "CCO",
            "category": "clean",
            "source": "unit",
            "notes": "",
        }],
    )
    monkeypatch.setitem(MoleculeRecognizer.ADAPTERS, "fake_error", FakeErrorAdapter)
    sample = load_manifest(manifest, tmp_path)[0]
    result = OCSREvaluator("fake_error").run([sample])
    assert result["rows"][0]["recognition_success"] is False
    assert "未预期错误" in result["rows"][0]["message"]


def test_metrics_grouping_and_latency_statistics() -> None:
    rows = [
        enrich_prediction({
            "ground_truth_smiles": "CCO",
            "predicted_smiles": "CCO",
            "recognition_success": True,
            "category": "clean",
            "backend": "fake",
            "preprocessing_strategy": "original",
            "inference_time_ms": 10,
        }, 0.95),
        enrich_prediction({
            "ground_truth_smiles": "CCO",
            "predicted_smiles": None,
            "recognition_success": False,
            "category": "noisy",
            "backend": "fake",
            "preprocessing_strategy": "original",
            "inference_time_ms": 30,
            "failure_reason": "no_smiles",
        }, 0.95),
    ]
    metrics = compute_metrics(rows)
    assert metrics["overall"]["total_samples"] == 2
    assert metrics["overall"]["recognition_success_count"] == 1
    assert metrics["overall"]["median_latency_ms"] == 20.0
    assert "clean" in metrics["groups"]["category"]


def test_run_directories_do_not_overwrite_and_report_bundle(tmp_path: Path) -> None:
    first = create_run_directory(tmp_path, "demo", timestamp="20260711_153000")
    second = create_run_directory(tmp_path, "demo", timestamp="20260711_153000")
    assert first != second
    rows = [
        enrich_prediction({
            "sample_id": "ethanol",
            "image_path": "ethanol.png",
            "ground_truth_smiles": "CCO",
            "predicted_smiles": "CCO",
            "recognition_success": True,
            "recognition_status": "success",
            "category": "clean",
            "source": "unit",
            "backend": "demo",
            "preprocessing_strategy": "original",
            "inference_time_ms": 1.0,
        }, 0.95)
    ]
    result = {
        "rows": rows,
        "metrics": compute_metrics(rows),
        "metadata": {
            "run_started_at": "2026-07-11T15:30:00+08:00",
            "git_commit": "abc",
            "python_version": "3.10",
            "rdkit_version": "test",
            "backend": "demo",
            "backend_status": {"model_name": "demo"},
            "preprocessing_strategy": "original",
            "limitations": "demo only",
        },
    }
    outputs = write_report_bundle(first, result, {"backend": "demo"})
    for key in ("config", "predictions", "metrics", "report", "failure_cases", "charts"):
        assert Path(outputs[key]).exists()

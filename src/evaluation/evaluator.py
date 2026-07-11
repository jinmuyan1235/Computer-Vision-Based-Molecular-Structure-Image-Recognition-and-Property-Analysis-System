"""OCSR benchmark evaluator."""

from __future__ import annotations

import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import config
from src.evaluation.dataset import BenchmarkSample, load_manifest
from src.evaluation.metrics import compute_metrics, enrich_prediction
from src.ocsr.recognizer import MoleculeRecognizer
from src.preprocess.image_preprocessor import ImagePreprocessor


@dataclass(frozen=True)
class EvaluationConfig:
    """Configuration for a benchmark run."""

    manifest: Path
    dataset_root: Path
    backend: str
    output: Path
    preprocessing_strategy: str
    similarity_threshold: float
    limit: int | None
    continue_on_error: bool
    save_predictions: bool


def current_git_commit() -> str | None:
    """Return the current git commit SHA when available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=config.PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def rdkit_version() -> str | None:
    """Return RDKit version without failing benchmark startup."""
    try:
        import rdkit

        return rdkit.__version__
    except Exception:
        return None


class OCSREvaluator:
    """Run OCSR predictions and calculate benchmark metrics."""

    def __init__(
        self,
        backend: str,
        preprocessing_strategy: str = "backend-default",
        similarity_threshold: float = 0.95,
        continue_on_error: bool = True,
    ) -> None:
        self.backend = backend
        self.preprocessing_strategy = preprocessing_strategy
        self.similarity_threshold = similarity_threshold
        self.continue_on_error = continue_on_error
        self.recognizer = MoleculeRecognizer(backend)
        self.preprocessor = ImagePreprocessor()

    def _select_input(self, sample: BenchmarkSample) -> Any:
        strategy = self.preprocessing_strategy
        if strategy == "backend-default":
            strategy = "original" if self.recognizer.preferred_image_stage == "original" else "normalized"
        if strategy == "original":
            return sample.image_path
        stages = self.preprocessor.preprocess_pipeline(sample.image_path)
        if strategy not in stages:
            raise ValueError(f"Unsupported preprocessing strategy: {self.preprocessing_strategy}")
        return stages[strategy]

    def evaluate_sample(self, sample: BenchmarkSample) -> dict[str, Any]:
        """Evaluate a single sample and return a CSV-friendly row."""
        started = time.perf_counter()
        base_row: dict[str, Any] = {
            "sample_id": sample.sample_id,
            "image_path": str(sample.image_path),
            "manifest_image_path": sample.manifest_image_path,
            "ground_truth_smiles": sample.ground_truth_smiles,
            "category": sample.category,
            "source": sample.source,
            "notes": sample.notes,
            "backend": self.backend,
            "preprocessing_strategy": self.preprocessing_strategy,
        }
        try:
            target = self._select_input(sample)
            result = self.recognizer.recognize(target)
            elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
            inference_time_ms = result.inference_time_ms if result.inference_time_ms is not None else elapsed_ms
            base_row.update(
                {
                    "predicted_smiles": result.smiles,
                    "confidence": result.confidence,
                    "recognition_status": result.status,
                    "recognition_success": result.status == "success" and bool(result.smiles),
                    "message": result.message,
                    "failure_reason": "" if result.status == "success" else result.message,
                    "inference_time_ms": inference_time_ms,
                    "model_name": result.model_name,
                    "model_version": result.model_version,
                    "device": result.device,
                    "package_version": result.package_version,
                }
            )
        except Exception as exc:
            if not self.continue_on_error:
                raise
            elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
            base_row.update(
                {
                    "predicted_smiles": None,
                    "confidence": None,
                    "recognition_status": "failed",
                    "recognition_success": False,
                    "message": f"evaluation_error: {exc}",
                    "failure_reason": f"evaluation_error: {exc}",
                    "inference_time_ms": elapsed_ms,
                    "model_name": None,
                    "model_version": None,
                    "device": None,
                    "package_version": None,
                }
            )
        return enrich_prediction(base_row, self.similarity_threshold)

    def run(self, samples: list[BenchmarkSample]) -> dict[str, Any]:
        """Evaluate all samples and return rows, metrics and run metadata."""
        run_started = time.perf_counter()
        rows = [self.evaluate_sample(sample) for sample in samples]
        total_runtime_ms = round((time.perf_counter() - run_started) * 1000, 3)
        metrics = compute_metrics(rows, self.similarity_threshold)
        metrics["overall"]["total_runtime_ms"] = total_runtime_ms
        metadata = {
            "run_started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "git_commit": current_git_commit(),
            "python_version": platform.python_version(),
            "python_executable": sys.executable,
            "rdkit_version": rdkit_version(),
            "backend": self.backend,
            "backend_status": self.recognizer.status(),
            "preprocessing_strategy": self.preprocessing_strategy,
            "similarity_threshold": self.similarity_threshold,
            "total_runtime_ms": total_runtime_ms,
            "limitations": (
                "Demo backend results validate the benchmark framework only and do not represent real OCSR accuracy."
                if self.backend == "demo"
                else "Metrics depend on the configured external OCSR backend and dataset quality."
            ),
        }
        return {"rows": rows, "metrics": metrics, "metadata": metadata}


def run_from_manifest(evaluation_config: EvaluationConfig) -> dict[str, Any]:
    """Load a manifest and run an evaluation."""
    samples = load_manifest(evaluation_config.manifest, evaluation_config.dataset_root)
    if evaluation_config.limit is not None:
        samples = samples[: evaluation_config.limit]
    evaluator = OCSREvaluator(
        backend=evaluation_config.backend,
        preprocessing_strategy=evaluation_config.preprocessing_strategy,
        similarity_threshold=evaluation_config.similarity_threshold,
        continue_on_error=evaluation_config.continue_on_error,
    )
    return evaluator.run(samples)

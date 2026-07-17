"""Immutable filesystem snapshots for visually reviewed detector datasets."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SNAPSHOT_FILES = (
    "detector_training_manifest.csv",
    "visual_verified.csv",
    "visual_rejected.csv",
    "missing_files.csv",
    "review_consistency_report.json",
)
SNAPSHOT_DIRECTORIES = ("single_reviews", "analysis")


def _csv_count(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_visual_dataset(
    *,
    version: str,
    review_dir: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    """Copy all first-round review artifacts into a new non-overwriting version."""
    if not version.strip():
        raise ValueError("version must not be empty.")
    review_root = Path(review_dir).expanduser().resolve()
    target = Path(output).expanduser().resolve()
    if target.exists():
        raise FileExistsError(f"Dataset version already exists and will not be overwritten: {target}")
    required = [review_root / name for name in (*SNAPSHOT_FILES, *SNAPSHOT_DIRECTORIES)]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Required visual-review artifacts are missing: " + ", ".join(missing))
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=target.parent))
    try:
        for name in SNAPSHOT_FILES:
            shutil.copy2(review_root / name, staging / name)
        for name in SNAPSHOT_DIRECTORIES:
            shutil.copytree(review_root / name, staging / name)
        summary = {
            "version": version,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_review_dir": str(review_root),
            "detector_training_samples": _csv_count(staging / "detector_training_manifest.csv"),
            "visual_verified_samples": _csv_count(staging / "visual_verified.csv"),
            "visual_rejected_samples": _csv_count(staging / "visual_rejected.csv"),
            "missing_file_samples": _csv_count(staging / "missing_files.csv"),
            "single_review_records": len(list((staging / "single_reviews").glob("*.json"))),
            "analysis_files": len([path for path in (staging / "analysis").rglob("*") if path.is_file()]),
        }
        (staging / "dataset_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        frozen_files = sorted(
            path for path in staging.rglob("*")
            if path.is_file() and path.name != "checksums.sha256"
        )
        checksum_lines = [f"{_sha256(path)}  {path.relative_to(staging).as_posix()}" for path in frozen_files]
        (staging / "checksums.sha256").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
        staging.replace(target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {**summary, "output": str(target), "checksummed_files": len(checksum_lines)}

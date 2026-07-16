"""Persistent correction feedback store for OCSR data curation."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

from src.chem.smiles_validator import validate_smiles
from src.export.json_exporter import save_json
from src.runtime.run_store import mark_run_protected_from_report
from src.runtime.metadata import sha256_file
from src.utils.file_utils import ensure_directory, safe_stem


CORRECTION_TYPES = {"atom", "bond", "charge", "stereo", "missing_fragment", "other"}
REVIEW_STATUSES = {"pending", "verified", "rejected", "returned", "duplicate", "license_unclear"}
FEEDBACK_ACTIONS = {"correction_only", "accepted_for_dataset"}

FEEDBACK_MANIFEST_FIELDS = [
    "analysis_id",
    "saved_at",
    "image_sha256",
    "image_path",
    "annotation_path",
    "predicted_smiles",
    "predicted_canonical_smiles",
    "corrected_smiles",
    "corrected_canonical_smiles",
    "backend",
    "model_name",
    "model_version",
    "model_sha256",
    "device",
    "correction_type",
    "review_status",
    "feedback_action",
    "include_in_training",
    "duplicate_image",
    "duplicate_of",
    "source_reference",
    "source_license",
    "privacy_notes",
    "notes",
]

EXPORT_MANIFEST_FIELDS = [
    "sample_id",
    "image_path",
    "ground_truth_smiles",
    "expected_action",
    "category",
    "source",
    "split",
    "scaffold_key",
    "source_document",
    "image_quality",
    "complexity",
    "perturbation",
    "structure_features",
    "notes",
]


def _feedback_root(output_dir: str | Path) -> Path:
    return ensure_directory(Path(output_dir).expanduser().resolve() / "feedback")


def feedback_root(output_dir: str | Path) -> Path:
    """Return the feedback directory, creating it when needed."""
    return _feedback_root(output_dir)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except Exception:
        return str(path.resolve())


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [{key: (value or "").strip() for key, value in row.items()} for row in csv.DictReader(handle)]


def _write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def read_feedback_manifest(feedback_root: str | Path) -> list[dict[str, str]]:
    """Read the feedback manifest from a feedback root directory."""
    return _read_csv(Path(feedback_root).expanduser().resolve() / "manifest.csv")


def _source_image_path(report: dict[str, Any]) -> Path | None:
    input_data = report.get("input") or {}
    raw_path = input_data.get("path")
    if not raw_path:
        return None
    path = Path(str(raw_path)).expanduser()
    return path.resolve() if path.exists() else None


def _archive_image(report: dict[str, Any], feedback_root: Path) -> tuple[str | None, Path | None]:
    source = _source_image_path(report)
    input_data = report.get("input") or {}
    image_sha = input_data.get("image_sha256") or (sha256_file(source) if source else None)
    if not image_sha or source is None or not source.is_file():
        return image_sha, None

    image_dir = ensure_directory(feedback_root / "images")
    target = image_dir / f"{image_sha}.png"
    if target.is_file():
        return str(image_sha), target
    try:
        with Image.open(source) as image:
            image.convert("RGB").save(target)
    except Exception:
        target.write_bytes(source.read_bytes())
    return str(image_sha), target


def _duplicate_for(manifest_rows: list[dict[str, str]], image_sha256: str | None, analysis_id: str) -> str | None:
    if not image_sha256:
        return None
    for row in manifest_rows:
        if row.get("image_sha256") == image_sha256 and row.get("analysis_id") != analysis_id:
            return row.get("analysis_id") or None
    return None


def _normalize_choice(value: str | None, allowed: set[str], default: str) -> str:
    normalized = (value or default).strip().lower()
    return normalized if normalized in allowed else default


def _annotation_payload(
    report: dict[str, Any],
    feedback_root: Path,
    image_sha256: str | None,
    archived_image: Path | None,
    duplicate_of: str | None,
    correction_type: str,
    review_status: str,
    feedback_action: str,
    include_in_training: bool,
    notes: str,
    source_reference: str,
    source_license: str,
    privacy_notes: str,
) -> dict[str, Any]:
    input_data = report.get("input") or {}
    ocsr = report.get("ocsr") or {}
    correction = report.get("correction") or {}
    final = report.get("final") or {}
    annotation = {
        "analysis_id": str(report.get("analysis_id") or "analysis"),
        "saved_at": _utc_now_iso(),
        "image_sha256": image_sha256,
        "image_path": _relative_or_absolute(archived_image, feedback_root) if archived_image else None,
        "original_input": {
            "filename": input_data.get("filename"),
            "path": input_data.get("path"),
            "document_id": input_data.get("document_id"),
            "page_number": input_data.get("page_number"),
            "region_id": input_data.get("region_id"),
            "bbox": input_data.get("bbox"),
        },
        "prediction": {
            "predicted_smiles": ocsr.get("predicted_smiles") or ocsr.get("smiles"),
            "predicted_canonical_smiles": ocsr.get("predicted_canonical_smiles"),
            "predicted_standardized_smiles": ocsr.get("predicted_standardized_smiles"),
            "confidence": ocsr.get("confidence"),
            "backend": ocsr.get("backend"),
            "model_name": ocsr.get("model_name"),
            "model_version": ocsr.get("model_version"),
            "model_sha256": ocsr.get("model_sha256"),
            "device": ocsr.get("device"),
        },
        "correction": correction,
        "final": final,
        "images": {
            "predicted_molecule": (report.get("images") or {}).get("predicted_molecule"),
            "corrected_molecule": (report.get("images") or {}).get("corrected_molecule"),
            "redrawn_molecule": (report.get("images") or {}).get("redrawn_molecule"),
        },
        "history": report.get("correction_events") or report.get("audit") or [],
        "feedback": {
            "correction_type": correction_type,
            "review_status": review_status,
            "feedback_action": feedback_action,
            "include_in_training": include_in_training,
            "duplicate_image": duplicate_of is not None,
            "duplicate_of": duplicate_of,
            "source_reference": source_reference,
            "source_license": source_license,
            "privacy_notes": privacy_notes,
            "notes": notes,
        },
    }
    return annotation


def _manifest_row(annotation: dict[str, Any], feedback_root: Path, annotation_path: Path) -> dict[str, Any]:
    prediction = annotation.get("prediction") or {}
    correction = annotation.get("correction") or {}
    feedback = annotation.get("feedback") or {}
    return {
        "analysis_id": annotation.get("analysis_id"),
        "saved_at": annotation.get("saved_at"),
        "image_sha256": annotation.get("image_sha256"),
        "image_path": annotation.get("image_path"),
        "annotation_path": _relative_or_absolute(annotation_path, feedback_root),
        "predicted_smiles": prediction.get("predicted_smiles"),
        "predicted_canonical_smiles": prediction.get("predicted_canonical_smiles"),
        "corrected_smiles": correction.get("corrected_smiles"),
        "corrected_canonical_smiles": correction.get("corrected_canonical_smiles"),
        "backend": prediction.get("backend"),
        "model_name": prediction.get("model_name"),
        "model_version": prediction.get("model_version"),
        "model_sha256": prediction.get("model_sha256"),
        "device": prediction.get("device"),
        "correction_type": feedback.get("correction_type"),
        "review_status": feedback.get("review_status"),
        "feedback_action": feedback.get("feedback_action"),
        "include_in_training": feedback.get("include_in_training"),
        "duplicate_image": feedback.get("duplicate_image"),
        "duplicate_of": feedback.get("duplicate_of"),
        "source_reference": feedback.get("source_reference"),
        "source_license": feedback.get("source_license"),
        "privacy_notes": feedback.get("privacy_notes"),
        "notes": feedback.get("notes"),
    }


def _upsert_manifest_row(manifest_path: Path, row: dict[str, Any]) -> None:
    rows = _read_csv(manifest_path)
    rows = [item for item in rows if item.get("analysis_id") != str(row.get("analysis_id"))]
    rows.append({field: row.get(field, "") for field in FEEDBACK_MANIFEST_FIELDS})
    _write_csv(manifest_path, rows, FEEDBACK_MANIFEST_FIELDS)


def save_feedback_sample(
    report: dict[str, Any],
    output_dir: str | Path,
    notes: str = "",
    correction_type: str = "other",
    review_status: str = "pending",
    feedback_action: str = "correction_only",
    include_in_training: bool | None = None,
    source_reference: str = "",
    source_license: str = "",
    privacy_notes: str = "",
) -> dict[str, Any]:
    """Persist an image, annotation JSON and feedback manifest row."""
    traced = report
    analysis_id = str(traced.get("analysis_id") or "analysis")
    correction = traced.get("correction") or {}
    if not correction.get("corrected_smiles"):
        raise ValueError("保存反馈前需要先应用有效的人工修正。")
    correction_type = _normalize_choice(correction_type, CORRECTION_TYPES, "other")
    review_status = _normalize_choice(review_status, REVIEW_STATUSES, "pending")
    feedback_action = _normalize_choice(feedback_action, FEEDBACK_ACTIONS, "correction_only")
    if include_in_training is None:
        include_in_training = feedback_action == "accepted_for_dataset" and review_status == "verified"
    feedback_root = _feedback_root(output_dir)
    manifest_path = feedback_root / "manifest.csv"
    existing_rows = _read_csv(manifest_path)
    image_sha256, archived_image = _archive_image(traced, feedback_root)
    duplicate_of = _duplicate_for(existing_rows, image_sha256, analysis_id)
    annotation = _annotation_payload(
        traced,
        feedback_root,
        image_sha256,
        archived_image,
        duplicate_of,
        correction_type,
        review_status,
        feedback_action,
        bool(include_in_training),
        notes,
        source_reference,
        source_license,
        privacy_notes,
    )
    annotations_dir = ensure_directory(feedback_root / "annotations")
    annotation_path = annotations_dir / f"{safe_stem(analysis_id)}.json"
    save_json(annotation, annotation_path)
    manifest_row = _manifest_row(annotation, feedback_root, annotation_path)
    _upsert_manifest_row(manifest_path, manifest_row)
    mark_run_protected_from_report(traced, "feedback")
    return {
        "feedback_root": str(feedback_root),
        "annotation_path": str(annotation_path.resolve()),
        "manifest_path": str(manifest_path.resolve()),
        "image_path": str(archived_image.resolve()) if archived_image else None,
        "image_sha256": image_sha256,
        "duplicate_image": duplicate_of is not None,
        "duplicate_of": duplicate_of,
        "include_in_training": bool(include_in_training),
        "review_status": review_status,
    }


def _load_annotation(feedback_root: Path, row: dict[str, str]) -> dict[str, Any] | None:
    annotation_path = row.get("annotation_path") or ""
    path = Path(annotation_path)
    if not path.is_absolute():
        path = feedback_root / annotation_path
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def load_feedback_annotation(feedback_root: str | Path, row: dict[str, str]) -> dict[str, Any] | None:
    """Load a feedback annotation referenced by a manifest row."""
    return _load_annotation(Path(feedback_root).expanduser().resolve(), row)


def update_feedback_review(
    feedback_root: str | Path,
    analysis_id: str,
    review_status: str,
    feedback_action: str | None = None,
    include_in_training: bool | None = None,
    reviewer_notes: str = "",
    duplicate_of: str | None = None,
    source_license: str | None = None,
) -> dict[str, Any]:
    """Update one feedback item after independent review."""
    root = Path(feedback_root).expanduser().resolve()
    manifest_path = root / "manifest.csv"
    rows = _read_csv(manifest_path)
    target = next((row for row in rows if row.get("analysis_id") == analysis_id), None)
    if target is None:
        raise FileNotFoundError(f"未找到待审核反馈：{analysis_id}")
    review_status = _normalize_choice(review_status, REVIEW_STATUSES, "pending")
    if feedback_action is not None:
        feedback_action = _normalize_choice(feedback_action, FEEDBACK_ACTIONS, "correction_only")
    if include_in_training is None:
        include_in_training = review_status == "verified" and feedback_action == "accepted_for_dataset"
    annotation = _load_annotation(root, target)
    if annotation is None:
        raise FileNotFoundError(f"无法读取反馈标注：{target.get('annotation_path')}")

    feedback = annotation.setdefault("feedback", {})
    feedback["review_status"] = review_status
    feedback["feedback_action"] = feedback_action or feedback.get("feedback_action") or "correction_only"
    feedback["include_in_training"] = bool(include_in_training)
    feedback["reviewed_at"] = _utc_now_iso()
    if reviewer_notes:
        feedback["reviewer_notes"] = reviewer_notes
    if duplicate_of is not None:
        feedback["duplicate_image"] = True
        feedback["duplicate_of"] = duplicate_of
    if source_license is not None:
        feedback["source_license"] = source_license

    annotation_path = Path(target.get("annotation_path") or "")
    if not annotation_path.is_absolute():
        annotation_path = root / annotation_path
    save_json(annotation, annotation_path)
    manifest_row = _manifest_row(annotation, root, annotation_path)
    _upsert_manifest_row(manifest_path, manifest_row)
    return {
        "analysis_id": analysis_id,
        "review_status": review_status,
        "feedback_action": manifest_row.get("feedback_action"),
        "include_in_training": bool(manifest_row.get("include_in_training")),
        "annotation_path": str(annotation_path.resolve()),
        "manifest_path": str(manifest_path.resolve()),
    }


def export_feedback_manifest(
    feedback_root: str | Path,
    output_manifest: str | Path,
    split: str = "train",
    review_status: str = "verified",
    keep_duplicates: bool = False,
) -> dict[str, Any]:
    """Export verified feedback rows to the OCSR benchmark manifest format."""
    root = Path(feedback_root).expanduser().resolve()
    manifest_rows = _read_csv(root / "manifest.csv")
    exported: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    required_status = _normalize_choice(review_status, REVIEW_STATUSES, "verified")
    skipped = {"review_status": 0, "not_training": 0, "duplicate": 0, "invalid_smiles": 0, "missing_image": 0}
    for row in manifest_rows:
        if row.get("review_status") != required_status:
            skipped["review_status"] += 1
            continue
        if str(row.get("include_in_training")).lower() not in {"true", "1", "yes"}:
            skipped["not_training"] += 1
            continue
        image_sha = row.get("image_sha256") or ""
        if image_sha in seen_hashes and not keep_duplicates:
            skipped["duplicate"] += 1
            continue
        annotation = _load_annotation(root, row)
        if not annotation:
            skipped["missing_image"] += 1
            continue
        image_path = annotation.get("image_path")
        if not image_path:
            skipped["missing_image"] += 1
            continue
        corrected = ((annotation.get("correction") or {}).get("corrected_smiles") or "").strip()
        validation = validate_smiles(corrected)
        if not validation["valid"]:
            skipped["invalid_smiles"] += 1
            continue
        feedback = annotation.get("feedback") or {}
        prediction = annotation.get("prediction") or {}
        exported.append({
            "sample_id": f"feedback_{safe_stem(str(annotation.get('analysis_id') or image_sha or len(exported)))}",
            "image_path": image_path,
            "ground_truth_smiles": validation["canonical_smiles"],
            "expected_action": "recognize",
            "category": f"feedback_{feedback.get('correction_type') or 'other'}",
            "source": "human_correction_feedback",
            "split": split,
            "scaffold_key": "feedback_unassigned",
            "source_document": feedback.get("source_reference") or "feedback",
            "image_quality": "user_uploaded",
            "complexity": "unspecified",
            "perturbation": "unknown",
            "structure_features": feedback.get("correction_type") or "other",
            "notes": (
                f"backend={prediction.get('backend')}; model={prediction.get('model_name')}; "
                f"license={feedback.get('source_license') or 'unspecified'}; {feedback.get('notes') or ''}"
            ).strip(),
        })
        if image_sha:
            seen_hashes.add(image_sha)
    destination = Path(output_manifest).expanduser().resolve()
    _write_csv(destination, exported, EXPORT_MANIFEST_FIELDS)
    return {
        "output_manifest": str(destination),
        "exported_count": len(exported),
        "skipped": skipped,
    }

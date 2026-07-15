"""Human SMILES correction workflow for image-based OCSR reports."""

from __future__ import annotations

import copy
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rdkit import DataStructs
from rdkit.Chem import AllChem

from config import DATA_DIR, OUTPUT_DIR
from src.chem.descriptors import calculate_descriptors
from src.chem.lipinski import evaluate_lipinski
from src.chem.mol_drawer import draw_molecule
from src.chem.smiles_validator import smiles_to_mol, validate_smiles
from src.chem.standardization import standardize_smiles
from src.analysis.recognition_decision import apply_recognition_decision
from src.ml.admet_baseline import ConfiguredADMETPredictor
from src.utils.file_utils import ensure_directory, safe_stem


def utc_now_iso() -> str:
    """Return a stable UTC timestamp for report metadata."""
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | Path) -> str | None:
    """Return a SHA-256 digest for a local file when it is available."""
    try:
        digest = hashlib.sha256()
        with Path(path).expanduser().resolve().open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except Exception:
        return None


def default_correction_state() -> dict[str, Any]:
    """Return the default correction block for a report."""
    return {
        "applied": False,
        "corrected_smiles": None,
        "corrected_canonical_smiles": None,
        "corrected_standardized_smiles": None,
        "corrected_at": None,
        "source": None,
        "last_error": None,
    }


def default_final_state() -> dict[str, Any]:
    """Return the default final result block for a report."""
    return {"smiles": None, "raw_smiles": None, "canonical_smiles": None, "standardized_smiles": None, "source": None}


def normalize_ocsr_block(ocsr: dict[str, Any] | None) -> dict[str, Any] | None:
    """Add prediction-specific aliases while keeping legacy OCSR fields."""
    if not ocsr:
        return ocsr
    normalized = dict(ocsr)
    predicted = normalized.get("predicted_smiles")
    if predicted is None:
        predicted = normalized.get("smiles")
    normalized["predicted_smiles"] = predicted
    if "predicted_canonical_smiles" not in normalized:
        validation = validate_smiles(predicted)
        normalized["predicted_canonical_smiles"] = validation["canonical_smiles"] if validation["valid"] else None
    if "predicted_standardized_smiles" not in normalized:
        standardized = standardize_smiles(predicted)
        normalized["predicted_standardized_smiles"] = (
            standardized["chemical_identity"]["standardized_smiles"] if standardized["valid"] else None
        )
        normalized["predicted_chemical_identity"] = standardized["chemical_identity"]
        normalized["predicted_standardization"] = standardized["standardization"]
    return normalized


def ensure_traceability_blocks(report: dict[str, Any]) -> dict[str, Any]:
    """Ensure report contains correction/final blocks without mutating input."""
    updated = copy.deepcopy(report)
    updated["ocsr"] = normalize_ocsr_block(updated.get("ocsr"))
    updated.setdefault("correction", default_correction_state())
    for key, value in default_correction_state().items():
        updated["correction"].setdefault(key, value)
    updated.setdefault("final", default_final_state())
    for key, value in default_final_state().items():
        updated["final"].setdefault(key, value)
    updated.setdefault("images", {})
    updated["images"].setdefault("predicted_molecule", None)
    updated["images"].setdefault("corrected_molecule", None)
    updated.setdefault("chemical_identity", None)
    updated.setdefault("standardization", {"profile": None, "changed": False, "steps": [], "warnings": []})
    updated.setdefault("structure_warnings", [])
    return updated


def structure_similarity(smiles_a: str | None, smiles_b: str | None) -> float | None:
    """Return Morgan Tanimoto similarity between two valid SMILES strings."""
    mol_a = smiles_to_mol(smiles_a or "")
    mol_b = smiles_to_mol(smiles_b or "")
    if mol_a is None or mol_b is None:
        return None
    try:
        fp_a = AllChem.GetMorganFingerprintAsBitVect(mol_a, 2, nBits=2048)
        fp_b = AllChem.GetMorganFingerprintAsBitVect(mol_b, 2, nBits=2048)
        return round(float(DataStructs.TanimotoSimilarity(fp_a, fp_b)), 6)
    except Exception:
        return None


def _report_prefix(report: dict[str, Any], suffix: str) -> str:
    analysis_id = str(report.get("analysis_id") or "analysis")
    filename = ((report.get("input") or {}).get("filename") or "molecule").rsplit(".", 1)[0]
    return f"{safe_stem(filename)}_{analysis_id[:8]}_{suffix}"


def _apply_final_smiles(
    report: dict[str, Any],
    smiles: str,
    source: str,
    output_dir: str | Path,
    image_slot: str,
    message: str,
) -> dict[str, Any]:
    """Validate and recalculate all chemistry fields for a final SMILES."""
    standardization_result = standardize_smiles(smiles)
    identity = standardization_result["chemical_identity"]
    validation = {
        "valid": standardization_result["valid"],
        "canonical_smiles": identity.get("canonical_smiles"),
        "standardized_smiles": identity.get("standardized_smiles"),
        "error": standardization_result["error"],
    }
    updated = ensure_traceability_blocks(report)
    if not validation["valid"]:
        updated["message"] = validation["error"]
        updated["validation"] = validation
        updated["chemical_identity"] = identity
        updated["standardization"] = standardization_result["standardization"]
        updated["structure_warnings"] = standardization_result["structure_warnings"]
        return updated
    canonical = str(validation["canonical_smiles"])
    analysis_smiles = str(validation["standardized_smiles"] or canonical)
    descriptors = calculate_descriptors(analysis_smiles)
    lipinski = evaluate_lipinski(descriptors)
    output_root = ensure_directory(output_dir)
    drawing_path = output_root / "structures" / f"{_report_prefix(updated, image_slot)}_structure.png"
    drawing = draw_molecule(analysis_smiles, drawing_path)
    updated["validation"] = validation
    updated["chemical_identity"] = identity
    updated["standardization"] = standardization_result["standardization"]
    updated["structure_warnings"] = standardization_result["structure_warnings"]
    updated["descriptors"] = descriptors
    updated["lipinski"] = lipinski
    updated["admet"] = ConfiguredADMETPredictor().predict(analysis_smiles)
    updated["final"] = {
        "smiles": analysis_smiles,
        "raw_smiles": smiles,
        "canonical_smiles": canonical,
        "standardized_smiles": analysis_smiles,
        "source": source,
    }
    updated["images"]["redrawn_molecule"] = drawing
    updated["images"][image_slot] = drawing
    updated["status"] = "success"
    updated["message"] = message
    updated = apply_recognition_decision(updated)
    if source in {"user_correction", "manual_after_ocsr_failure"}:
        updated["recognition_decision"] = {
            "decision": "accepted",
            "risk_level": "low",
            "reason_codes": ["manual_correction"],
            "manual_review_recommended": False,
            "calibrated_confidence": None,
            "quality_score": (updated.get("image_quality") or {}).get("quality_score"),
            "message": "用户已人工修正并重新计算性质。",
        }
    return updated


def apply_smiles_correction(
    report: dict[str, Any],
    corrected_smiles: str,
    output_dir: str | Path = OUTPUT_DIR,
) -> dict[str, Any]:
    """Return a corrected report without mutating the original report."""
    updated = ensure_traceability_blocks(report)
    attempted = (corrected_smiles or "").strip()
    validation = validate_smiles(attempted)
    if not validation["valid"]:
        updated["correction"]["last_error"] = validation["error"]
        updated["correction"]["attempted_smiles"] = attempted
        updated["correction"]["attempted_at"] = utc_now_iso()
        return updated
    source = "manual_after_ocsr_failure"
    ocsr = updated.get("ocsr") or {}
    if ocsr.get("status") == "success" and ocsr.get("predicted_smiles"):
        source = "user_correction"
    corrected = _apply_final_smiles(
        updated,
        attempted,
        source,
        output_dir,
        "corrected_molecule",
        "已应用人工修正并重新计算性质。",
    )
    corrected_validation = corrected.get("validation") or {}
    corrected["correction"] = {
        "applied": True,
        "corrected_smiles": attempted,
        "corrected_canonical_smiles": corrected_validation.get("canonical_smiles"),
        "corrected_standardized_smiles": corrected_validation.get("standardized_smiles"),
        "corrected_at": utc_now_iso(),
        "source": "user",
        "last_error": None,
    }
    return corrected


def restore_original_prediction(report: dict[str, Any], output_dir: str | Path = OUTPUT_DIR) -> dict[str, Any]:
    """Return a report restored to its original model prediction when valid."""
    updated = ensure_traceability_blocks(report)
    ocsr = updated.get("ocsr") or {}
    predicted = ocsr.get("predicted_smiles") or ocsr.get("smiles")
    validation = validate_smiles(predicted)
    if not validation["valid"]:
        updated["correction"]["last_error"] = "原始模型预测无法被 RDKit 解析，不能恢复为最终分析结果。"
        return updated
    restored = _apply_final_smiles(
        updated,
        str(predicted),
        "ocsr",
        output_dir,
        "predicted_molecule",
        "已恢复为模型原始预测并重新计算性质。",
    )
    restored["correction"] = default_correction_state()
    return restored


def save_correction_feedback(
    report: dict[str, Any],
    output_dir: str | Path = DATA_DIR,
    notes: str = "",
    correction_type: str = "other",
    review_status: str = "pending",
    feedback_action: str = "correction_only",
    include_in_training: bool | None = None,
    source_reference: str = "",
    source_license: str = "",
    privacy_notes: str = "",
) -> dict[str, Any]:
    """Persist a user-triggered correction feedback sample and manifest row."""
    from src.feedback.store import save_feedback_sample

    traced = ensure_traceability_blocks(report)
    return save_feedback_sample(
        traced,
        output_dir=output_dir,
        notes=notes,
        correction_type=correction_type,
        review_status=review_status,
        feedback_action=feedback_action,
        include_in_training=include_in_training,
        source_reference=source_reference,
        source_license=source_license,
        privacy_notes=privacy_notes,
    )

"""OCSR benchmark metric calculations."""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict
import re
from typing import Any

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

import config
from src.chem.standardization import identity_key, standardize_smiles
from src.chem.smiles_validator import canonicalize_smiles, smiles_to_mol


def _rate(numerator: int | float, denominator: int | float) -> float:
    return round(float(numerator) / float(denominator), 6) if denominator else 0.0


def percentile(values: list[float], percent: float) -> float | None:
    """Return a nearest-rank percentile for a non-empty sorted list."""
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((percent / 100.0) * (len(ordered) - 1))))
    return round(float(ordered[index]), 3)


def molecule_identity(smiles: str | None, mode: str = "raw", profile: str | None = None) -> tuple[str | None, str | None]:
    """Return canonical SMILES and InChIKey when RDKit can provide them."""
    if not smiles:
        return None, None
    if mode == "standardized":
        return identity_key(smiles, "standardized", profile)
    canonical = canonicalize_smiles(smiles)
    if canonical is None:
        return None, None
    molecule = smiles_to_mol(canonical)
    inchikey: str | None = None
    if molecule is not None:
        try:
            inchikey = Chem.MolToInchiKey(molecule)
        except Exception:
            inchikey = None
    return canonical, inchikey


def _mol_profile(smiles: str | None) -> dict[str, Any]:
    mol = smiles_to_mol(smiles or "")
    if mol is None:
        return {
            "valid": False,
            "atom_count": None,
            "formal_charge": None,
            "bond_type_counts": {},
            "has_stereo": False,
            "canonical_no_stereo": None,
            "canonical_isomeric": None,
        }
    bond_counts = Counter(str(bond.GetBondType()) for bond in mol.GetBonds())
    has_chiral_atoms = bool(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
    has_stereo_bonds = any(str(bond.GetStereo()) != "STEREONONE" for bond in mol.GetBonds())
    return {
        "valid": True,
        "atom_count": int(mol.GetNumAtoms()),
        "formal_charge": int(Chem.GetFormalCharge(mol)),
        "bond_type_counts": dict(bond_counts),
        "has_stereo": bool(has_chiral_atoms or has_stereo_bonds),
        "canonical_no_stereo": Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False),
        "canonical_isomeric": Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True),
    }


def tanimoto_similarity(ground_truth_smiles: str | None, predicted_smiles: str | None) -> float | None:
    """Compute Morgan fingerprint Tanimoto similarity, returning None for invalid molecules."""
    truth_mol = smiles_to_mol(ground_truth_smiles or "")
    predicted_mol = smiles_to_mol(predicted_smiles or "")
    if truth_mol is None or predicted_mol is None:
        return None
    try:
        truth_fp = AllChem.GetMorganFingerprintAsBitVect(truth_mol, 2, nBits=2048)
        predicted_fp = AllChem.GetMorganFingerprintAsBitVect(predicted_mol, 2, nBits=2048)
        return round(float(DataStructs.TanimotoSimilarity(truth_fp, predicted_fp)), 6)
    except Exception:
        return None


def enrich_prediction(
    row: dict[str, Any],
    similarity_threshold: float,
    identity_comparison: str = "raw",
    standardization_profile: str | None = None,
) -> dict[str, Any]:
    """Add RDKit validity, identity and similarity fields to one prediction row."""
    predicted_smiles = row.get("predicted_smiles")
    profile = standardization_profile or config.CHEM_STANDARDIZATION_PROFILE
    truth_raw_canonical, truth_raw_inchikey = molecule_identity(row.get("ground_truth_smiles"), "raw", profile)
    predicted_raw_canonical, predicted_raw_inchikey = molecule_identity(predicted_smiles, "raw", profile)
    truth_standardized, truth_standardized_inchikey = molecule_identity(row.get("ground_truth_smiles"), "standardized", profile)
    predicted_standardized, predicted_standardized_inchikey = molecule_identity(predicted_smiles, "standardized", profile)
    if identity_comparison == "standardized":
        truth_canonical = truth_standardized
        truth_inchikey = truth_standardized_inchikey
        predicted_canonical = predicted_standardized
        predicted_inchikey = predicted_standardized_inchikey
    else:
        truth_canonical = truth_raw_canonical
        truth_inchikey = truth_raw_inchikey
        predicted_canonical = predicted_raw_canonical
        predicted_inchikey = predicted_raw_inchikey
    predicted_standardization = standardize_smiles(predicted_smiles, profile) if predicted_smiles else None
    truth_profile = _mol_profile(row.get("ground_truth_smiles"))
    predicted_profile = _mol_profile(predicted_smiles)
    rdkit_valid = predicted_raw_canonical is not None
    comparison_valid = predicted_canonical is not None
    canonical_exact = bool(comparison_valid and predicted_canonical == truth_canonical)
    equivalent = bool(canonical_exact or (truth_inchikey and predicted_inchikey and truth_inchikey == predicted_inchikey))
    similarity = tanimoto_similarity(row.get("ground_truth_smiles"), predicted_smiles)
    enriched = dict(row)
    enriched.update(
        {
            "identity_comparison": identity_comparison,
            "standardization_profile": profile,
            "ground_truth_canonical_smiles": truth_raw_canonical,
            "predicted_canonical_smiles": predicted_raw_canonical,
            "ground_truth_standardized_smiles": truth_standardized,
            "predicted_standardized_smiles": predicted_standardized,
            "ground_truth_inchikey": truth_raw_inchikey,
            "predicted_inchikey": predicted_raw_inchikey,
            "comparison_ground_truth_smiles": truth_canonical,
            "comparison_predicted_smiles": predicted_canonical,
            "comparison_inchikey": predicted_inchikey,
            "predicted_standardization_changed": (
                bool(predicted_standardization["standardization"]["changed"]) if predicted_standardization else False
            ),
            "rdkit_valid": rdkit_valid,
            "valid_smiles": rdkit_valid,
            "canonical_exact_match": canonical_exact,
            "exact_match": canonical_exact,
            "molecule_equivalent": equivalent,
            "tanimoto_similarity": similarity,
            "similarity_above_threshold": similarity is not None and similarity >= similarity_threshold,
            "rejection_success": (
                str(row.get("expected_action") or "").lower() == "reject"
                and not bool(row.get("recognition_success"))
            ),
            "ground_truth_has_stereo": truth_profile["has_stereo"],
            "stereochemistry_exact_match": (
                bool(
                    truth_profile["has_stereo"]
                    and predicted_profile["valid"]
                    and truth_profile["canonical_isomeric"] == predicted_profile["canonical_isomeric"]
                )
            ),
            "atom_count_match": (
                bool(predicted_profile["valid"] and truth_profile["atom_count"] == predicted_profile["atom_count"])
            ),
            "formal_charge_match": (
                bool(predicted_profile["valid"] and truth_profile["formal_charge"] == predicted_profile["formal_charge"])
            ),
            "bond_type_profile_match": (
                bool(predicted_profile["valid"] and truth_profile["bond_type_counts"] == predicted_profile["bond_type_counts"])
            ),
        }
    )
    if not enriched.get("failure_reason"):
        if enriched.get("recognition_success") and not rdkit_valid:
            enriched["failure_reason"] = "invalid_predicted_smiles"
        elif not enriched.get("recognition_success"):
            enriched["failure_reason"] = enriched.get("message") or "recognition_failed"
    return enriched


def _latency_metrics(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    latencies = [
        float(row["inference_time_ms"])
        for row in rows
        if row.get("inference_time_ms") not in {None, ""}
    ]
    if not latencies:
        return {"mean_latency_ms": None, "median_latency_ms": None, "p95_latency_ms": None}
    return {
        "mean_latency_ms": round(float(statistics.mean(latencies)), 3),
        "p50_latency_ms": percentile(latencies, 50),
        "median_latency_ms": round(float(statistics.median(latencies)), 3),
        "p95_latency_ms": percentile(latencies, 95),
    }


def _confidence_calibration(rows: list[dict[str, Any]], bins: int = 10) -> dict[str, Any]:
    points: list[tuple[float, bool]] = []
    for row in rows:
        confidence = row.get("confidence")
        if confidence in {None, ""}:
            continue
        try:
            value = float(confidence)
        except (TypeError, ValueError):
            continue
        if 0.0 <= value <= 1.0:
            points.append((value, bool(row.get("canonical_exact_match"))))
    if not points:
        return {"confidence_sample_count": 0, "expected_calibration_error": None}
    total = len(points)
    ece = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        if index == bins - 1:
            bucket = [point for point in points if lower <= point[0] <= upper]
        else:
            bucket = [point for point in points if lower <= point[0] < upper]
        if not bucket:
            continue
        accuracy = sum(correct for _, correct in bucket) / len(bucket)
        mean_confidence = sum(confidence for confidence, _ in bucket) / len(bucket)
        ece += (len(bucket) / total) * abs(mean_confidence - accuracy)
    return {"confidence_sample_count": total, "expected_calibration_error": round(float(ece), 6)}


def _rejection_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    negative_markers = ("distractor", "non_molecule", "reaction", "text", "table")
    negatives = [
        row for row in rows
        if str(row.get("expected_action") or "").lower() == "reject"
        or any(
            marker in str(row.get(field) or "").lower()
            for field in ("category", "source", "structure_features", "notes")
            for marker in negative_markers
        )
    ]
    rejected = sum(not bool(row.get("recognition_success")) for row in negatives)
    false_accepts = sum(bool(row.get("recognition_success")) for row in negatives)
    return {
        "rejection_target_count": len(negatives),
        "rejection_count": rejected,
        "rejection_coverage": _rate(rejected, len(negatives)),
        "false_accept_count": false_accepts,
        "false_accept_rate": _rate(false_accepts, len(negatives)),
    }


def _review_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    review_needed = [
        row for row in rows
        if str(row.get("recognition_decision") or "").lower() == "review_needed"
        or bool(row.get("manual_review_recommended"))
    ]
    high_risk_errors = [
        row for row in rows
        if _is_high_risk(row) and _is_error(row)
    ]
    reviewed_high_risk_errors = [
        row for row in high_risk_errors
        if str(row.get("recognition_decision") or "").lower() == "review_needed"
        or bool(row.get("manual_review_recommended"))
    ]
    return {
        "review_needed_count": len(review_needed),
        "review_needed_rate": _rate(len(review_needed), len(rows)),
        "high_risk_error_count": len(high_risk_errors),
        "high_risk_error_review_needed_count": len(reviewed_high_risk_errors),
        "high_risk_error_review_needed_rate": _rate(len(reviewed_high_risk_errors), len(high_risk_errors)),
    }


def _is_error(row: dict[str, Any]) -> bool:
    if str(row.get("expected_action") or "").lower() == "reject":
        return bool(row.get("recognition_success"))
    return not bool(row.get("canonical_exact_match"))


def _is_high_risk(row: dict[str, Any]) -> bool:
    fields = " ".join(
        str(row.get(field) or "").lower()
        for field in ("expected_action", "category", "complexity", "structure_features", "supported_scope", "notes")
    )
    markers = (
        "reject",
        "distractor",
        "non_molecule",
        "reaction",
        "stereo",
        "charge",
        "salt",
        "fragment",
        "metal",
        "multiple",
        "multi",
        "high",
    )
    return any(marker in fields for marker in markers)


def summarize_rows(rows: list[dict[str, Any]], similarity_threshold: float) -> dict[str, Any]:
    """Summarize benchmark rows into denominator-explicit metrics."""
    total = len(rows)
    recognition_success = sum(bool(row.get("recognition_success")) for row in rows)
    rdkit_valid = sum(bool(row.get("rdkit_valid")) for row in rows)
    canonical_exact = sum(bool(row.get("canonical_exact_match")) for row in rows)
    equivalent = sum(bool(row.get("molecule_equivalent")) for row in rows)
    failed = total - recognition_success
    stereo_required = sum(bool(row.get("ground_truth_has_stereo")) for row in rows)
    stereo_exact = sum(
        bool(row.get("ground_truth_has_stereo")) and bool(row.get("stereochemistry_exact_match"))
        for row in rows
    )
    valid_comparisons = sum(bool(row.get("rdkit_valid")) for row in rows)
    atom_count_errors = sum(bool(row.get("rdkit_valid")) and not bool(row.get("atom_count_match")) for row in rows)
    charge_errors = sum(bool(row.get("rdkit_valid")) and not bool(row.get("formal_charge_match")) for row in rows)
    bond_type_errors = sum(bool(row.get("rdkit_valid")) and not bool(row.get("bond_type_profile_match")) for row in rows)
    similarities = [float(row["tanimoto_similarity"]) for row in rows if row.get("tanimoto_similarity") is not None]
    above_threshold = sum(bool(row.get("similarity_above_threshold")) for row in rows)
    failure_reasons = Counter(str(row.get("failure_reason") or "none") for row in rows if row.get("failure_reason"))
    metrics: dict[str, Any] = {
        "total_samples": total,
        "recognition_success_count": recognition_success,
        "recognition_success_rate": _rate(recognition_success, total),
        "rdkit_valid_count": rdkit_valid,
        "rdkit_valid_rate": _rate(rdkit_valid, total),
        "valid_smiles_count": rdkit_valid,
        "valid_smiles_rate": _rate(rdkit_valid, total),
        "canonical_exact_match_count": canonical_exact,
        "canonical_exact_match_rate": _rate(canonical_exact, total),
        "exact_match_count": canonical_exact,
        "exact_match_rate": _rate(canonical_exact, total),
        "molecule_equivalent_count": equivalent,
        "molecule_equivalent_rate": _rate(equivalent, total),
        "stereo_required_count": stereo_required,
        "stereochemistry_exact_count": stereo_exact,
        "stereochemistry_exact_rate": _rate(stereo_exact, stereo_required),
        "atom_count_error_rate": _rate(atom_count_errors, valid_comparisons),
        "formal_charge_error_rate": _rate(charge_errors, valid_comparisons),
        "bond_type_error_rate": _rate(bond_type_errors, valid_comparisons),
        "failed_count": failed,
        "failed_rate": _rate(failed, total),
        "failure_reason_distribution": dict(failure_reasons),
        "similarity_threshold": similarity_threshold,
        "similarity_count": len(similarities),
        "mean_similarity": round(float(statistics.mean(similarities)), 6) if similarities else None,
        "median_similarity": round(float(statistics.median(similarities)), 6) if similarities else None,
        "similarity_above_threshold_count": above_threshold,
        "similarity_above_threshold_rate": _rate(above_threshold, total),
        "denominators": {
            "all_rates": total,
            "similarity_rates": total,
            "latency_metrics": "samples with inference_time_ms",
            "structure_error_rates": "predictions with RDKit-valid SMILES",
            "stereochemistry_exact_rate": "ground-truth samples containing stereochemistry",
        },
    }
    metrics.update(_latency_metrics(rows))
    metrics.update(_confidence_calibration(rows))
    metrics.update(_rejection_metrics(rows))
    metrics.update(_review_metrics(rows))
    return metrics


def group_metrics(rows: list[dict[str, Any]], similarity_threshold: float) -> dict[str, dict[str, Any]]:
    """Compute category/backend/preprocessing_strategy grouped metrics."""
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {
        field: defaultdict(list)
        for field in (
            "category",
            "source",
            "image_quality",
            "complexity",
            "perturbation",
            "structure_features",
            "split",
            "backend",
            "preprocessing_strategy",
        )
    }
    for row in rows:
        for field in grouped:
            grouped[field][str(row.get(field) or "unknown")].append(row)
    return {
        field: {key: summarize_rows(group_rows, similarity_threshold) for key, group_rows in values.items()}
        for field, values in grouped.items()
    }


def compute_metrics(rows: list[dict[str, Any]], similarity_threshold: float = 0.95) -> dict[str, Any]:
    """Compute all benchmark metrics and grouped summaries."""
    return {
        "overall": summarize_rows(rows, similarity_threshold),
        "groups": group_metrics(rows, similarity_threshold),
        "ensemble": ensemble_metrics(rows),
    }


def _candidate_backends(rows: list[dict[str, Any]]) -> list[str]:
    backends: set[str] = set()
    pattern = re.compile(r"^candidate_(.+)_canonical_exact_match$")
    for row in rows:
        for key in row:
            match = pattern.match(key)
            if match:
                backends.add(match.group(1))
    return sorted(backends)


def ensemble_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize ensemble-specific diagnostics without claiming unproven improvement."""
    ensemble_rows = [row for row in rows if row.get("backend") == "ensemble"]
    total = len(ensemble_rows)
    agreement = sum(bool(row.get("ensemble_agreement")) for row in ensemble_rows)
    disagreement = sum(bool(row.get("ensemble_disagreement")) for row in ensemble_rows)
    accepted = sum(bool(row.get("ensemble_accepted")) for row in ensemble_rows)
    accepted_with_warning = sum(bool(row.get("ensemble_accepted_with_warning")) for row in ensemble_rows)
    review_needed = sum(bool(row.get("ensemble_review_needed")) for row in ensemble_rows)
    rejected = sum(bool(row.get("ensemble_rejected")) for row in ensemble_rows)
    candidate_backend_metrics: dict[str, Any] = {}
    candidate_backends = _candidate_backends(ensemble_rows)
    for backend in candidate_backends:
        success_key = f"candidate_{backend}_recognition_success"
        valid_key = f"candidate_{backend}_rdkit_valid"
        exact_key = f"candidate_{backend}_canonical_exact_match"
        equivalent_key = f"candidate_{backend}_molecule_equivalent"
        candidate_backend_metrics[backend] = {
            "total_samples": total,
            "recognition_success_count": sum(bool(row.get(success_key)) for row in ensemble_rows),
            "recognition_success_rate": _rate(sum(bool(row.get(success_key)) for row in ensemble_rows), total),
            "rdkit_valid_count": sum(bool(row.get(valid_key)) for row in ensemble_rows),
            "rdkit_valid_rate": _rate(sum(bool(row.get(valid_key)) for row in ensemble_rows), total),
            "canonical_exact_match_count": sum(bool(row.get(exact_key)) for row in ensemble_rows),
            "canonical_exact_match_rate": _rate(sum(bool(row.get(exact_key)) for row in ensemble_rows), total),
            "molecule_equivalent_count": sum(bool(row.get(equivalent_key)) for row in ensemble_rows),
            "molecule_equivalent_rate": _rate(sum(bool(row.get(equivalent_key)) for row in ensemble_rows), total),
        }
    pairwise: dict[str, int] = {}
    if len(candidate_backends) >= 2:
        for first_index, first in enumerate(candidate_backends):
            for second in candidate_backends[first_index + 1 :]:
                first_key = f"candidate_{first}_canonical_exact_match"
                second_key = f"candidate_{second}_canonical_exact_match"
                pairwise[f"{first}_only_correct_vs_{second}"] = sum(
                    bool(row.get(first_key)) and not bool(row.get(second_key)) for row in ensemble_rows
                )
                pairwise[f"{second}_only_correct_vs_{first}"] = sum(
                    bool(row.get(second_key)) and not bool(row.get(first_key)) for row in ensemble_rows
                )
    any_candidate_correct = [
        any(bool(row.get(f"candidate_{backend}_canonical_exact_match")) for backend in candidate_backends)
        for row in ensemble_rows
    ]
    ensemble_correct = [bool(row.get("canonical_exact_match")) for row in ensemble_rows]
    ensemble_only_correct = sum(
        bool(ensemble_value) and not bool(candidate_value)
        for ensemble_value, candidate_value in zip(ensemble_correct, any_candidate_correct)
    )
    ensemble_missed_candidate_correct = sum(
        bool(candidate_value) and not bool(ensemble_value)
        for ensemble_value, candidate_value in zip(ensemble_correct, any_candidate_correct)
    )
    return {
        "total_ensemble_samples": total,
        "agreement_count": agreement,
        "agreement_rate": _rate(agreement, total),
        "disagreement_count": disagreement,
        "disagreement_rate": _rate(disagreement, total),
        "accepted_count": accepted,
        "accepted_rate": _rate(accepted, total),
        "accepted_with_warning_count": accepted_with_warning,
        "accepted_with_warning_rate": _rate(accepted_with_warning, total),
        "review_needed_count": review_needed,
        "review_needed_rate": _rate(review_needed, total),
        "rejected_count": rejected,
        "rejected_rate": _rate(rejected, total),
        "candidate_backend_metrics": candidate_backend_metrics,
        "single_model_correct_distribution": pairwise,
        "ensemble_only_correct_count": ensemble_only_correct,
        "ensemble_missed_available_correct_count": ensemble_missed_candidate_correct,
        "improvement_claim": (
            "No accuracy improvement is claimed unless benchmark data shows ensemble_only_correct_count "
            "exceeds ensemble_missed_available_correct_count on a representative dataset."
        ),
    }

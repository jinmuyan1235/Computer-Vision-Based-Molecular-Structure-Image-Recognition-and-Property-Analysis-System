"""OCSR benchmark metric calculations."""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from typing import Any

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

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


def molecule_identity(smiles: str | None) -> tuple[str | None, str | None]:
    """Return canonical SMILES and InChIKey when RDKit can provide them."""
    if not smiles:
        return None, None
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


def enrich_prediction(row: dict[str, Any], similarity_threshold: float) -> dict[str, Any]:
    """Add RDKit validity, identity and similarity fields to one prediction row."""
    predicted_smiles = row.get("predicted_smiles")
    truth_canonical, truth_inchikey = molecule_identity(row.get("ground_truth_smiles"))
    predicted_canonical, predicted_inchikey = molecule_identity(predicted_smiles)
    rdkit_valid = predicted_canonical is not None
    canonical_exact = bool(rdkit_valid and predicted_canonical == truth_canonical)
    equivalent = bool(canonical_exact or (truth_inchikey and predicted_inchikey and truth_inchikey == predicted_inchikey))
    similarity = tanimoto_similarity(row.get("ground_truth_smiles"), predicted_smiles)
    enriched = dict(row)
    enriched.update(
        {
            "ground_truth_canonical_smiles": truth_canonical,
            "predicted_canonical_smiles": predicted_canonical,
            "ground_truth_inchikey": truth_inchikey,
            "predicted_inchikey": predicted_inchikey,
            "rdkit_valid": rdkit_valid,
            "canonical_exact_match": canonical_exact,
            "molecule_equivalent": equivalent,
            "tanimoto_similarity": similarity,
            "similarity_above_threshold": similarity is not None and similarity >= similarity_threshold,
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
        "median_latency_ms": round(float(statistics.median(latencies)), 3),
        "p95_latency_ms": percentile(latencies, 95),
    }


def summarize_rows(rows: list[dict[str, Any]], similarity_threshold: float) -> dict[str, Any]:
    """Summarize benchmark rows into denominator-explicit metrics."""
    total = len(rows)
    recognition_success = sum(bool(row.get("recognition_success")) for row in rows)
    rdkit_valid = sum(bool(row.get("rdkit_valid")) for row in rows)
    canonical_exact = sum(bool(row.get("canonical_exact_match")) for row in rows)
    equivalent = sum(bool(row.get("molecule_equivalent")) for row in rows)
    failed = total - recognition_success
    similarities = [float(row["tanimoto_similarity"]) for row in rows if row.get("tanimoto_similarity") is not None]
    above_threshold = sum(bool(row.get("similarity_above_threshold")) for row in rows)
    failure_reasons = Counter(str(row.get("failure_reason") or "none") for row in rows if row.get("failure_reason"))
    metrics: dict[str, Any] = {
        "total_samples": total,
        "recognition_success_count": recognition_success,
        "recognition_success_rate": _rate(recognition_success, total),
        "rdkit_valid_count": rdkit_valid,
        "rdkit_valid_rate": _rate(rdkit_valid, total),
        "canonical_exact_match_count": canonical_exact,
        "canonical_exact_match_rate": _rate(canonical_exact, total),
        "molecule_equivalent_count": equivalent,
        "molecule_equivalent_rate": _rate(equivalent, total),
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
        },
    }
    metrics.update(_latency_metrics(rows))
    return metrics


def group_metrics(rows: list[dict[str, Any]], similarity_threshold: float) -> dict[str, dict[str, Any]]:
    """Compute category/backend/preprocessing_strategy grouped metrics."""
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {
        "category": defaultdict(list),
        "backend": defaultdict(list),
        "preprocessing_strategy": defaultdict(list),
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
    }

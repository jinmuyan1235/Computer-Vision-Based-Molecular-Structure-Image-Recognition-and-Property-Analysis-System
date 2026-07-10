"""SMILES parsing, validation and canonicalization with RDKit."""

from __future__ import annotations

from typing import Any

from rdkit import Chem


def smiles_to_mol(smiles: str) -> Chem.Mol | None:
    """Parse a SMILES string into an RDKit molecule, returning None if invalid."""
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    try:
        return Chem.MolFromSmiles(smiles.strip())
    except Exception:
        return None


def canonicalize_smiles(smiles: str) -> str | None:
    """Return isomeric canonical SMILES, or None for an invalid input."""
    molecule = smiles_to_mol(smiles)
    if molecule is None:
        return None
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def validate_smiles(smiles: str | None) -> dict[str, Any]:
    """Validate SMILES and return a stable, JSON-friendly result dictionary."""
    if smiles is None or not isinstance(smiles, str) or not smiles.strip():
        return {"valid": False, "canonical_smiles": None, "error": "SMILES 不能为空。"}
    canonical = canonicalize_smiles(smiles)
    if canonical is None:
        return {
            "valid": False,
            "canonical_smiles": None,
            "error": "RDKit 无法解析该 SMILES，请检查原子、键和括号。",
        }
    return {"valid": True, "canonical_smiles": canonical, "error": None}

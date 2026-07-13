"""SMILES parsing, validation and canonicalization with RDKit."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

from rdkit import Chem, rdBase


@contextmanager
def suppress_rdkit_parse_errors():
    """Temporarily suppress expected RDKit SMILES parse errors."""
    rdBase.DisableLog("rdApp.error")
    try:
        yield
    finally:
        rdBase.EnableLog("rdApp.error")


def smiles_to_mol(smiles: str) -> Chem.Mol | None:
    """Parse a SMILES string into an RDKit molecule, returning None if invalid."""
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    try:
        with suppress_rdkit_parse_errors():
            return Chem.MolFromSmiles(smiles.strip())
    except Exception:
        return None


def canonicalize_smiles(smiles: str) -> str | None:
    """Return isomeric canonical SMILES, or None for an invalid input."""
    molecule = smiles_to_mol(smiles)
    if molecule is None or unsupported_structure_reason(molecule):
        return None
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def unsupported_structure_reason(molecule: Chem.Mol | None) -> str | None:
    """Return a reason for structures this app should not analyze as normal molecules."""
    if molecule is None:
        return None
    dummy_atoms = [atom.GetIdx() for atom in molecule.GetAtoms() if atom.GetAtomicNum() == 0]
    if dummy_atoms:
        return "SMILES 含有通配符或查询原子（*），不能作为确定分子进入性质计算。"
    return None


def validate_smiles(smiles: str | None) -> dict[str, Any]:
    """Validate SMILES and return a stable, JSON-friendly result dictionary."""
    if smiles is None or not isinstance(smiles, str) or not smiles.strip():
        return {"valid": False, "canonical_smiles": None, "error": "SMILES 不能为空。"}
    molecule = smiles_to_mol(smiles)
    unsupported = unsupported_structure_reason(molecule)
    if unsupported:
        return {"valid": False, "canonical_smiles": None, "error": unsupported}
    if molecule is None:
        return {
            "valid": False,
            "canonical_smiles": None,
            "error": "RDKit 无法解析该 SMILES，请检查原子、键和括号。",
        }
    canonical = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
    return {"valid": True, "canonical_smiles": canonical, "error": None}

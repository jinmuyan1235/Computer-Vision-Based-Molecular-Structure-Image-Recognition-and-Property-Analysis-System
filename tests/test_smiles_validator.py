"""Tests for SMILES validation and canonicalization."""

from src.chem.smiles_validator import canonicalize_smiles, validate_smiles


def test_valid_smiles() -> None:
    result = validate_smiles("CCO")
    assert result["valid"] is True
    assert result["canonical_smiles"] == "CCO"
    assert result["error"] is None


def test_invalid_smiles() -> None:
    result = validate_smiles("C1(CC")
    assert result["valid"] is False
    assert result["canonical_smiles"] is None
    assert result["error"]


def test_canonicalize_aspirin() -> None:
    assert canonicalize_smiles("CC(=O)OC1=CC=CC=C1C(=O)O") == "CC(=O)Oc1ccccc1C(=O)O"

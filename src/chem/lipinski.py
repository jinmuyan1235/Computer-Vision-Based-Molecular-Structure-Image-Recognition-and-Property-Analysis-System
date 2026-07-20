"""Lipinski and extended rule-based drug-likeness assessment."""

from __future__ import annotations

from typing import Any, Mapping


def evaluate_lipinski(descriptors: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate Lipinski thresholds plus an extended rotatable-bond rule."""
    checks = (
        (float(descriptors["molecular_weight"]) <= 500, "MW > 500"),
        (float(descriptors["logp"]) <= 5, "LogP > 5"),
        (int(descriptors["hbd"]) <= 5, "HBD > 5"),
        (int(descriptors["hba"]) <= 10, "HBA > 10"),
        (int(descriptors["rotatable_bonds"]) <= 10, "Rotatable Bonds > 10"),
    )
    violations = [label for passed, label in checks if not passed]
    passed = not violations
    if passed:
        summary = "该分子基本符合 Lipinski 类药性规则及扩展可旋转键规则。"
    else:
        summary = f"规则超限项：{', '.join(violations)}。"
    return {"passed": passed, "violations": violations, "summary": summary}


def analyze_lipinski(smiles: str) -> dict[str, Any]:
    """Convenience wrapper that calculates descriptors before rule evaluation."""
    from .descriptors import calculate_descriptors

    return evaluate_lipinski(calculate_descriptors(smiles))

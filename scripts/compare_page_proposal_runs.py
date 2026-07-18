"""Apply the predeclared production gate to two page-proposal evaluation runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _load_csv(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return {row["source_document"]: row for row in csv.DictReader(handle)}


def compare(baseline: Path, candidate: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    first = json.loads((baseline / "metrics.json").read_text(encoding="utf-8"))
    second = json.loads((candidate / "metrics.json").read_text(encoding="utf-8"))
    first_docs = _load_csv(baseline / "per_document_metrics.csv")
    second_docs = _load_csv(candidate / "per_document_metrics.csv")
    document_changes = []
    improved_documents = 0; severe_regressions = 0
    for document in sorted(set(first_docs) | set(second_docs)):
        old = first_docs.get(document, {}); new = second_docs.get(document, {})
        recall_delta = float(new.get("recall") or 0) - float(old.get("recall") or 0)
        f1_delta = float(new.get("f1") or 0) - float(old.get("f1") or 0)
        improved_documents += int(f1_delta > 0)
        severe_regressions += int(recall_delta < -0.10)
        document_changes.append({"source_document": document, "recall_delta": recall_delta, "f1_delta": f1_delta})
    checks = {
        "recall_not_lower": second["molecule_proposal_recall"] >= first["molecule_proposal_recall"],
        "precision_not_materially_lower": second["molecule_proposal_precision"] >= first["molecule_proposal_precision"] - 0.02,
        "merged_errors_not_higher": second["merged_region_error_count"] <= first["merged_region_error_count"],
        "missed_molecules_not_higher": second["missed_molecule_count"] <= first["missed_molecule_count"],
        "at_least_two_documents_improve": improved_documents >= 2,
        "no_severe_document_recall_regression": severe_regressions == 0,
        "proposal_count_not_exploded": second["proposal_count"] <= max(first["proposal_count"] * 1.5, first["proposal_count"] + 5),
    }
    result = {
        "baseline": first, "candidate": second, "checks": checks,
        "candidate_proposal_passes_gate": all(checks.values()), "per_document_changes": document_changes,
        "default_recommendation": (
            "proposal=candidate,crop_screening=candidate" if all(checks.values())
            else "proposal=baseline,crop_screening=candidate"
        ),
    }
    (output / "comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Page proposal production gate", "", f"Pass: **{result['candidate_proposal_passes_gate']}**", "", "## Checks", ""]
    lines.extend(f"- {name}: {passed}" for name, passed in checks.items())
    lines.extend(["", f"Recommended default: `{result['default_recommendation']}`", ""])
    (output / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True); parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True); args = parser.parse_args()
    result = compare(Path(args.baseline), Path(args.candidate), Path(args.output))
    print(json.dumps(result, ensure_ascii=False, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())

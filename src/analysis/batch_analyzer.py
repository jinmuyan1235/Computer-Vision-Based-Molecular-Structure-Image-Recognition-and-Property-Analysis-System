"""Fault-tolerant batch processing and summary generation."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from config import OUTPUT_DIR
from src.export.csv_exporter import save_csv
from src.export.json_exporter import save_json
from src.utils.file_utils import ensure_directory, iter_image_files
from .molecule_report import MoleculeReportGenerator


def flatten_report(report: dict[str, Any]) -> dict[str, Any]:
    """Flatten a nested molecule report into one tabular row."""
    input_data = report.get("input") or {}
    ocsr = report.get("ocsr") or {}
    validation = report.get("validation") or {}
    descriptors = report.get("descriptors") or {}
    lipinski = report.get("lipinski") or {}
    return {
        "filename": input_data.get("filename"),
        "status": report.get("status"),
        "message": report.get("message"),
        "backend": ocsr.get("backend"),
        "ocsr_status": ocsr.get("status"),
        "smiles": ocsr.get("smiles"),
        "confidence": ocsr.get("confidence"),
        "valid": bool(validation.get("valid", False)),
        "canonical_smiles": validation.get("canonical_smiles"),
        "formula": descriptors.get("formula"),
        "molecular_weight": descriptors.get("molecular_weight"),
        "logp": descriptors.get("logp"),
        "tpsa": descriptors.get("tpsa"),
        "hbd": descriptors.get("hbd"),
        "hba": descriptors.get("hba"),
        "rotatable_bonds": descriptors.get("rotatable_bonds"),
        "lipinski_passed": lipinski.get("passed"),
        "redrawn_molecule": (report.get("images") or {}).get("redrawn_molecule"),
    }


class BatchAnalyzer:
    """Analyze all supported images in a folder without stopping on bad files."""

    def __init__(self, backend: str | None = None, output_dir: str | Path = OUTPUT_DIR) -> None:
        self.output_dir = ensure_directory(output_dir)
        self.generator = MoleculeReportGenerator(backend=backend, output_dir=self.output_dir)

    def analyze_folder(self, input_dir: str | Path) -> dict[str, Any]:
        """Process a folder, export CSV/JSON, and return results plus statistics."""
        image_files = list(iter_image_files(input_dir))
        reports: list[dict[str, Any]] = []
        for image_path in image_files:
            try:
                reports.append(self.generator.generate(image_path=image_path))
            except Exception as exc:
                reports.append({
                    "status": "failed",
                    "message": f"未预期错误：{exc}",
                    "input": {"type": "image", "filename": image_path.name, "path": str(image_path)},
                })
        rows = [flatten_report(report) for report in reports]
        successful = sum(row["status"] == "success" for row in rows)
        valid = sum(bool(row["valid"]) for row in rows)
        total = len(rows)
        failure_reasons = Counter(row["message"] for row in rows if row["status"] != "success")
        summary = {
            "total": total,
            "successful": successful,
            "failed": total - successful,
            "valid_smiles": valid,
            "success_rate": round(successful / total, 4) if total else 0.0,
            "valid_rate": round(valid / total, 4) if total else 0.0,
            "failure_reasons": dict(failure_reasons),
        }
        csv_path = save_csv(rows, self.output_dir / "batch_results.csv")
        json_path = save_json({"summary": summary, "results": reports}, self.output_dir / "batch_results.json")
        chart_path = self._save_summary_chart(summary)
        return {
            "summary": summary,
            "rows": rows,
            "dataframe": pd.DataFrame(rows),
            "reports": reports,
            "exports": {"csv": csv_path, "json": json_path, "summary_chart": chart_path},
        }

    def _save_summary_chart(self, summary: dict[str, Any]) -> str:
        """Save a small success/validity/failure count chart."""
        path = self.output_dir / "batch_summary.png"
        labels = ["Successful", "Valid SMILES", "Failed"]
        values = [summary["successful"], summary["valid_smiles"], summary["failed"]]
        figure, axis = plt.subplots(figsize=(7, 4))
        bars = axis.bar(labels, values, color=["#2a9d8f", "#457b9d", "#e76f51"])
        axis.set_ylabel("Image count")
        axis.set_title("Batch analysis summary")
        axis.bar_label(bars, padding=3)
        axis.set_ylim(0, max(values + [1]) * 1.2)
        figure.tight_layout()
        figure.savefig(path, dpi=150)
        plt.close(figure)
        return str(path.resolve())

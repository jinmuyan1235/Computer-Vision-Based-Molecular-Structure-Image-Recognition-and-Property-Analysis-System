"""Fault-tolerant batch processing and summary generation."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from config import OUTPUT_DIR
from src.export.csv_exporter import save_csv
from src.export.json_exporter import save_json
from src.utils.file_utils import ensure_directory, iter_image_files
from .molecule_report import MoleculeReportGenerator


def flatten_report(report: dict[str, Any]) -> dict[str, Any]:
    """Flatten a nested molecule report into one tabular row."""
    input_data = report.get("input") or {}
    ocsr = report.get("ocsr") or {}
    correction = report.get("correction") or {}
    final = report.get("final") or {}
    validation = report.get("validation") or {}
    identity = report.get("chemical_identity") or {}
    standardization = report.get("standardization") or {}
    structure_warnings = report.get("structure_warnings") or []
    descriptors = report.get("descriptors") or {}
    lipinski = report.get("lipinski") or {}
    consensus = ocsr.get("consensus") or {}
    candidates = ocsr.get("candidates") or []
    return {
        "filename": input_data.get("filename"),
        "status": report.get("status"),
        "message": report.get("message"),
        "backend": ocsr.get("backend"),
        "ocsr_status": ocsr.get("status"),
        "smiles": ocsr.get("smiles"),
        "predicted_smiles": ocsr.get("predicted_smiles") or ocsr.get("smiles"),
        "predicted_canonical_smiles": ocsr.get("predicted_canonical_smiles"),
        "predicted_standardized_smiles": ocsr.get("predicted_standardized_smiles"),
        "corrected_smiles": correction.get("corrected_smiles"),
        "corrected_canonical_smiles": correction.get("corrected_canonical_smiles"),
        "correction_applied": bool(correction.get("applied", False)),
        "final_smiles": final.get("smiles") or ocsr.get("smiles"),
        "final_raw_smiles": final.get("raw_smiles"),
        "final_standardized_smiles": final.get("standardized_smiles") or validation.get("standardized_smiles"),
        "final_result_source": final.get("source"),
        "confidence": ocsr.get("confidence"),
        "inference_time_ms": ocsr.get("inference_time_ms"),
        "model_name": ocsr.get("model_name"),
        "model_version": ocsr.get("model_version"),
        "device": ocsr.get("device"),
        "candidate_count": len(candidates),
        "consensus_status": consensus.get("status"),
        "recommended_backend": consensus.get("recommended_backend"),
        "ensemble_disagreement": consensus.get("status") == "disagreement",
        "ensemble_candidates": json.dumps(candidates, ensure_ascii=False),
        "valid": bool(validation.get("valid", False)),
        "canonical_smiles": validation.get("canonical_smiles"),
        "standardized_smiles": validation.get("standardized_smiles") or identity.get("standardized_smiles"),
        "inchikey": identity.get("inchikey"),
        "inchi": identity.get("inchi"),
        "identity_formula": identity.get("formula"),
        "formal_charge": identity.get("formal_charge"),
        "fragment_count": identity.get("fragment_count"),
        "stereocenter_count": identity.get("stereocenter_count"),
        "standardization_profile": standardization.get("profile"),
        "standardization_changed": bool(standardization.get("changed", False)),
        "structure_warning_count": len(structure_warnings),
        "structure_warnings": json.dumps(structure_warnings, ensure_ascii=False),
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
        canonical_counts = Counter(row.get("canonical_smiles") for row in rows if row.get("canonical_smiles"))
        inchikey_counts = Counter(row.get("inchikey") for row in rows if row.get("inchikey"))
        standardized_counts = Counter(row.get("standardized_smiles") for row in rows if row.get("standardized_smiles"))
        duplicate_groups = {
            "canonical_smiles": {key: value for key, value in canonical_counts.items() if value > 1},
            "inchikey": {key: value for key, value in inchikey_counts.items() if value > 1},
            "standardized_smiles": {key: value for key, value in standardized_counts.items() if value > 1},
        }
        summary = {
            "total": total,
            "successful": successful,
            "failed": total - successful,
            "valid_smiles": valid,
            "success_rate": round(successful / total, 4) if total else 0.0,
            "valid_rate": round(valid / total, 4) if total else 0.0,
            "failure_reasons": dict(failure_reasons),
            "duplicates": {
                "canonical_duplicate_count": sum(count - 1 for count in canonical_counts.values() if count > 1),
                "inchikey_duplicate_count": sum(count - 1 for count in inchikey_counts.values() if count > 1),
                "standardized_duplicate_count": sum(count - 1 for count in standardized_counts.values() if count > 1),
                "groups": duplicate_groups,
            },
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
        width, height = 700, 400
        margin = 56
        labels = ["识别成功", "有效 SMILES", "识别失败"]
        values = [int(summary["successful"]), int(summary["valid_smiles"]), int(summary["failed"])]
        colors = ["#2a9d8f", "#457b9d", "#e76f51"]
        max_value = max(values + [1])
        image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)
        font = self._load_chart_font(18)
        label_font = self._load_chart_font(15)
        draw.text((margin, 18), "批量处理统计", fill="#222222", font=font)
        draw.line((margin, height - margin, width - margin // 2, height - margin), fill="#333333", width=2)
        draw.line((margin, margin, margin, height - margin), fill="#333333", width=2)
        bar_area_width = width - margin * 2
        bar_width = 92
        gap = (bar_area_width - bar_width * len(values)) // max(len(values) - 1, 1)
        for index, (label, value, color) in enumerate(zip(labels, values, colors)):
            x0 = margin + index * (bar_width + gap)
            bar_height = int((height - margin * 2) * (value / max_value))
            y0 = height - margin - bar_height
            draw.rectangle((x0, y0, x0 + bar_width, height - margin), fill=color)
            draw.text((x0 + 32, max(y0 - 20, margin - 8)), str(value), fill="#222222", font=label_font)
            draw.text((x0, height - margin + 10), label, fill="#222222", font=label_font)
        image.save(path)
        return str(path.resolve())

    @staticmethod
    def _load_chart_font(size: int) -> ImageFont.ImageFont:
        for font_path in (
            Path("C:/Windows/Fonts/msyh.ttc"),
            Path("C:/Windows/Fonts/simhei.ttf"),
            Path("C:/Windows/Fonts/simsun.ttc"),
        ):
            if font_path.is_file():
                return ImageFont.truetype(str(font_path), size=size)
        return ImageFont.load_default()

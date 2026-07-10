"""Optional PDF export that does not affect the main workflow when unavailable."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


def save_pdf(report: Mapping[str, Any], output_path: str | Path) -> dict[str, Any]:
    """Create a simple PDF report, returning a friendly status dictionary."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import cm
        from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
        from reportlab.lib import colors
    except (ImportError, ModuleNotFoundError):
        return {
            "success": False,
            "path": None,
            "message": "未安装 reportlab，已跳过 PDF 导出；JSON/CSV 功能不受影响。",
        }

    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        styles = getSampleStyleSheet()
        story = [Paragraph("Molecule Vision OCSR Report", styles["Title"]), Spacer(1, 0.4 * cm)]
        validation = report.get("validation", {}) or {}
        ocsr = report.get("ocsr", {}) or {}
        descriptors = report.get("descriptors", {}) or {}
        lipinski = report.get("lipinski", {}) or {}
        rows = [
            ["Field", "Value"],
            ["Input", str(report.get("input", {}).get("filename") or report.get("input", {}).get("smiles", ""))],
            ["Backend", str(ocsr.get("backend", "manual"))],
            ["SMILES", str(ocsr.get("smiles") or report.get("input", {}).get("smiles", ""))],
            ["Canonical SMILES", str(validation.get("canonical_smiles", ""))],
            ["Valid", str(validation.get("valid", False))],
        ]
        rows.extend([[str(key), str(value)] for key, value in descriptors.items()])
        rows.append(["Rule passed", str(lipinski.get("passed", ""))])
        table = Table(rows, colWidths=[4.5 * cm, 12 * cm], repeatRows=1)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#24445c")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
        ]))
        story.append(table)
        redrawn = (report.get("images", {}) or {}).get("redrawn_molecule")
        if redrawn and Path(redrawn).is_file():
            story.extend([Spacer(1, 0.5 * cm), Image(redrawn, width=12 * cm, height=9 * cm)])
        SimpleDocTemplate(str(destination), pagesize=A4).build(story)
        return {"success": True, "path": str(destination), "message": "PDF 报告已生成。"}
    except Exception as exc:
        return {"success": False, "path": None, "message": f"PDF 导出失败：{exc}"}

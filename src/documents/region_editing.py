"""Human region-edit helpers with audit records."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from typing import Any

from src.documents.models import normalize_bbox


def _audit(operation: str, before: dict[str, Any] | None, after: dict[str, Any] | None, note: str | None = None) -> dict[str, Any]:
    return {
        "operation": operation,
        "before": before,
        "after": after,
        "note": note,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": "user",
    }


def apply_region_edits(document_result: dict[str, Any], edits: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply add/update/delete/mark edits without mutating the original result."""
    updated = deepcopy(document_result)
    pages_by_number = {int(page["page_number"]): page for page in updated.get("pages", [])}
    regions = list(updated.get("regions", []))

    for edit in edits:
        action = str(edit.get("action", "")).strip().lower()
        region_id = edit.get("region_id")
        if action == "add":
            page_number = int(edit["page_number"])
            page = pages_by_number[page_number]
            bbox = normalize_bbox(edit["bbox"], int(page["width"]), int(page["height"]))
            existing = [region for region in regions if int(region["page_number"]) == page_number]
            new_id = edit.get("new_region_id") or f"p{page_number:03d}_r{len(existing) + 1:03d}_user"
            region = {
                "document_id": updated["document_id"],
                "page_number": page_number,
                "region_id": new_id,
                "bbox": list(bbox),
                "region_type": edit.get("region_type", "molecule"),
                "detection_confidence": None,
                "crop_path": None,
                "source": "user",
                "detector_name": None,
                "status": "edited",
                "message": edit.get("note") or "Added by user.",
                "audit": [_audit("add", None, {"bbox": list(bbox), "region_type": edit.get("region_type", "molecule")}, edit.get("note"))],
                "ocsr": {},
                "final_result": {},
                "report": None,
            }
            regions.append(region)
            continue

        target = next((region for region in regions if region.get("region_id") == region_id), None)
        if target is None:
            raise ValueError(f"Unknown region_id for edit: {region_id}")
        before = {"bbox": target.get("bbox"), "region_type": target.get("region_type"), "status": target.get("status")}
        if action == "delete":
            target["status"] = "deleted"
            target["region_type"] = "non_molecule"
            target.setdefault("audit", []).append(_audit("delete", before, {"status": "deleted"}, edit.get("note")))
        elif action == "mark":
            target["region_type"] = edit.get("region_type", "non_molecule")
            target["status"] = "edited"
            target.setdefault("audit", []).append(
                _audit("mark", before, {"region_type": target["region_type"], "status": "edited"}, edit.get("note"))
            )
        elif action == "update":
            page = pages_by_number[int(target["page_number"])]
            target["bbox"] = list(normalize_bbox(edit.get("bbox", target["bbox"]), int(page["width"]), int(page["height"])))
            if edit.get("region_type"):
                target["region_type"] = edit["region_type"]
            target["status"] = "edited"
            target["crop_path"] = None
            target["ocsr"] = {}
            target["final_result"] = {}
            target["report"] = None
            target.setdefault("audit", []).append(
                _audit("update_bbox", before, {"bbox": target["bbox"], "region_type": target["region_type"]}, edit.get("note"))
            )
        else:
            raise ValueError(f"Unsupported region edit action: {action}")

    updated["regions"] = regions
    updated["summary"] = summarize_regions(regions)
    return updated


def summarize_regions(regions: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize region statuses and types for document-level reporting."""
    active = [region for region in regions if region.get("status") != "deleted"]
    molecule_regions = [region for region in active if region.get("region_type") == "molecule"]
    recognized = [region for region in molecule_regions if (region.get("report") or {}).get("status") == "success"]
    failed = [region for region in molecule_regions if region.get("report") and (region.get("report") or {}).get("status") != "success"]
    by_type: dict[str, int] = {}
    for region in active:
        region_type = str(region.get("region_type") or "unknown")
        by_type[region_type] = by_type.get(region_type, 0) + 1
    return {
        "page_count": None,
        "region_count": len(active),
        "molecule_region_count": len(molecule_regions),
        "recognized_region_count": len(recognized),
        "failed_region_count": len(failed),
        "region_type_distribution": by_type,
    }

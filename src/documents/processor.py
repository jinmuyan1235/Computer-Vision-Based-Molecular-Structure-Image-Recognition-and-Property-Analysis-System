"""Document-level OCSR pipeline: pages, regions, crops, recognition, exports."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4
import shutil
import zipfile

import cv2
import numpy as np
from PIL import Image, ImageDraw

import config
from src.analysis.batch_analyzer import flatten_report
from src.analysis.molecule_report import MoleculeReportGenerator
from src.documents.detectors import BaseMoleculeRegionDetector, HeuristicMoleculeRegionDetector
from src.documents.input_loader import DocumentInputLoader
from src.documents.models import DocumentPage, DocumentRegion, normalize_bbox, relative_path
from src.documents.region_editing import apply_region_edits, summarize_regions
from src.export.csv_exporter import save_csv
from src.export.json_exporter import save_json
from src.utils.file_utils import ensure_directory


class DocumentOCSRProcessor:
    """Run document rendering, region detection, region OCSR, and export."""

    def __init__(
        self,
        backend: str | None = None,
        output_dir: str | Path = config.DOCUMENT_OUTPUT_DIR,
        detector: BaseMoleculeRegionDetector | None = None,
        loader: DocumentInputLoader | None = None,
    ) -> None:
        self.output_dir = ensure_directory(output_dir)
        self.backend = backend
        self.detector = detector or HeuristicMoleculeRegionDetector()
        self.loader = loader or DocumentInputLoader(self.output_dir)
        self.report_generator = MoleculeReportGenerator(backend=backend, output_dir=self.output_dir)

    def process(self, input_path: str | Path, run_ocsr: bool = True) -> dict[str, Any]:
        """Process a PDF, page image, or ZIP image collection."""
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8]
        document_id, pages = self.loader.load(input_path)
        document_dir = ensure_directory(self.output_dir / f"{document_id}_{run_id}")
        pages = self._move_pages_into_run(pages, document_dir)
        regions: list[DocumentRegion] = []
        detection_errors: list[dict[str, Any]] = []
        for page in pages:
            try:
                detected = self.detector.detect(page)
                if len(regions) + len(detected) > config.DOCUMENT_MAX_REGIONS:
                    allowed = max(config.DOCUMENT_MAX_REGIONS - len(regions), 0)
                    detected = detected[:allowed]
                    detection_errors.append({
                        "page_number": page.page_number,
                        "message": "Maximum region count reached; remaining detections were skipped.",
                    })
                regions.extend(detected)
            except Exception as exc:
                detection_errors.append({"page_number": page.page_number, "message": str(exc)})
        if run_ocsr:
            for region in regions:
                self.recognize_region(region, pages, document_dir)
        result = self._result(input_path, document_id, document_dir, pages, regions, detection_errors)
        result["exports"] = self.export(result, document_dir)
        return result

    def recognize_region(
        self,
        region: DocumentRegion | dict[str, Any],
        pages: list[DocumentPage] | list[dict[str, Any]],
        document_dir: str | Path,
    ) -> DocumentRegion | dict[str, Any]:
        """Crop and recognize one molecule region, leaving non-molecule regions untouched."""
        if isinstance(region, dict):
            region_obj = DocumentRegion(
                document_id=region["document_id"],
                page_number=int(region["page_number"]),
                region_id=region["region_id"],
                bbox=tuple(region["bbox"]),
                region_type=region.get("region_type", "unknown"),
                detection_confidence=region.get("detection_confidence"),
                crop_path=region.get("crop_path"),
                source=region.get("source", "detector"),
                detector_name=region.get("detector_name"),
                status=region.get("status", "detected"),
                message=region.get("message"),
                audit=region.get("audit", []),
                ocsr=region.get("ocsr", {}),
                final_result=region.get("final_result", {}),
                report=region.get("report"),
            )
            self.recognize_region(region_obj, pages, document_dir)
            region.update(region_obj.to_dict())
            return region

        if region.status == "deleted":
            return region
        if region.region_type != "molecule":
            region.status = "skipped"
            region.message = region.message or f"Region type {region.region_type} is not sent to single-molecule OCSR."
            return region
        page = self._find_page(pages, region.page_number)
        crop_path = self.crop_region(page, region, document_dir)
        region.crop_path = str(crop_path.resolve())
        try:
            report = self.report_generator.generate(image_path=crop_path)
            report.setdefault("document_region", {})
            report["document_region"].update({
                "document_id": region.document_id,
                "page_number": region.page_number,
                "region_id": region.region_id,
                "bbox": list(region.bbox),
                "region_type": region.region_type,
                "detection_confidence": region.detection_confidence,
            })
            report["input"]["document_id"] = region.document_id
            report["input"]["page_number"] = region.page_number
            report["input"]["region_id"] = region.region_id
            report["input"]["bbox"] = list(region.bbox)
            region.report = report
            region.ocsr = report.get("ocsr") or {}
            region.final_result = report.get("final") or {}
            region.status = "recognized" if report.get("status") == "success" else "failed"
            region.message = report.get("message")
        except Exception as exc:
            region.status = "failed"
            region.message = f"Region OCSR failed: {exc}"
            region.report = {
                "status": "failed",
                "message": region.message,
                "input": {
                    "type": "document_region",
                    "document_id": region.document_id,
                    "page_number": region.page_number,
                    "region_id": region.region_id,
                    "bbox": list(region.bbox),
                    "path": str(crop_path),
                },
            }
        return region

    def apply_edits(self, document_result: dict[str, Any], edits: list[dict[str, Any]], rerun_ocsr: bool = False) -> dict[str, Any]:
        """Apply human bbox/type edits and optionally re-run OCSR on edited molecule regions."""
        updated = apply_region_edits(document_result, edits)
        document_dir = Path(updated["output_dir"])
        if rerun_ocsr:
            for region in updated.get("regions", []):
                if region.get("status") in {"edited", "detected"} and region.get("region_type") == "molecule":
                    self.recognize_region(region, updated.get("pages", []), document_dir)
        updated["summary"] = self._summary(updated.get("pages", []), updated.get("regions", []), updated.get("detection_errors", []))
        updated["exports"] = self.export(updated, document_dir)
        return updated

    def crop_region(self, page: DocumentPage | dict[str, Any], region: DocumentRegion, document_dir: str | Path) -> Path:
        """Crop a region image while preserving page-coordinate bbox metadata."""
        page_path = Path(page.image_path if isinstance(page, DocumentPage) else page["image_path"])
        page_width = int(page.width if isinstance(page, DocumentPage) else page["width"])
        page_height = int(page.height if isinstance(page, DocumentPage) else page["height"])
        region.bbox = normalize_bbox(region.bbox, page_width, page_height)
        image = cv2.imdecode(np.fromfile(str(page_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Unable to decode page image: {page_path}")
        x1, y1, x2, y2 = region.bbox
        crop = image[y1:y2, x1:x2]
        crop_dir = ensure_directory(Path(document_dir) / "crops")
        crop_path = crop_dir / f"{region.document_id}_{region.region_id}.png"
        cv2.imencode(".png", crop)[1].tofile(str(crop_path))
        return crop_path

    def export(self, document_result: dict[str, Any], document_dir: str | Path) -> dict[str, str]:
        """Export JSON, CSV, annotated pages, failures, crops, redrawn images, and a ZIP package."""
        output_root = ensure_directory(document_dir)
        rows = self.region_rows(document_result)
        csv_path = save_csv(rows, output_root / "regions.csv")
        failures = [row for row in rows if row.get("region_type") == "molecule" and row.get("status") not in {"recognized"}]
        failure_path = save_csv(failures, output_root / "failed_regions.csv")
        annotated_paths = self._save_annotated_pages(document_result, output_root)
        redrawn_dir = ensure_directory(output_root / "redrawn")
        for region in document_result.get("regions", []):
            redrawn = (((region.get("report") or {}).get("images") or {}).get("redrawn_molecule"))
            if redrawn and Path(redrawn).is_file():
                shutil.copy2(redrawn, redrawn_dir / f"{region.get('region_id')}_redrawn.png")
        exports = {
            "regions_csv": csv_path,
            "failure_cases_csv": failure_path,
            "annotated_pages_dir": str((output_root / "annotated_pages").resolve()),
            "annotated_pages": ",".join(annotated_paths),
            "crops_dir": str((output_root / "crops").resolve()),
            "redrawn_dir": str(redrawn_dir.resolve()),
        }
        document_result["exports"] = exports
        json_path = save_json(document_result, output_root / "document_result.json")
        exports["json"] = json_path
        zip_path = self._zip_outputs(output_root)
        exports["zip"] = str(zip_path.resolve())
        document_result["exports"] = exports
        save_json(document_result, output_root / "document_result.json")
        return exports

    @staticmethod
    def region_rows(document_result: dict[str, Any]) -> list[dict[str, Any]]:
        """Flatten document regions for CSV export."""
        rows: list[dict[str, Any]] = []
        output_dir = Path(document_result.get("output_dir", "."))
        for region in document_result.get("regions", []):
            report = region.get("report") or {}
            flat = flatten_report(report) if report else {}
            row = {
                "document_id": document_result.get("document_id"),
                "page_number": region.get("page_number"),
                "region_id": region.get("region_id"),
                "bbox_x1": (region.get("bbox") or [None, None, None, None])[0],
                "bbox_y1": (region.get("bbox") or [None, None, None, None])[1],
                "bbox_x2": (region.get("bbox") or [None, None, None, None])[2],
                "bbox_y2": (region.get("bbox") or [None, None, None, None])[3],
                "region_type": region.get("region_type"),
                "detection_confidence": region.get("detection_confidence"),
                "source": region.get("source"),
                "status": region.get("status"),
                "message": region.get("message"),
                "crop_path": relative_path(region.get("crop_path"), output_dir) if region.get("crop_path") else None,
                "audit_count": len(region.get("audit") or []),
            }
            row.update({key: value for key, value in flat.items() if key not in row})
            rows.append(row)
        return rows

    def _result(
        self,
        input_path: str | Path,
        document_id: str,
        document_dir: Path,
        pages: list[DocumentPage],
        regions: list[DocumentRegion],
        detection_errors: list[dict[str, Any]],
    ) -> dict[str, Any]:
        region_dicts = [region.to_dict() for region in regions]
        page_dicts = [page.to_dict() for page in pages]
        return {
            "document_id": document_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "input_path": str(Path(input_path).expanduser().resolve()),
            "output_dir": str(document_dir.resolve()),
            "backend": self.report_generator.recognizer.backend,
            "detector": self.detector.name,
            "pages": page_dicts,
            "regions": region_dicts,
            "detection_errors": detection_errors,
            "summary": self._summary(page_dicts, region_dicts, detection_errors),
            "exports": {},
        }

    @staticmethod
    def _summary(
        pages: list[DocumentPage] | list[dict[str, Any]],
        regions: list[DocumentRegion] | list[dict[str, Any]],
        detection_errors: list[dict[str, Any]],
    ) -> dict[str, Any]:
        region_dicts = [region.to_dict() if isinstance(region, DocumentRegion) else region for region in regions]
        summary = summarize_regions(region_dicts)
        summary["page_count"] = len(pages)
        summary["detection_error_count"] = len(detection_errors)
        return summary

    @staticmethod
    def _find_page(pages: list[DocumentPage] | list[dict[str, Any]], page_number: int) -> DocumentPage | dict[str, Any]:
        for page in pages:
            current = page.page_number if isinstance(page, DocumentPage) else int(page["page_number"])
            if current == page_number:
                return page
        raise ValueError(f"Page not found for region: {page_number}")

    @staticmethod
    def _move_pages_into_run(pages: list[DocumentPage], document_dir: Path) -> list[DocumentPage]:
        page_dir = ensure_directory(document_dir / "pages")
        moved: list[DocumentPage] = []
        for page in pages:
            source = Path(page.image_path)
            destination = page_dir / source.name
            if source.resolve() != destination.resolve():
                shutil.copy2(source, destination)
            page.image_path = str(destination.resolve())
            moved.append(page)
        return moved

    @staticmethod
    def _save_annotated_pages(document_result: dict[str, Any], output_root: Path) -> list[str]:
        annotated_dir = ensure_directory(output_root / "annotated_pages")
        regions_by_page: dict[int, list[dict[str, Any]]] = {}
        for region in document_result.get("regions", []):
            if region.get("status") == "deleted":
                continue
            regions_by_page.setdefault(int(region.get("page_number", 0)), []).append(region)
        paths: list[str] = []
        colors = {
            "molecule": "green",
            "reaction_like": "orange",
            "table": "blue",
            "text": "gray",
            "unknown": "purple",
            "non_molecule": "red",
        }
        for page in document_result.get("pages", []):
            source = Path(page["image_path"])
            with Image.open(source).convert("RGB") as image:
                draw = ImageDraw.Draw(image)
                for region in regions_by_page.get(int(page["page_number"]), []):
                    bbox = [int(value) for value in region.get("bbox", [0, 0, 0, 0])]
                    color = colors.get(region.get("region_type"), "purple")
                    draw.rectangle(bbox, outline=color, width=3)
                    draw.text((bbox[0] + 3, max(0, bbox[1] - 14)), str(region.get("region_id")), fill=color)
                destination = annotated_dir / f"{document_result.get('document_id')}_p{int(page['page_number']):03d}_annotated.png"
                image.save(destination)
                paths.append(str(destination.resolve()))
        return paths

    @staticmethod
    def _zip_outputs(output_root: Path) -> Path:
        zip_path = output_root / "document_results.zip"
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for item in output_root.rglob("*"):
                if item == zip_path or item.is_dir():
                    continue
                archive.write(item, item.relative_to(output_root))
        return zip_path

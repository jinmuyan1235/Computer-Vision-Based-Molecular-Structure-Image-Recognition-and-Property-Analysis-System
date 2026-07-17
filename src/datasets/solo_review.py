"""Single-developer review ledger for the OCSR machine-review human queue."""

from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from PIL import Image
from rdkit import Chem

import config
from src.chem.smiles_validator import validate_smiles
from src.datasets.splits import scaffold_for_smiles
from src.utils.file_utils import ensure_directory, safe_stem


SOLO_STATUSES = ("machine_verified", "human_verified_single", "rejected", "uncertain")
REGION_TYPES = ("molecule", "reaction", "text", "table", "invalid_crop")
REVIEW_SCOPES = {
    "pending_human_review": {"pending_human_review"},
    "machine_verified": {"machine_verified"},
    "pending_machine_review": {"pending_machine_review"},
    "all_reviewable": {"pending_human_review", "machine_verified", "pending_machine_review"},
}
OUTCOME_FIELDS = (
    "sample_id", "image_path", "ground_truth_smiles", "ground_truth_canonical_smiles", "ground_truth_inchikey",
    "expected_action", "category", "source", "split", "scaffold_key", "source_document", "source_license",
    "attribution", "verification_status", "reviewed_at", "reviewer", "original_prediction", "final_smiles",
    "bbox_before", "bbox_after", "correction_types", "review_notes", "original_queue_status",
)
RECHECK_FIELDS = (
    "recheck_id", "sample_id", "created_at", "recheck_status", "completed_at", "source_document",
    "image_path", "category", "source_license", "review_notes_hidden",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [{key: value or "" for key, value in row.items()} for row in csv.DictReader(handle)]


def _write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def _parse_json(raw: str | None, default: Any) -> Any:
    try:
        return json.loads(raw or "")
    except (TypeError, ValueError):
        return default


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


@dataclass(frozen=True)
class SoloReviewResult:
    sample_id: str
    verification_status: str
    audit_path: Path
    image_path: str


class SoloReviewStore:
    """Persist one developer's decisions without modifying source manifests."""

    def __init__(
        self,
        dataset_root: str | Path = config.DATA_DIR / "ocsr_collections",
        *,
        review_root: str | Path = config.DATA_DIR / "review",
    ) -> None:
        self.dataset_root = Path(dataset_root).expanduser().resolve()
        self.review_root = ensure_directory(Path(review_root).expanduser().resolve())
        self.machine_manifest_path = self.review_root / "machine_review_manifest.csv"
        self.queue_path = self.review_root / "human_review_queue.csv"
        self.audit_dir = ensure_directory(self.review_root / "single_reviews")
        self.crop_dir = ensure_directory(self.review_root / "single_review_crops")
        self.recheck_dir = ensure_directory(self.review_root / "rechecks")
        self.recheck_path = self.review_root / "recheck_queue.csv"

    def list_items(
        self,
        *,
        scope: str = "pending_human_review",
        include_reviewed: bool = True,
    ) -> list[dict[str, Any]]:
        """List reviewable samples from the machine manifest, with queue fallback."""
        if scope not in REVIEW_SCOPES:
            raise ValueError(f"Unsupported review scope: {scope}")
        audits = self._audits()
        items: list[dict[str, Any]] = []
        for row in self._source_rows():
            if str(row.get("verification_status") or "") not in REVIEW_SCOPES[scope]:
                continue
            sample_id = str(row.get("sample_id") or "")
            audit = audits.get(sample_id)
            if audit and not include_reviewed:
                continue
            items.append({
                **row,
                "audit": audit or {},
                "effective_status": str((audit or {}).get("verification_status") or row.get("verification_status") or ""),
                "crop_path_abs": self.resolve_dataset_path(row.get("image_path")),
                "page_path_abs": self.resolve_dataset_path(row.get("source_page_path")),
            })
        return sorted(items, key=lambda item: (bool(item.get("audit")), str(item.get("sample_id") or "")))

    def get_item(self, sample_id: str) -> dict[str, Any] | None:
        return next((item for item in self.list_items(scope="all_reviewable") if item.get("sample_id") == sample_id), None)

    def queue_stats(self) -> dict[str, int]:
        """Return source-state totals plus current single-review outcomes."""
        rows = self._source_rows()
        audits = self._audits()
        rejected_ids = {
            str(row.get("sample_id") or "")
            for row in rows
            if str(row.get("verification_status") or "").startswith("rejected_")
        }
        rejected_ids.update(
            sample_id for sample_id, audit in audits.items()
            if audit.get("verification_status") == "rejected"
        )
        return {
            "total": len(rows),
            "pending_human": sum(row.get("verification_status") == "pending_human_review" for row in rows),
            "machine_verified": sum(row.get("verification_status") == "machine_verified" for row in rows),
            "pending_machine": sum(row.get("verification_status") == "pending_machine_review" for row in rows),
            "reviewed": len(audits),
            "rejected": len({sample_id for sample_id in rejected_ids if sample_id}),
        }

    def submit(
        self,
        sample_id: str,
        *,
        verification_status: str,
        final_smiles: str = "",
        bbox_after: list[int] | tuple[int, int, int, int] | None = None,
        region_type: str | None = None,
        review_notes: str = "",
        reviewer: str = "local",
        selected_prediction: str = "",
    ) -> SoloReviewResult:
        """Save a single-person review decision and regenerate outcome CSVs."""
        if verification_status not in SOLO_STATUSES:
            raise ValueError(f"Unsupported single-review status: {verification_status}")
        item = self.get_item(sample_id)
        if item is None:
            raise ValueError(f"Unknown sample_id: {sample_id}")
        original_bbox = self._bbox(item.get("bbox"))
        final_bbox = list(bbox_after) if bbox_after is not None else original_bbox
        if final_bbox and not self._valid_bbox(final_bbox):
            raise ValueError("bbox_after must be [x1, y1, x2, y2] with a positive area.")
        final_region = str(region_type or item.get("machine_category") or item.get("category") or "invalid_crop").lower()
        if final_region not in REGION_TYPES:
            raise ValueError(f"Unsupported region type: {final_region}")
        baseline_smiles = self._prediction_smiles(item, selected_prediction)
        final_smiles = str(final_smiles or baseline_smiles or "").strip()
        structure = self._structure(final_smiles) if verification_status == "human_verified_single" else self._structure("")
        if verification_status == "human_verified_single" and not structure["valid"]:
            raise ValueError("A valid final SMILES is required for human_verified_single.")
        correction_types: list[str] = []
        if structure["canonical_smiles"] and structure["canonical_smiles"] != self._structure(baseline_smiles)["canonical_smiles"]:
            correction_types.append("smiles")
        if final_bbox != original_bbox:
            correction_types.append("bbox")
        if final_region != str(item.get("machine_category") or item.get("category") or ""):
            correction_types.append("region_type")
        image_path = self._review_image(item, final_bbox, original_bbox)
        payload = {
            "sample_id": sample_id,
            "reviewed_at": _now(),
            "reviewer": reviewer.strip() or "local",
            "verification_status": verification_status,
            "original_prediction": self._original_predictions(item),
            "selected_prediction": selected_prediction,
            "final_smiles": final_smiles,
            "final_canonical_smiles": structure["canonical_smiles"],
            "final_inchikey": structure["inchikey"],
            "bbox_before": original_bbox,
            "bbox_after": final_bbox,
            "region_type_before": item.get("machine_category") or item.get("category"),
            "region_type_after": final_region,
            "correction_types": correction_types,
            "review_notes": review_notes,
            "reviewed_image_path": image_path,
            "source_queue_row": {key: value for key, value in item.items() if key not in {"audit", "crop_path_abs", "page_path_abs"}},
        }
        audit_path = self.audit_dir / f"{safe_stem(sample_id)}.json"
        audit_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        self.export_outcomes()
        return SoloReviewResult(sample_id, verification_status, audit_path, image_path)

    def create_recheck_queue(self, proportion: float, *, seed: int | None = None) -> dict[str, Any]:
        """Randomly select completed single reviews for a blinded delayed recheck."""
        if not 0.0 <= proportion <= 1.0:
            raise ValueError("proportion must be between 0 and 1.")
        verified = [audit for audit in self._audits().values() if audit.get("verification_status") == "human_verified_single"]
        randomizer = random.Random(seed)
        randomizer.shuffle(verified)
        count = round(len(verified) * proportion)
        selected = verified[:count]
        existing = {row.get("sample_id"): row for row in _read_csv(self.recheck_path)}
        for audit in selected:
            sample_id = str(audit["sample_id"])
            if sample_id in existing:
                continue
            source = audit.get("source_queue_row") or {}
            existing[sample_id] = {
                "recheck_id": f"recheck_{safe_stem(sample_id)}",
                "sample_id": sample_id,
                "created_at": _now(),
                "recheck_status": "pending",
                "completed_at": "",
                "source_document": source.get("source_document", ""),
                "image_path": audit.get("reviewed_image_path") or source.get("image_path", ""),
                "category": audit.get("region_type_after") or source.get("category", ""),
                "source_license": source.get("source_license", ""),
                "review_notes_hidden": "true",
            }
        rows = sorted(existing.values(), key=lambda row: str(row.get("sample_id") or ""))
        _write_csv(self.recheck_path, rows, RECHECK_FIELDS)
        report = self._write_consistency_report()
        return {"selected": len(selected), "queue_size": len(rows), "recheck_queue": str(self.recheck_path), "consistency_report": str(report)}

    def list_recheck_items(self, *, pending_only: bool = True) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for row in _read_csv(self.recheck_path):
            if pending_only and row.get("recheck_status") != "pending":
                continue
            item = self.get_item(str(row.get("sample_id") or ""))
            if item is None:
                continue
            # Deliberately omit the first review's final answer and notes.
            item["recheck"] = row
            item["audit"] = {}
            item["first_review_hidden"] = True
            items.append(item)
        return items

    def submit_recheck(
        self,
        sample_id: str,
        *,
        verification_status: str,
        final_smiles: str = "",
        bbox_after: list[int] | tuple[int, int, int, int] | None = None,
        region_type: str | None = None,
        review_notes: str = "",
    ) -> Path:
        if verification_status not in {"human_verified_single", "rejected", "uncertain"}:
            raise ValueError("Recheck status must be human_verified_single, rejected, or uncertain.")
        item = next((candidate for candidate in self.list_recheck_items(pending_only=False) if candidate.get("sample_id") == sample_id), None)
        if item is None:
            raise ValueError(f"Sample is not in the recheck queue: {sample_id}")
        original_bbox = self._bbox(item.get("bbox"))
        final_bbox = list(bbox_after) if bbox_after is not None else original_bbox
        if final_bbox and not self._valid_bbox(final_bbox):
            raise ValueError("bbox_after must have a positive area.")
        final_region = str(region_type or item.get("machine_category") or item.get("category") or "invalid_crop").lower()
        structure = self._structure(final_smiles) if verification_status == "human_verified_single" else self._structure("")
        if verification_status == "human_verified_single" and not structure["valid"]:
            raise ValueError("A valid final SMILES is required for a positive recheck.")
        payload = {
            "sample_id": sample_id,
            "reviewed_at": _now(),
            "verification_status": verification_status,
            "final_smiles": final_smiles,
            "final_canonical_smiles": structure["canonical_smiles"],
            "bbox_after": final_bbox,
            "region_type_after": final_region,
            "review_notes": review_notes,
        }
        path = self.recheck_dir / f"{safe_stem(sample_id)}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        rows = _read_csv(self.recheck_path)
        for row in rows:
            if row.get("sample_id") == sample_id:
                row["recheck_status"] = "completed"
                row["completed_at"] = payload["reviewed_at"]
        _write_csv(self.recheck_path, rows, RECHECK_FIELDS)
        self._write_consistency_report()
        return path

    def export_outcomes(self) -> dict[str, Path]:
        outcomes = [self._outcome_row(audit) for audit in self._audits().values()]
        paths = {
            "human_verified_single": self.review_root / "human_verified_single.csv",
            "rejected": self.review_root / "human_rejected.csv",
            "uncertain": self.review_root / "uncertain.csv",
        }
        _write_csv(paths["human_verified_single"], [row for row in outcomes if row["verification_status"] == "human_verified_single"], OUTCOME_FIELDS)
        _write_csv(paths["rejected"], [row for row in outcomes if row["verification_status"] == "rejected"], OUTCOME_FIELDS)
        _write_csv(paths["uncertain"], [row for row in outcomes if row["verification_status"] == "uncertain"], OUTCOME_FIELDS)
        return paths

    def resolve_dataset_path(self, raw_path: str | None) -> str | None:
        if not raw_path:
            return None
        path = Path(str(raw_path)).expanduser()
        if not path.is_absolute():
            path = self.dataset_root / path
        return str(path.resolve()) if path.is_file() else None

    def _source_rows(self) -> list[dict[str, str]]:
        """Prefer the complete machine manifest; keep queue compatibility for old runs."""
        if self.machine_manifest_path.is_file():
            return _read_csv(self.machine_manifest_path)
        return _read_csv(self.queue_path)

    def prediction_redraw(self, sample_id: str, backend: str, smiles: str) -> str | None:
        """Render a prediction lazily for the Streamlit review page."""
        from src.chem.mol_drawer import draw_molecule

        if not self._structure(smiles)["valid"]:
            return None
        path = self.review_root / "prediction_redraws" / f"{safe_stem(sample_id)}_{safe_stem(backend)}.png"
        if not path.is_file():
            try:
                draw_molecule(smiles, path)
            except Exception:
                return None
        return str(path.resolve())

    def _audits(self) -> dict[str, dict[str, Any]]:
        audits: dict[str, dict[str, Any]] = {}
        for path in self.audit_dir.glob("*.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            sample_id = str(payload.get("sample_id") or "")
            if sample_id:
                audits[sample_id] = payload
        return audits

    @staticmethod
    def _bbox(raw_bbox: str | list[int] | tuple[int, ...] | None) -> list[int]:
        value = _parse_json(raw_bbox, []) if isinstance(raw_bbox, str) else raw_bbox
        if not isinstance(value, (list, tuple)) or len(value) != 4:
            return []
        try:
            return [int(number) for number in value]
        except (TypeError, ValueError):
            return []

    @staticmethod
    def _valid_bbox(bbox: list[int]) -> bool:
        return len(bbox) == 4 and bbox[0] >= 0 and bbox[1] >= 0 and bbox[2] > bbox[0] and bbox[3] > bbox[1]

    @staticmethod
    def _structure(smiles: str) -> dict[str, Any]:
        validation = validate_smiles(smiles)
        canonical = str(validation.get("canonical_smiles") or "")
        if not validation.get("valid"):
            return {"valid": False, "canonical_smiles": "", "inchikey": ""}
        molecule = Chem.MolFromSmiles(canonical)
        return {
            "valid": molecule is not None,
            "canonical_smiles": canonical,
            "inchikey": Chem.MolToInchiKey(molecule) if molecule is not None else "",
        }

    @staticmethod
    def _prediction_smiles(item: dict[str, Any], selected: str) -> str:
        selected = selected.lower().strip()
        if selected in {"molscribe", "decimer", "ensemble"}:
            return str(item.get(f"{selected}_smiles") or "")
        return str(item.get("ensemble_smiles") or item.get("molscribe_smiles") or item.get("decimer_smiles") or "")

    @staticmethod
    def _original_predictions(item: dict[str, Any]) -> dict[str, Any]:
        return {
            backend: _parse_json(item.get(f"{backend}_raw"), {"smiles": item.get(f"{backend}_smiles") or ""})
            for backend in ("molscribe", "decimer", "ensemble")
        }

    def _review_image(self, item: dict[str, Any], bbox_after: list[int], bbox_before: list[int]) -> str:
        if bbox_after != bbox_before and item.get("page_path_abs") and bbox_after:
            page = Path(str(item["page_path_abs"]))
            try:
                with Image.open(page) as image:
                    x1, y1, x2, y2 = bbox_after
                    if x2 > image.width or y2 > image.height:
                        raise ValueError("bbox_after exceeds the source page.")
                    output = self.crop_dir / f"{safe_stem(str(item['sample_id']))}.png"
                    image.crop((x1, y1, x2, y2)).convert("RGB").save(output)
                    return str(output.resolve())
            except Exception as exc:
                raise ValueError(f"Could not create corrected crop: {exc}") from exc
        return str(item.get("crop_path_abs") or "")

    def _outcome_row(self, audit: dict[str, Any]) -> dict[str, Any]:
        source = audit.get("source_queue_row") or {}
        final_smiles = str(audit.get("final_smiles") or "")
        structure = self._structure(final_smiles)
        return {
            "sample_id": audit.get("sample_id", ""),
            "image_path": audit.get("reviewed_image_path", ""),
            "ground_truth_smiles": final_smiles if audit.get("verification_status") == "human_verified_single" else "",
            "ground_truth_canonical_smiles": audit.get("final_canonical_smiles") or structure["canonical_smiles"],
            "ground_truth_inchikey": audit.get("final_inchikey") or structure["inchikey"],
            "expected_action": "recognize" if audit.get("verification_status") == "human_verified_single" else "reject",
            "category": audit.get("region_type_after") or source.get("machine_category") or source.get("category", ""),
            "source": source.get("source_kind", ""),
            "split": source.get("split", ""),
            "scaffold_key": scaffold_for_smiles(audit.get("final_canonical_smiles") or ""),
            "source_document": source.get("source_document", ""),
            "source_license": source.get("source_license", ""),
            "attribution": source.get("attribution", ""),
            "verification_status": audit.get("verification_status", ""),
            "reviewed_at": audit.get("reviewed_at", ""),
            "reviewer": audit.get("reviewer", ""),
            "original_prediction": _json(audit.get("original_prediction") or {}),
            "final_smiles": final_smiles,
            "bbox_before": _json(audit.get("bbox_before") or []),
            "bbox_after": _json(audit.get("bbox_after") or []),
            "correction_types": _json(audit.get("correction_types") or []),
            "review_notes": audit.get("review_notes", ""),
            "original_queue_status": source.get("verification_status", ""),
        }

    def _write_consistency_report(self) -> Path:
        completed: list[dict[str, Any]] = []
        for path in self.recheck_dir.glob("*.json"):
            try:
                completed.append(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                continue
        audits = self._audits()
        comparisons: list[dict[str, Any]] = []
        for recheck in completed:
            first = audits.get(str(recheck.get("sample_id") or ""))
            if not first:
                continue
            same_status = first.get("verification_status") == recheck.get("verification_status")
            same_smiles = str(first.get("final_canonical_smiles") or "") == str(recheck.get("final_canonical_smiles") or "")
            same_bbox = list(first.get("bbox_after") or []) == list(recheck.get("bbox_after") or [])
            same_region = first.get("region_type_after") == recheck.get("region_type_after")
            comparisons.append({
                "sample_id": recheck.get("sample_id"),
                "same_status": same_status,
                "same_smiles": same_smiles,
                "same_bbox": same_bbox,
                "same_region_type": same_region,
                "consistent": same_status and same_smiles and same_bbox and same_region,
            })
        total = len(comparisons)
        report = {
            "completed_rechecks": total,
            "consistent_rechecks": sum(item["consistent"] for item in comparisons),
            "consistency_rate": (sum(item["consistent"] for item in comparisons) / total) if total else None,
            "comparisons": comparisons,
        }
        path = self.review_root / "review_consistency_report.json"
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

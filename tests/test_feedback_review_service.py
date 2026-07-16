from __future__ import annotations

import json
from pathlib import Path

from src.analysis.correction import apply_smiles_correction, save_correction_feedback
from src.analysis.molecule_report import MoleculeReportGenerator
from src.feedback.review_service import FeedbackReviewService
from src.feedback.store import read_feedback_manifest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _corrected_report(tmp_path: Path, analysis_id: str, corrected_smiles: str = "CCN") -> dict:
    sample = PROJECT_ROOT / "data" / "samples" / "aspirin.png"
    report = MoleculeReportGenerator("manual", tmp_path).generate(smiles="CCO", analysis_id=analysis_id)
    report["input"].update({
        "type": "image",
        "filename": f"{analysis_id}.png",
        "path": str(sample),
        "image_sha256": f"sha-{analysis_id}",
    })
    return apply_smiles_correction(report, corrected_smiles, tmp_path)


def _save_pending(tmp_path: Path, analysis_id: str, corrected_smiles: str = "CCN") -> dict:
    report = _corrected_report(tmp_path, analysis_id, corrected_smiles)
    return save_correction_feedback(
        report,
        tmp_path,
        correction_type="atom",
        review_status="pending",
        feedback_action="correction_only",
        include_in_training=False,
        source_reference="unit-test",
        source_license="internal",
        notes="needs review",
    )


def test_pending_feedback_requires_review_before_training_export(tmp_path: Path) -> None:
    _save_pending(tmp_path, "review001")
    service = FeedbackReviewService(tmp_path)

    pending = service.list_items("pending")
    assert [item["analysis_id"] for item in pending] == ["review001"]
    assert pending[0]["corrected_smiles"] == "CCN"
    assert pending[0]["predicted_smiles"] == "CCO"

    manifest_path = tmp_path / "verified_manifest.csv"
    before = service.export_verified_manifest(manifest_path)
    assert before["exported_count"] == 0

    approved = service.approve_for_dataset("review001", "looks correct")
    assert approved["review_status"] == "verified"
    assert approved["include_in_training"] is True

    rows = read_feedback_manifest(tmp_path / "feedback")
    assert rows[0]["review_status"] == "verified"
    assert rows[0]["feedback_action"] == "accepted_for_dataset"
    assert rows[0]["include_in_training"] == "True"
    annotation = json.loads(Path(tmp_path / "feedback" / rows[0]["annotation_path"]).read_text(encoding="utf-8"))
    assert annotation["feedback"]["reviewer_notes"] == "looks correct"

    after = service.export_verified_manifest(manifest_path)
    assert after["exported_count"] == 1


def test_review_service_non_approval_actions_do_not_enter_training(tmp_path: Path) -> None:
    _save_pending(tmp_path, "return001", "CCN")
    _save_pending(tmp_path, "reject001", "CCC")
    _save_pending(tmp_path, "dupe001", "CCCl")
    _save_pending(tmp_path, "license001", "CCBr")
    service = FeedbackReviewService(tmp_path)

    assert service.return_for_revision("return001", "fix source")["review_status"] == "returned"
    assert service.reject_sample("reject001", "not a molecule")["review_status"] == "rejected"
    assert service.mark_duplicate("dupe001", duplicate_of="return001")["review_status"] == "duplicate"
    assert service.mark_license_unclear("license001")["review_status"] == "license_unclear"

    all_items = {item["analysis_id"]: item for item in service.list_items("all")}
    assert all_items["dupe001"]["feedback"]["duplicate_of"] == "return001"
    assert all_items["license001"]["feedback"]["source_license"] == "unclear"
    assert service.export_verified_manifest(tmp_path / "verified_manifest.csv")["exported_count"] == 0

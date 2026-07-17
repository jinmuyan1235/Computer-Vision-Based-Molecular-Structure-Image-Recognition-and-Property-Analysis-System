"""Tests for the single-developer human review ledger and delayed recheck flow."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw

from src.datasets.solo_review import SoloReviewStore


QUEUE_FIELDS = (
    "sample_id", "verification_status", "image_path", "source_page_path", "bbox", "machine_category", "category",
    "source_kind", "source_document", "source_license", "source_url", "attribution", "expected_action", "split",
    "molscribe_smiles", "decimer_smiles", "ensemble_smiles", "molscribe_raw", "decimer_raw", "ensemble_raw",
    "risk_reasons", "image_quality_score", "image_quality_level",
)


def _page(path: Path) -> None:
    image = Image.new("RGB", (180, 140), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 20, 100, 100), outline="black", width=4)
    draw.line((20, 60, 100, 60), fill="black", width=3)
    image.save(path)


def _queue_row(sample_id: str, image_name: str) -> dict[str, str]:
    raw = json.dumps({"status": "success", "smiles": "CCO"})
    return {
        "sample_id": sample_id,
        "verification_status": "pending_human_review",
        "image_path": f"images/{image_name}",
        "source_page_path": "pages/page.png",
        "bbox": "[20, 20, 100, 100]",
        "machine_category": "molecule",
        "category": "molecule",
        "source_kind": "pmc_oa",
        "source_document": "PMC-test",
        "source_license": "cc-by-4.0",
        "source_url": "https://example.test/source",
        "attribution": "Example attribution",
        "expected_action": "recognize",
        "split": "train",
        "molscribe_smiles": "CCO",
        "decimer_smiles": "CCO",
        "ensemble_smiles": "CCO",
        "molscribe_raw": raw,
        "decimer_raw": raw,
        "ensemble_raw": raw,
        "risk_reasons": "[\"model_disagreement\"]",
        "image_quality_score": "0.77",
        "image_quality_level": "medium",
    }


def _setup(tmp_path: Path, sample_ids: tuple[str, ...] = ("sample-1",)) -> tuple[Path, Path, SoloReviewStore]:
    dataset = tmp_path / "dataset"
    review = tmp_path / "review"
    (dataset / "images").mkdir(parents=True)
    (dataset / "pages").mkdir(parents=True)
    _page(dataset / "pages" / "page.png")
    rows = []
    for index, sample_id in enumerate(sample_ids):
        image = dataset / "images" / f"crop-{index}.png"
        _page(image)
        rows.append(_queue_row(sample_id, image.name))
    review.mkdir()
    with (review / "human_review_queue.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=QUEUE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return dataset, review, SoloReviewStore(dataset, review_root=review)


def test_single_review_saves_audit_corrected_crop_and_evaluation_manifest(tmp_path: Path) -> None:
    _, review, store = _setup(tmp_path)

    result = store.submit(
        "sample-1",
        verification_status="human_verified_single",
        final_smiles="CCN",
        bbox_after=[30, 25, 110, 105],
        region_type="molecule",
        review_notes="Corrected one atom and crop.",
        reviewer="developer",
        selected_prediction="ensemble",
    )

    audit = json.loads(result.audit_path.read_text(encoding="utf-8"))
    assert audit["verification_status"] == "human_verified_single"
    assert audit["bbox_before"] == [20, 20, 100, 100]
    assert set(audit["correction_types"]) == {"smiles", "bbox"}
    assert Path(result.image_path).is_file()
    output = list(csv.DictReader((review / "human_verified_single.csv").open(encoding="utf-8")))
    assert output[0]["ground_truth_canonical_smiles"] == "CCN"
    assert output[0]["review_notes"] == "Corrected one atom and crop."


def test_rejected_and_uncertain_have_separate_outputs(tmp_path: Path) -> None:
    _, review, store = _setup(tmp_path, ("reject", "uncertain"))

    store.submit("reject", verification_status="rejected", review_notes="Not a molecule.")
    store.submit("uncertain", verification_status="uncertain", review_notes="Crop is ambiguous.")

    rejected = list(csv.DictReader((review / "human_rejected.csv").open(encoding="utf-8")))
    uncertain = list(csv.DictReader((review / "uncertain.csv").open(encoding="utf-8")))
    assert rejected[0]["sample_id"] == "reject"
    assert uncertain[0]["sample_id"] == "uncertain"
    assert rejected[0]["ground_truth_smiles"] == ""


def test_delayed_recheck_hides_first_answer_and_writes_consistency_report(tmp_path: Path) -> None:
    _, review, store = _setup(tmp_path)
    store.submit("sample-1", verification_status="human_verified_single", final_smiles="CCO", review_notes="First answer.")

    selected = store.create_recheck_queue(1.0, seed=7)
    recheck_items = store.list_recheck_items()

    assert selected["selected"] == 1
    assert recheck_items[0]["first_review_hidden"] is True
    assert recheck_items[0]["audit"] == {}
    store.submit_recheck("sample-1", verification_status="human_verified_single", final_smiles="CCO", bbox_after=[20, 20, 100, 100], region_type="molecule")
    report = json.loads((review / "review_consistency_report.json").read_text(encoding="utf-8"))
    assert report["completed_rechecks"] == 1
    assert report["consistency_rate"] == 1.0

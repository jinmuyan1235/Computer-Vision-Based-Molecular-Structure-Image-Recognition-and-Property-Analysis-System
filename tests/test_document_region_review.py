"""Tests for the interactive document-region review state boundary."""

from __future__ import annotations

import json
from pathlib import Path

from src.documents.processor import DocumentOCSRProcessor
from src.documents.region_review import (
    apply_canvas_event,
    background_failure_reason,
    canvas_bbox_to_page,
    canvas_event_from_query,
    persist_document_result_atomic,
    save_region_selection,
)


def _document(tmp_path: Path) -> dict:
    return {
        "document_id": "doc-1",
        "output_dir": str(tmp_path),
        "pages": [{"page_number": 1, "width": 2000, "height": 1000, "image_path": str(tmp_path / "page.png")}],
        "regions": [
            {
                "document_id": "doc-1",
                "page_number": 1,
                "region_id": "p001_r001",
                "bbox": [200, 100, 1000, 500],
                "region_type": "molecule",
                "status": "recognized",
                "confirmed": True,
                "annotation_status": "confirmed",
                "audit": [],
                "ocsr": {"smiles": "CCO"},
                "final_result": {"smiles": "CCO"},
                "report": {"status": "success"},
            },
            {
                "document_id": "doc-1",
                "page_number": 1,
                "region_id": "p001_r002",
                "bbox": [1200, 100, 1800, 500],
                "region_type": "molecule",
                "status": "confirmed",
                "confirmed": True,
                "annotation_status": "confirmed",
                "audit": [],
                "ocsr": {},
                "final_result": {},
                "report": None,
            },
        ],
        "summary": {"page_count": 1, "region_count": 2},
        "exports": {},
    }


def test_canvas_bbox_maps_to_original_page_coordinates() -> None:
    assert canvas_bbox_to_page([50, 25, 250, 125], 500, 250, 2000, 1000) == [200, 100, 1000, 500]

    event = canvas_event_from_query(
        {
            "doc_bbox_action": "update",
            "doc_bbox_region_id": "p001_r001",
            "doc_bbox_x1": "50",
            "doc_bbox_y1": "25",
            "doc_bbox_x2": "250",
            "doc_bbox_y2": "125",
            "doc_canvas_width": "500",
            "doc_canvas_height": "250",
            "doc_bbox_nonce": "n1",
        },
        {"page_number": 1, "width": 2000, "height": 1000},
    )
    assert event == {
        "action": "update",
        "region_id": "p001_r001",
        "page_number": 1,
        "nonce": "n1",
        "bbox": [200, 100, 1000, 500],
    }


def test_canvas_create_move_delete_and_atomic_state_save(tmp_path: Path) -> None:
    result = _document(tmp_path)
    created, selected_id = apply_canvas_event(result, {"action": "create", "page_number": 1, "bbox": [50, 60, 400, 300]})
    assert selected_id
    created_region = next(region for region in created["regions"] if region["region_id"] == selected_id)
    assert created_region["region_type"] == "molecule"
    assert created_region["confirmed"] is False

    moved, moved_id = apply_canvas_event(
        created,
        {"action": "update", "region_id": selected_id, "page_number": 1, "bbox": [70, 80, 450, 330]},
    )
    moved_region = next(region for region in moved["regions"] if region["region_id"] == selected_id)
    assert moved_id == selected_id
    assert moved_region["bbox"] == [70, 80, 450, 330]
    assert moved_region["confirmed"] is False

    target = persist_document_result_atomic(moved)
    saved = json.loads(target.read_text(encoding="utf-8"))
    assert next(region for region in saved["regions"] if region["region_id"] == selected_id)["bbox"] == [70, 80, 450, 330]
    assert saved["exports"]["json"] == str(target)

    deleted, selection = apply_canvas_event(moved, {"action": "delete", "region_id": selected_id, "page_number": 1})
    assert selection is None
    assert next(region for region in deleted["regions"] if region["region_id"] == selected_id)["status"] == "deleted"


def test_save_and_recognize_state_is_atomic_and_model_ready(tmp_path: Path) -> None:
    result = _document(tmp_path)
    saved = save_region_selection(result, "p001_r001", [220, 120, 980, 480], recognize=True)
    selected = next(region for region in saved["regions"] if region["region_id"] == "p001_r001")

    assert result["regions"][0]["bbox"] == [200, 100, 1000, 500]
    assert selected["bbox"] == [220, 120, 980, 480]
    assert selected["region_type"] == "molecule"
    assert selected["confirmed"] is True
    assert selected["annotation_status"] == "confirmed"
    assert selected["report"] is None
    target = persist_document_result_atomic(saved)
    on_disk = json.loads(target.read_text(encoding="utf-8"))
    assert on_disk["regions"][0]["bbox"] == [220, 120, 980, 480]


def test_background_recognition_only_runs_requested_region(tmp_path: Path) -> None:
    processor = object.__new__(DocumentOCSRProcessor)
    calls: list[str] = []

    def recognize(region: dict, _pages: list[dict], _output: Path) -> None:
        calls.append(str(region["region_id"]))
        region["status"] = "recognized"

    processor.recognize_region = recognize  # type: ignore[method-assign]
    processor._summary = lambda pages, regions, errors: {"page_count": len(pages), "region_count": len(regions)}  # type: ignore[method-assign]
    processor.export = lambda updated, output: updated.get("exports", {})  # type: ignore[method-assign]

    updated = processor.apply_edits(
        _document(tmp_path),
        [{"action": "recognize", "region_id": "p001_r001"}],
        rerun_ocsr=True,
    )

    assert calls == ["p001_r001"]
    assert updated["regions"][1]["status"] == "confirmed"


def test_background_recognition_can_run_all_confirmed_regions(tmp_path: Path) -> None:
    processor = object.__new__(DocumentOCSRProcessor)
    calls: list[str] = []

    def recognize(region: dict, _pages: list[dict], _output: Path) -> None:
        calls.append(str(region["region_id"]))
        region["status"] = "recognized"

    processor.recognize_region = recognize  # type: ignore[method-assign]
    processor._summary = lambda pages, regions, errors: {"page_count": len(pages), "region_count": len(regions)}  # type: ignore[method-assign]
    processor.export = lambda updated, output: updated.get("exports", {})  # type: ignore[method-assign]

    processor.apply_edits(
        _document(tmp_path),
        [
            {"action": "recognize", "region_id": "p001_r001"},
            {"action": "recognize", "region_id": "p001_r002"},
        ],
        rerun_ocsr=True,
    )

    assert calls == ["p001_r001", "p001_r002"]


def test_background_failure_reason_prefers_explicit_worker_message() -> None:
    assert background_failure_reason(1, {"message": "CUDA 显存不足"}, "", "trace") == "CUDA 显存不足"
    assert background_failure_reason(2, None, "", "line one\n模型文件缺失") == "模型文件缺失"
    assert background_failure_reason(9, None, "", "") == "后台进程退出码 9"

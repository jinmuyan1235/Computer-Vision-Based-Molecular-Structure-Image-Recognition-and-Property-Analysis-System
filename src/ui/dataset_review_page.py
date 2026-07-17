"""Streamlit workspace for visual OCSR review and trusted-label confirmation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw
import streamlit as st

import config
from src.datasets.solo_review import REGION_TYPES, VISUAL_REVIEW_STATUSES, SoloReviewStore


def render_dataset_review_page() -> None:
    """Render the review page without treating model output as chemical truth."""
    st.header("OCSR Data Review")
    store = SoloReviewStore(config.DATA_DIR / "ocsr_collections", review_root=config.DATA_DIR / "review")
    mode = st.segmented_control("Review mode", ["Queue", "Delayed recheck"], default="Queue", key="solo_review_mode")
    if mode == "Delayed recheck":
        _render_recheck_workspace(store)
    else:
        _render_queue_workspace(store)


def _render_queue_workspace(store: SoloReviewStore) -> None:
    stats = store.queue_stats()
    metrics = st.columns(6)
    for column, label, key in zip(metrics, ("Total", "Pending human", "Machine verified", "Pending machine", "Reviewed", "Rejected"), ("total", "pending_human", "machine_verified", "pending_machine", "reviewed", "rejected")):
        column.metric(label, stats[key])
    controls = st.columns([0.36, 0.22, 0.42])
    scope_label = controls[0].selectbox("Review range", ["Pending human review", "Machine verified", "Pending machine review", "All reviewable"], index=0, key="solo_review_scope")
    scope = {
        "Pending human review": "pending_human_review", "Machine verified": "machine_verified",
        "Pending machine review": "pending_machine_review", "All reviewable": "all_reviewable",
    }[scope_label]
    show_reviewed = controls[1].toggle("Show reviewed", value=False, key="solo_show_reviewed")
    items = store.list_items(scope=scope, include_reviewed=show_reviewed)
    if not items:
        st.info("No matching reviewable samples in data/review/machine_review_manifest.csv.")
        return
    ids = [str(item["sample_id"]) for item in items]
    selected = controls[2].selectbox("Sample", ids, key="solo_queue_sample")
    _render_item(store, next(item for item in items if item["sample_id"] == selected), recheck=False)


def _render_recheck_workspace(store: SoloReviewStore) -> None:
    controls = st.columns([0.20, 0.16, 0.16, 0.48])
    proportion = controls[0].number_input("Recheck ratio", min_value=0.0, max_value=1.0, value=0.2, step=0.05)
    seed = controls[1].number_input("Seed", min_value=0, value=7, step=1)
    if controls[2].button("Create recheck queue", type="secondary"):
        result = store.create_recheck_queue(float(proportion), seed=int(seed))
        st.success(f"Selected {result['selected']} samples")
        st.rerun()
    items = store.list_recheck_items()
    controls[3].metric("Pending rechecks", len(items))
    if not items:
        st.info("No pending delayed rechecks.")
        return
    selected = st.selectbox("Recheck sample", [str(item["sample_id"]) for item in items], key="solo_recheck_sample")
    _render_item(store, next(item for item in items if item["sample_id"] == selected), recheck=True)


def _render_item(store: SoloReviewStore, item: dict[str, Any], *, recheck: bool) -> None:
    st.subheader(str(item.get("sample_id") or "Sample"))
    status_columns = st.columns(4)
    status_columns[0].metric("Queue status", item.get("verification_status") or "-")
    status_columns[1].metric("Visual review", item.get("audit", {}).get("visual_review_status") or "Not reviewed")
    status_columns[2].metric("Quality", item.get("image_quality_level") or "-")
    status_columns[3].metric("Machine category", item.get("machine_category") or item.get("category") or "-")
    _render_file_status(item)
    _render_images(item)
    _render_predictions(store, item)
    _render_details(item)
    if not recheck and item.get("audit"):
        with st.expander("Latest review audit", expanded=False):
            st.json(item["audit"])
    _render_actions(store, item, recheck=recheck)


def _render_file_status(item: dict[str, Any]) -> None:
    st.caption("Source file resolution")
    lines: list[str] = []
    for label, info in (("Original document page", item.get("page_path_info", {})), ("Candidate image", item.get("image_path_info", {})), ("Candidate crop", item.get("crop_path_info", {}))):
        lines.extend((
            f"{label}: {'exists' if info.get('exists') else 'missing'}",
            f"  manifest path: {info.get('manifest_path', '')}",
            f"  dataset root: {info.get('dataset_root', item.get('dataset_root_resolved', ''))}",
            f"  resolved path: {info.get('resolved_path', '')}",
        ))
    # Avoid Streamlit's Arrow serialization here: it has crashed in this WSL runtime.
    st.code("\n".join(lines), language=None)
    if not item.get("files_complete"):
        st.warning("A required source file is missing. Visual status is suggested as missing_source_file and ground-truth acceptance is disabled.")


def _render_images(item: dict[str, Any]) -> None:
    columns = st.columns(3)
    page = item.get("page_path_abs")
    bbox = _parse_json(item.get("bbox"), [])
    with columns[0]:
        st.caption("Original document page with bbox")
        if page:
            st.image(_page_with_bbox(Path(str(page)), bbox), use_container_width=True)
        else:
            st.info("Original document page is unavailable.")
    with columns[1]:
        st.caption("Candidate crop from source")
        if item.get("crop_path_abs"):
            st.image(str(item["crop_path_abs"]), use_container_width=True)
        else:
            st.info("Candidate crop is unavailable.")
    with columns[2]:
        st.caption("BBox preview cut from original page")
        preview = _crop_preview(page, bbox)
        if preview is not None:
            st.image(preview, use_container_width=True)
        else:
            st.info("No bbox preview can be generated from the original page.")


def _render_predictions(store: SoloReviewStore, item: dict[str, Any]) -> None:
    st.caption("Model predictions and model redraws")
    columns = st.columns(3)
    for column, backend in zip(columns, ("molscribe", "decimer", "ensemble")):
        with column:
            st.caption(backend.title())
            smiles = str(item.get(f"{backend}_smiles") or "")
            st.code(smiles or "No valid SMILES", language=None)
            if item.get("trusted_ground_truth_available"):
                st.caption("Matches trusted ground truth: " + ("yes" if item.get(f"{backend}_matches_ground_truth") else "no"))
            redraw = store.prediction_redraw(str(item.get("sample_id") or "sample"), backend, smiles)
            if redraw:
                st.image(redraw, caption="Model redraw, not the original image", use_container_width=True)
            else:
                st.info("No redraw")
            with st.expander("Raw prediction", expanded=False):
                st.json(_parse_json(item.get(f"{backend}_raw"), {"smiles": smiles}))


def _render_details(item: dict[str, Any]) -> None:
    details = st.columns(3)
    with details[0]:
        st.caption("Risks")
        st.json(_parse_json(item.get("risk_reasons"), []))
    with details[1]:
        st.caption("Deterministic checks")
        st.json(_parse_json(item.get("deterministic_errors"), []))
        failures = _model_failures(item)
        if failures:
            st.caption("Model failures")
            st.json(failures)
    with details[2]:
        st.caption("Source and license")
        st.json({
            "source_document": item.get("source_document"), "source_url": item.get("source_url"),
            "source_license": item.get("source_license"), "attribution": item.get("attribution"),
            "bbox": _parse_json(item.get("bbox"), []),
        })


def _render_actions(store: SoloReviewStore, item: dict[str, Any], *, recheck: bool) -> None:
    sample_id = str(item.get("sample_id") or "")
    st.divider()
    st.subheader("Visual Review")
    bbox_before = _parse_json(item.get("bbox"), [])
    bbox_values = bbox_before if len(bbox_before) == 4 else [0, 0, 0, 0]
    suggested = "missing_source_file" if not item.get("files_complete") else _current_visual_status(item)
    visual_status = st.selectbox("Visual result", VISUAL_REVIEW_STATUSES, index=VISUAL_REVIEW_STATUSES.index(suggested), key=f"visual_status_{'recheck_' if recheck else ''}{sample_id}")
    editor = st.columns([0.24, 0.20, 0.56])
    region_type = editor[0].selectbox("Region type", REGION_TYPES, index=_region_index(item), key=f"solo_region_{'recheck_' if recheck else ''}{sample_id}")
    reviewer = editor[1].text_input("Reviewer", value="local", disabled=recheck, key=f"solo_reviewer_{sample_id}")
    notes = editor[2].text_input("Review notes", key=f"solo_notes_{'recheck_' if recheck else ''}{sample_id}")
    bbox_columns = st.columns(4)
    bbox_after = [int(bbox_columns[index].number_input(label, min_value=0, value=int(bbox_values[index]), step=1, key=f"solo_bbox_{'recheck_' if recheck else ''}{sample_id}_{label}")) for index, label in enumerate(("x1", "y1", "x2", "y2"))]
    if st.button("Save visual review", type="primary", key=f"save_visual_{'recheck_' if recheck else ''}{sample_id}"):
        try:
            if recheck:
                store.submit_recheck(sample_id, visual_review_status=visual_status, bbox_after=bbox_after, region_type=region_type, review_notes=notes)
            else:
                store.submit_visual(sample_id, visual_review_status=visual_status, bbox_after=bbox_after, region_type=region_type, review_notes=notes, reviewer=reviewer)
            st.rerun()
        except Exception as exc:
            st.error(str(exc))
    if recheck:
        return
    st.divider()
    st.subheader("Structure Ground Truth Review")
    _render_ground_truth_confirmation(store, item, reviewer, notes)


def _render_ground_truth_confirmation(store: SoloReviewStore, item: dict[str, Any], reviewer: str, notes: str) -> None:
    if not item.get("trusted_ground_truth_available"):
        st.info("没有可信结构真值，不可用于 OCSR 准确率评测。该样本只能完成 Visual Review，并可进入 detector/rejector 数据集。")
        return
    st.json({
        "ground_truth_origin": item.get("ground_truth_origin"), "ground_truth_smiles": item.get("ground_truth_smiles"),
        "ground_truth_inchikey": item.get("ground_truth_inchikey"), "source_compound_id": item.get("source_compound_id") or item.get("source_id"),
        "source_structure_file": item.get("source_structure_file"), "source_document": item.get("source_document"),
        "molscribe_matches_ground_truth": item.get("molscribe_matches_ground_truth"),
        "decimer_matches_ground_truth": item.get("decimer_matches_ground_truth"),
        "ensemble_matches_ground_truth": item.get("ensemble_matches_ground_truth"),
    })
    visual_complete = item.get("audit", {}).get("visual_review_status") == "valid_single_molecule_crop"
    disabled = not item.get("files_complete") or not visual_complete
    if not visual_complete:
        st.info("Save Visual Review as valid_single_molecule_crop before confirming the trusted structure label.")
    if st.button("Accept trusted ground truth", disabled=disabled, key=f"accept_truth_{item.get('sample_id')}"):
        try:
            store.submit_structure_ground_truth(str(item["sample_id"]), reviewer=reviewer, review_notes=notes)
            st.rerun()
        except Exception as exc:
            st.error(str(exc))


def _page_with_bbox(path: Path, bbox: list[int]) -> Image.Image:
    with Image.open(path) as image:
        preview = image.convert("RGB")
    if len(bbox) == 4:
        ImageDraw.Draw(preview).rectangle(tuple(bbox), outline="#d94841", width=max(2, preview.width // 300))
    return preview


def _crop_preview(page_path: str | None, bbox: list[int]) -> Image.Image | None:
    if not page_path or len(bbox) != 4:
        return None
    try:
        with Image.open(page_path) as image:
            x1, y1, x2, y2 = bbox
            if min(x1, y1) < 0 or x2 <= x1 or y2 <= y1 or x2 > image.width or y2 > image.height:
                return None
            return image.crop((x1, y1, x2, y2)).convert("RGB")
    except Exception:
        return None


def _current_visual_status(item: dict[str, Any]) -> str:
    value = str(item.get("audit", {}).get("visual_review_status") or "uncertain_visual")
    return value if value in VISUAL_REVIEW_STATUSES else "uncertain_visual"


def _region_index(item: dict[str, Any]) -> int:
    value = str(item.get("audit", {}).get("region_type_after") or item.get("machine_category") or item.get("category") or "invalid_crop")
    return REGION_TYPES.index(value) if value in REGION_TYPES else REGION_TYPES.index("invalid_crop")


def _parse_json(raw: str | None, default: Any) -> Any:
    try:
        return json.loads(raw or "")
    except (TypeError, ValueError):
        return default


def _model_failures(item: dict[str, Any]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for backend in ("molscribe", "decimer", "ensemble"):
        raw = _parse_json(item.get(f"{backend}_raw"), {})
        if not isinstance(raw, dict):
            continue
        if str(raw.get("status") or "").lower() not in {"", "success"} or raw.get("message"):
            failures.append({"backend": backend, "status": raw.get("status"), "message": raw.get("message")})
    return failures

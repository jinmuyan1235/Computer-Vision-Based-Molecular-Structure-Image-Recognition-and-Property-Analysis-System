"""Streamlit page for single-developer review of OCSR machine-review candidates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw
import streamlit as st

import config
from src.datasets.solo_review import REGION_TYPES, SoloReviewStore


def render_dataset_review_page() -> None:
    """Render the focused single-review workspace and its delayed recheck mode."""
    st.header("OCSR Data Review")
    store = SoloReviewStore(config.DATA_DIR / "ocsr_collections", review_root=config.DATA_DIR / "review")
    mode = st.segmented_control("Review mode", ["Queue", "Delayed recheck"], default="Queue", key="solo_review_mode")
    if mode == "Delayed recheck":
        _render_recheck_workspace(store)
    else:
        _render_queue_workspace(store)


def _render_queue_workspace(store: SoloReviewStore) -> None:
    controls = st.columns([0.25, 0.18, 0.18, 0.39])
    show_reviewed = controls[0].toggle("Show reviewed", value=False, key="solo_show_reviewed")
    items = store.list_items(include_reviewed=show_reviewed)
    controls[1].metric("Queue", len(items))
    controls[2].metric("Reviewed", sum(bool(item.get("audit")) for item in items))
    if not items:
        st.info("No items in data/review/human_review_queue.csv.")
        return
    ids = [str(item["sample_id"]) for item in items]
    selected = controls[3].selectbox("Sample", ids, key="solo_queue_sample")
    item = next(item for item in items if item["sample_id"] == selected)
    _render_item(store, item, recheck=False)


def _render_recheck_workspace(store: SoloReviewStore) -> None:
    controls = st.columns([0.2, 0.16, 0.16, 0.48])
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
    item = next(item for item in items if item["sample_id"] == selected)
    _render_item(store, item, recheck=True)


def _render_item(store: SoloReviewStore, item: dict[str, Any], *, recheck: bool) -> None:
    st.subheader(str(item.get("sample_id") or "Sample"))
    status_columns = st.columns(4)
    status_columns[0].metric("Status", item.get("effective_status") or item.get("verification_status") or "-")
    status_columns[1].metric("Quality", item.get("image_quality_level") or "-")
    status_columns[2].metric("Quality score", item.get("image_quality_score") or "-")
    status_columns[3].metric("Category", item.get("machine_category") or item.get("category") or "-")
    _render_images(item)
    _render_predictions(store, item)
    details = st.columns(2)
    with details[0]:
        st.caption("Risks")
        st.json(_parse_json(item.get("risk_reasons"), []))
    with details[1]:
        st.caption("Source")
        st.json({
            "document": item.get("source_document"),
            "url": item.get("source_url"),
            "license": item.get("source_license"),
            "attribution": item.get("attribution"),
            "bbox": _parse_json(item.get("bbox"), []),
        })
    if not recheck and item.get("audit"):
        st.caption("Latest single review")
        st.json(item["audit"])
    _render_actions(store, item, recheck=recheck)


def _render_images(item: dict[str, Any]) -> None:
    columns = st.columns(3)
    page = item.get("page_path_abs")
    bbox = _parse_json(item.get("bbox"), [])
    with columns[0]:
        st.caption("Document page")
        if page:
            st.image(_page_with_bbox(Path(str(page)), bbox), use_container_width=True)
        else:
            st.info("Source page is unavailable for this queue item.")
    with columns[1]:
        st.caption("Current crop")
        if item.get("crop_path_abs"):
            st.image(str(item["crop_path_abs"]), use_container_width=True)
        else:
            st.info("Crop is unavailable.")
    with columns[2]:
        st.caption("Current bbox preview")
        preview = _crop_preview(page, bbox)
        if preview is not None:
            st.image(preview, use_container_width=True)
        else:
            st.info("No source page crop preview.")


def _render_predictions(store: SoloReviewStore, item: dict[str, Any]) -> None:
    columns = st.columns(3)
    for column, backend in zip(columns, ("molscribe", "decimer", "ensemble")):
        with column:
            st.caption(backend.title())
            smiles = str(item.get(f"{backend}_smiles") or "")
            st.code(smiles or "No valid SMILES", language=None)
            redraw = store.prediction_redraw(str(item.get("sample_id") or "sample"), backend, smiles)
            if redraw:
                st.image(redraw, use_container_width=True)
            else:
                st.info("No redraw")
            with st.expander("Raw prediction", expanded=False):
                st.json(_parse_json(item.get(f"{backend}_raw"), {"smiles": smiles}))


def _render_actions(store: SoloReviewStore, item: dict[str, Any], *, recheck: bool) -> None:
    sample_id = str(item.get("sample_id") or "")
    bbox_before = _parse_json(item.get("bbox"), [])
    bbox_values = bbox_before if len(bbox_before) == 4 else [0, 0, 0, 0]
    editor = st.columns([0.18, 0.18, 0.18, 0.18, 0.28])
    backend = editor[0].selectbox("Candidate", ["ensemble", "molscribe", "decimer", "manual"], key=f"solo_backend_{'recheck_' if recheck else ''}{sample_id}")
    suggested = "" if backend == "manual" else str(item.get(f"{backend}_smiles") or "")
    final_smiles = editor[1].text_input("Final SMILES", value=suggested, key=f"solo_smiles_{'recheck_' if recheck else ''}{sample_id}_{backend}")
    region_type = editor[2].selectbox("Region type", REGION_TYPES, index=_region_index(item), key=f"solo_region_{'recheck_' if recheck else ''}{sample_id}")
    reviewer = editor[3].text_input("Reviewer", value="local", disabled=recheck, key=f"solo_reviewer_{sample_id}")
    review_notes = editor[4].text_input("Review notes", key=f"solo_notes_{'recheck_' if recheck else ''}{sample_id}")
    bbox_columns = st.columns(4)
    bbox_after = [
        int(bbox_columns[index].number_input(label, min_value=0, value=int(bbox_values[index]), step=1, key=f"solo_bbox_{'recheck_' if recheck else ''}{sample_id}_{label}"))
        for index, label in enumerate(("x1", "y1", "x2", "y2"))
    ]
    actions = st.columns(3)
    if actions[0].button("Accept", type="primary", key=f"solo_accept_{'recheck_' if recheck else ''}{sample_id}"):
        _submit(store, item, recheck, "human_verified_single", final_smiles, bbox_after, region_type, review_notes, reviewer, backend)
    if actions[1].button("Reject", key=f"solo_reject_{'recheck_' if recheck else ''}{sample_id}"):
        _submit(store, item, recheck, "rejected", final_smiles, bbox_after, region_type, review_notes, reviewer, backend)
    if actions[2].button("Uncertain", key=f"solo_uncertain_{'recheck_' if recheck else ''}{sample_id}"):
        _submit(store, item, recheck, "uncertain", final_smiles, bbox_after, region_type, review_notes, reviewer, backend)


def _submit(store: SoloReviewStore, item: dict[str, Any], recheck: bool, status: str, final_smiles: str, bbox_after: list[int], region_type: str, notes: str, reviewer: str, backend: str) -> None:
    try:
        if recheck:
            store.submit_recheck(str(item["sample_id"]), verification_status=status, final_smiles=final_smiles, bbox_after=bbox_after, region_type=region_type, review_notes=notes)
        else:
            store.submit(str(item["sample_id"]), verification_status=status, final_smiles=final_smiles, bbox_after=bbox_after, region_type=region_type, review_notes=notes, reviewer=reviewer, selected_prediction=backend)
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


def _region_index(item: dict[str, Any]) -> int:
    value = str(item.get("machine_category") or item.get("category") or "invalid_crop")
    return REGION_TYPES.index(value) if value in REGION_TYPES else REGION_TYPES.index("invalid_crop")


def _parse_json(raw: str | None, default: Any) -> Any:
    try:
        return json.loads(raw or "")
    except (TypeError, ValueError):
        return default

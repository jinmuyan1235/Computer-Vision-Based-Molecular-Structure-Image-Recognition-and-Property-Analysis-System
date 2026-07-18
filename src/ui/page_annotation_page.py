"""Streamlit page-level bbox annotation workspace."""

from __future__ import annotations

import os
from pathlib import Path

from PIL import Image, ImageDraw
import streamlit as st

from src.datasets.page_annotations import PAGE_REGION_CLASSES, PageAnnotationStore
from src.documents.detectors import HeuristicMoleculeRegionDetector
from src.documents.models import DocumentPage


LAYOUT_TAGS = ("structure_dense", "reaction_scheme", "table_or_sar", "ordinary_text")


def _proposal_boxes(page: dict, root: Path, profile: str) -> list[tuple[int, int, int, int]]:
    image_path = root / page["image_path"]
    model_page = DocumentPage(
        document_id=page["source_document"], page_number=int(page["page_number"]),
        image_path=str(image_path), width=int(page["width"]), height=int(page["height"]),
    )
    detector = HeuristicMoleculeRegionDetector(proposal_config=profile, crop_screening_config="candidate")
    return [tuple(region.bbox) for region in detector.propose(model_page)]


def _overlay(page: dict, root: Path, baseline: bool, candidate: bool) -> Image.Image:
    image = Image.open(root / page["image_path"]).convert("RGB")
    draw = ImageDraw.Draw(image)
    for item in page.get("annotations", []):
        draw.rectangle(tuple(item["bbox"]), outline="#00a86b", width=4)
        draw.text((item["bbox"][0] + 3, item["bbox"][1] + 3), item["class"], fill="#007a4d")
    if baseline:
        for bbox in _proposal_boxes(page, root, "baseline"):
            draw.rectangle(bbox, outline="#e74c3c", width=3)
    if candidate:
        for bbox in _proposal_boxes(page, root, "candidate"):
            draw.rectangle(bbox, outline="#3465d9", width=3)
    return image


def render_page_annotation_workspace() -> None:
    root = Path(os.getenv("OCSR_PAGE_ANNOTATION_ROOT", "data/page_annotations/visual-page-holdout-v0.1"))
    try:
        store = PageAnnotationStore(root)
    except FileNotFoundError as exc:
        st.info(str(exc)); return
    payload = store.load()
    page_ids = store.page_ids()
    completed = sum(page.get("annotation_status") == "completed" for page in payload["pages"].values())
    st.subheader("Page Annotation")
    st.caption(
        f"Independent page truth: {completed}/{len(page_ids)} pages saved. "
        "Proposal source is hidden by default; do not use OCSR predictions as visual truth."
    )
    selected = st.selectbox("Document / page", page_ids, key="page_annotation_selected")
    page = store.page(selected)
    toggles = st.columns(3)
    show_baseline = toggles[0].toggle("Show baseline boxes", value=False, key="page_show_baseline")
    show_candidate = toggles[1].toggle("Show candidate boxes", value=False, key="page_show_candidate")
    toggles[2].caption("Green=truth · Red=baseline · Blue=candidate")
    st.image(_overlay(page, store.root, show_baseline, show_candidate), use_container_width=True)
    draft_key = f"page_annotation_draft_{selected}"
    if draft_key not in st.session_state:
        st.session_state[draft_key] = [
            {"bbox": list(item["bbox"]), "class": item["class"]}
            for item in page.get("annotations", [])
        ]
    draft = st.session_state[draft_key]
    st.markdown("#### Ground-truth boxes")
    edited_rows: list[dict] = []
    delete_index: int | None = None
    for index, item in enumerate(draft):
        columns = st.columns([0.8, 0.8, 0.8, 0.8, 1.3, 0.7])
        values = [
            columns[offset].number_input(
                label, min_value=0, value=int(item["bbox"][offset]), step=1,
                key=f"page_box_{selected}_{index}_{label}",
            )
            for offset, label in enumerate(("x1", "y1", "x2", "y2"))
        ]
        region_class = columns[4].selectbox(
            "class", PAGE_REGION_CLASSES, index=PAGE_REGION_CLASSES.index(item["class"]),
            key=f"page_box_{selected}_{index}_class",
        )
        if columns[5].button("Delete", key=f"page_box_{selected}_{index}_delete"):
            delete_index = index
        edited_rows.append({"bbox": values, "class": region_class})
    if delete_index is not None:
        draft.pop(delete_index)
        for key in list(st.session_state):
            if key.startswith(f"page_box_{selected}_"):
                del st.session_state[key]
        st.rerun()
    if st.button("Add bbox", key=f"page_add_{selected}"):
        width, height = int(page["width"]), int(page["height"])
        draft.append({
            "bbox": [width // 4, height // 4, 3 * width // 4, 3 * height // 4],
            "class": "molecule",
        })
        st.rerun()
    annotator = st.text_input("Annotator", value=page.get("annotator", ""), key=f"page_annotator_{selected}")
    saved_layout_tags = page.get("layout_tags", [])
    if not isinstance(saved_layout_tags, list):
        saved_layout_tags = []
    layout_tags = st.multiselect(
        "Page layout tags (quality description only; not used to select pages)",
        LAYOUT_TAGS, default=saved_layout_tags, key=f"page_layout_{selected}",
    )
    if st.button("Save this page", type="primary", key=f"page_save_{selected}"):
        try:
            store.save_page(selected, edited_rows, annotator=annotator, layout_tags=layout_tags)
        except (KeyError, TypeError, ValueError) as exc:
            st.error(str(exc))
        else:
            st.success(f"Saved {len(edited_rows)} boxes for {selected}.")
            st.rerun()

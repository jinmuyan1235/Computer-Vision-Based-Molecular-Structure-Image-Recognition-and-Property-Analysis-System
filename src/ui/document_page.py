"""PDF and multi-molecule document page."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from src.documents.input_loader import DocumentInputError, OptionalDependencyError
from src.documents.processor import DocumentOCSRProcessor
from src.ui.image_viewer import show_document_page
from src.ui.labels import REGION_TYPE_LABELS, localize_region_rows
from src.ui.state import get_document_processor, remember_backend_status
from src.ui.streamlit_compat import dataframe_stretch
from src.ui.styles import page_intro


PROCESS_MODE_DETECT = "仅检测分子区域（速度快，不执行结构识别）"
PROCESS_MODE_FULL = "检测并识别分子结构（调用 OCSR，耗时较长）"


def render_document_page(backend: str) -> None:
    page_intro("PDF/多分子文档", "上传 PDF、页面图片或 ZIP 图片集合，先检测分子区域，再按需执行 OCSR。")
    upload = st.file_uploader(
        "上传 PDF / 页面图片 / ZIP",
        type=["pdf", "png", "jpg", "jpeg", "zip"],
        key="document_upload",
    )
    mode = st.radio("处理模式", [PROCESS_MODE_DETECT, PROCESS_MODE_FULL], index=0, horizontal=False)
    run_ocsr = mode == PROCESS_MODE_FULL
    if upload is not None and st.button("开始处理文档", type="primary", key="process_document"):
        suffix = Path(upload.name).suffix.lower()
        prefix = Path(upload.name).stem + "_"
        with tempfile.NamedTemporaryFile(prefix=prefix, suffix=suffix, delete=False) as temporary:
            temporary.write(upload.getvalue())
            temporary_path = Path(temporary.name)
        try:
            progress_text = st.empty()
            progress_bar = st.progress(0.0)

            def progress_callback(current: int, total: int, region_id: str) -> None:
                progress_text.info(f"正在识别第 {current}/{total} 个候选区域：{region_id}")
                progress_bar.progress(current / max(total, 1))

            with st.spinner("正在渲染页面并检测区域……"):
                st.session_state["document_result"] = get_document_processor(backend).process(
                    temporary_path,
                    run_ocsr=run_ocsr,
                    progress_callback=progress_callback if run_ocsr else None,
                )
                remember_backend_status(backend)
            progress_text.empty()
            progress_bar.empty()
        except OptionalDependencyError as exc:
            st.error(str(exc))
        except (DocumentInputError, FileNotFoundError, ValueError) as exc:
            st.error(str(exc))
        finally:
            temporary_path.unlink(missing_ok=True)
    if "document_result" in st.session_state:
        st.session_state["document_result"] = show_document_result(st.session_state["document_result"], backend)


def show_document_result(document_result: dict, backend: str) -> dict:
    summary = document_result.get("summary") or {}
    st.subheader("文档区域识别结果")
    metrics = st.columns(4)
    metrics[0].metric("页数", summary.get("page_count", 0))
    metrics[1].metric("检测区域", summary.get("region_count", 0))
    metrics[2].metric("分子候选区域", summary.get("molecule_region_count", 0))
    metrics[3].metric("识别成功", summary.get("recognized_region_count", 0))
    processing = document_result.get("processing") or {}
    if processing.get("total_time_ms") is not None:
        st.caption(f"文档处理总耗时：{processing.get('total_time_ms')} ms")
    if document_result.get("detection_errors"):
        with st.expander("检测提示", expanded=False):
            st.json(document_result.get("detection_errors"))

    annotated = [item for item in (document_result.get("exports", {}).get("annotated_pages") or "").split(",") if item]
    if annotated:
        st.subheader("区域标注预览")
        page_names = [Path(path).name for path in annotated]
        selected_name = st.selectbox("选择页码", page_names, key="annotated_page_select")
        selected_path = annotated[page_names.index(selected_name)]
        show_document_page(selected_path, selected_name)

    rows = DocumentOCSRProcessor.region_rows(document_result)
    if rows:
        important = [
            "page_number",
            "region_id",
            "region_type",
            "detection_confidence",
            "status",
            "message",
            "final_smiles",
            "valid",
            "inference_time_ms",
            "processing_time_ms",
            "screening_reason",
        ]
        display_rows = [{key: row.get(key) for key in important} for row in rows]
        dataframe_stretch(pd.DataFrame(localize_region_rows(display_rows)), hide_index=True)
        with st.expander("查看完整字段", expanded=False):
            dataframe_stretch(pd.DataFrame(localize_region_rows(rows)), hide_index=True)
    else:
        st.info("未检测到分子候选区域。可以在下方手动添加区域。")

    document_result = _region_editor(document_result, backend)
    _download_panel(document_result)
    return document_result


def _region_editor(document_result: dict, backend: str) -> dict:
    active = [region for region in document_result.get("regions", []) if region.get("status") != "deleted"]
    processor = get_document_processor(backend)
    if active:
        st.subheader("编辑检测区域")
        region_ids = [region["region_id"] for region in active]
        selected_id = st.selectbox("区域", region_ids, key="document_region_select")
        selected = next(region for region in active if region["region_id"] == selected_id)
        bbox = selected.get("bbox") or [0, 0, 1, 1]
        first_row = st.columns(2)
        x1 = first_row[0].number_input("x1", min_value=0, value=int(bbox[0]), key=f"edit_x1_{selected_id}")
        y1 = first_row[1].number_input("y1", min_value=0, value=int(bbox[1]), key=f"edit_y1_{selected_id}")
        second_row = st.columns(2)
        x2 = second_row[0].number_input("x2", min_value=1, value=int(bbox[2]), key=f"edit_x2_{selected_id}")
        y2 = second_row[1].number_input("y2", min_value=1, value=int(bbox[3]), key=f"edit_y2_{selected_id}")
        allowed = list(REGION_TYPE_LABELS)
        current = selected.get("region_type") if selected.get("region_type") in allowed else "unknown"
        region_type = st.selectbox(
            "区域类型",
            allowed,
            index=allowed.index(current),
            format_func=lambda value: REGION_TYPE_LABELS[value],
            key=f"edit_type_{selected_id}",
        )
        actions = st.columns(3)
        if actions[0].button("更新区域并重新识别", key=f"update_region_{selected_id}"):
            document_result = processor.apply_edits(
                document_result,
                [{"action": "update", "region_id": selected_id, "bbox": [x1, y1, x2, y2], "region_type": region_type}],
                rerun_ocsr=True,
            )
            st.session_state["document_result"] = document_result
            st.success("区域已更新并重新处理。")
        if actions[1].button("删除区域", key=f"delete_region_{selected_id}"):
            document_result = processor.apply_edits(
                document_result,
                [{"action": "delete", "region_id": selected_id, "note": "用户在界面删除区域。"}],
                rerun_ocsr=False,
            )
            st.session_state["document_result"] = document_result
            st.warning("区域已删除。")
        if actions[2].button("标记为非分子", key=f"mark_region_{selected_id}"):
            document_result = processor.apply_edits(
                document_result,
                [{"action": "mark", "region_id": selected_id, "region_type": "non_molecule"}],
                rerun_ocsr=False,
            )
            st.session_state["document_result"] = document_result
            st.success("区域已标记为非分子。")

    st.subheader("添加遗漏区域")
    page_numbers = [int(page["page_number"]) for page in document_result.get("pages", [])]
    if page_numbers:
        page_number = st.selectbox("页码", page_numbers, key="add_region_page")
        row1 = st.columns(2)
        add_x1 = row1[0].number_input("新区域 x1", min_value=0, value=0, key="add_x1")
        add_y1 = row1[1].number_input("新区域 y1", min_value=0, value=0, key="add_y1")
        row2 = st.columns(2)
        add_x2 = row2[0].number_input("新区域 x2", min_value=1, value=200, key="add_x2")
        add_y2 = row2[1].number_input("新区域 y2", min_value=1, value=200, key="add_y2")
        add_type = st.selectbox(
            "新区域类型",
            ["molecule", "text", "table", "reaction_like", "unknown"],
            format_func=lambda value: REGION_TYPE_LABELS[value],
            key="add_type",
        )
        if st.button("添加区域并识别", key="add_region"):
            document_result = processor.apply_edits(
                document_result,
                [{"action": "add", "page_number": page_number, "bbox": [add_x1, add_y1, add_x2, add_y2], "region_type": add_type}],
                rerun_ocsr=True,
            )
            st.session_state["document_result"] = document_result
            st.success("区域已添加。")
    return document_result


def _download_panel(document_result: dict) -> None:
    exports = document_result.get("exports") or {}
    with st.expander("结果导出", expanded=True):
        if exports.get("json") and Path(exports["json"]).is_file():
            st.download_button("下载文档分析结果", Path(exports["json"]).read_bytes(), "document_result.json", "application/json")
        if exports.get("regions_csv") and Path(exports["regions_csv"]).is_file():
            st.download_button("下载区域结果表", Path(exports["regions_csv"]).read_bytes(), "regions.csv", "text/csv")
        if exports.get("zip") and Path(exports["zip"]).is_file():
            st.download_button("下载完整结果包", Path(exports["zip"]).read_bytes(), "document_results.zip", "application/zip")

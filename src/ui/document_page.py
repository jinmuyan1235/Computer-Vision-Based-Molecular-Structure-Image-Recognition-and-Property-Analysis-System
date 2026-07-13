"""PDF and multi-molecule document page."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from src.documents.input_loader import DocumentInputError, OptionalDependencyError
from src.documents.processor import DocumentOCSRProcessor
from src.ui.image_viewer import show_document_page
from src.ui.labels import REGION_TYPE_LABELS, localize_region_rows
from src.ui.state import current_runtime_key, get_document_processor, remember_backend_status, runtime_config_from_key
from src.ui.streamlit_compat import dataframe_stretch
from src.ui.styles import page_intro


PROCESS_MODE_DETECT = "仅检测分子区域（速度快，不执行结构识别）"
PROCESS_MODE_FULL = "检测并识别分子结构（调用 OCSR，耗时较长）"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


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
                if backend != "demo":
                    st.session_state["document_result"] = _process_document_subprocess(temporary_path, backend, run_ocsr)
                else:
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
        except (DocumentInputError, FileNotFoundError, RuntimeError, ValueError) as exc:
            st.error(str(exc))
        finally:
            temporary_path.unlink(missing_ok=True)
    if "document_result" in st.session_state:
        st.session_state["document_result"] = show_document_result(st.session_state["document_result"], backend)


def _process_document_subprocess(input_path: Path, backend: str, run_ocsr: bool) -> dict:
    """Run document OCSR outside Streamlit so native model crashes cannot kill the UI server."""
    runtime = runtime_config_from_key(current_runtime_key())
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "process_document.py"),
        "--input",
        str(input_path),
        "--backend",
        backend,
    ]
    if not run_ocsr:
        command.append("--detect-only")
    if runtime.get("molscribe_device"):
        command.extend(["--molscribe-device", str(runtime["molscribe_device"])])
    if runtime.get("decimer_device"):
        command.extend(["--decimer-device", str(runtime["decimer_device"])])
    if runtime.get("visible_gpu_index") is not None:
        command.extend(["--visible-gpu-index", str(runtime["visible_gpu_index"])])

    env = os.environ.copy()
    env.setdefault("MOLSCRIBE_ISOLATED_SUBPROCESS", "true")
    env.setdefault("DECIMER_ISOLATED_SUBPROCESS", "true")
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    payload = _extract_json_object(completed.stdout)
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip().splitlines()
        message = detail[-1] if detail else f"文档处理子进程退出码 {completed.returncode}"
        if payload and payload.get("message"):
            message = str(payload["message"])
        raise RuntimeError(message)
    if not payload or not payload.get("result_path"):
        raise RuntimeError("文档处理子进程未返回结果文件路径。")
    result_path = Path(str(payload["result_path"]))
    if not result_path.is_file():
        raise RuntimeError(f"文档处理结果文件不存在：{result_path}")
    return json.loads(result_path.read_text(encoding="utf-8"))


def _apply_document_edits_subprocess(
    document_result: dict,
    backend: str,
    edits: list[dict],
    rerun_ocsr: bool,
) -> dict:
    """Apply document edits outside Streamlit when a real backend may be involved."""
    runtime = runtime_config_from_key(current_runtime_key())
    output_dir = Path(document_result.get("output_dir") or tempfile.gettempdir()).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    current_path = output_dir / "document_result_for_edit.json"
    current_path.write_text(json.dumps(document_result, ensure_ascii=False, indent=2), encoding="utf-8")
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "process_document_edit.py"),
        "--document-result",
        str(current_path),
        "--edits-json",
        json.dumps(edits, ensure_ascii=False),
        "--backend",
        backend,
    ]
    if rerun_ocsr:
        command.append("--rerun-ocsr")
    if runtime.get("molscribe_device"):
        command.extend(["--molscribe-device", str(runtime["molscribe_device"])])
    if runtime.get("decimer_device"):
        command.extend(["--decimer-device", str(runtime["decimer_device"])])
    if runtime.get("visible_gpu_index") is not None:
        command.extend(["--visible-gpu-index", str(runtime["visible_gpu_index"])])

    env = os.environ.copy()
    env.setdefault("MOLSCRIBE_ISOLATED_SUBPROCESS", "true")
    env.setdefault("DECIMER_ISOLATED_SUBPROCESS", "true")
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    payload = _extract_json_object(completed.stdout)
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip().splitlines()
        message = detail[-1] if detail else f"文档编辑子进程退出码 {completed.returncode}"
        if payload and payload.get("message"):
            message = str(payload["message"])
        raise RuntimeError(message)
    if not payload or not payload.get("result_path"):
        raise RuntimeError("文档编辑子进程未返回结果文件路径。")
    result_path = Path(str(payload["result_path"]))
    if not result_path.is_file():
        raise RuntimeError(f"文档编辑结果文件不存在：{result_path}")
    return json.loads(result_path.read_text(encoding="utf-8"))


def _apply_document_edits(document_result: dict, backend: str, edits: list[dict], rerun_ocsr: bool) -> dict:
    if backend == "demo":
        processor = get_document_processor(backend)
        return processor.apply_edits(document_result, edits, rerun_ocsr=rerun_ocsr)
    return _apply_document_edits_subprocess(document_result, backend, edits, rerun_ocsr)


def _extract_json_object(text: str) -> dict | None:
    """Extract a JSON object from stdout that may also contain native-library logs."""
    stripped = text.strip()
    if not stripped:
        return None
    decoder = json.JSONDecoder()
    for index, char in enumerate(stripped):
        if char != "{":
            continue
        try:
            value, _end = decoder.raw_decode(stripped[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


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
            try:
                document_result = _apply_document_edits(
                    document_result,
                    backend,
                    [{"action": "update", "region_id": selected_id, "bbox": [x1, y1, x2, y2], "region_type": region_type}],
                    rerun_ocsr=True,
                )
                st.session_state["document_result"] = document_result
                st.success("区域已更新并重新处理。")
            except RuntimeError as exc:
                st.error(f"区域更新失败：{exc}")
        if actions[1].button("删除区域", key=f"delete_region_{selected_id}"):
            try:
                document_result = _apply_document_edits(
                    document_result,
                    backend,
                    [{"action": "delete", "region_id": selected_id, "note": "用户在界面删除区域。"}],
                    rerun_ocsr=False,
                )
                st.session_state["document_result"] = document_result
                st.warning("区域已删除。")
            except RuntimeError as exc:
                st.error(f"区域删除失败：{exc}")
        if actions[2].button("标记为非分子", key=f"mark_region_{selected_id}"):
            try:
                document_result = _apply_document_edits(
                    document_result,
                    backend,
                    [{"action": "mark", "region_id": selected_id, "region_type": "non_molecule"}],
                    rerun_ocsr=False,
                )
                st.session_state["document_result"] = document_result
                st.success("区域已标记为非分子。")
            except RuntimeError as exc:
                st.error(f"区域标记失败：{exc}")

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
            try:
                document_result = _apply_document_edits(
                    document_result,
                    backend,
                    [{"action": "add", "page_number": page_number, "bbox": [add_x1, add_y1, add_x2, add_y2], "region_type": add_type}],
                    rerun_ocsr=True,
                )
                st.session_state["document_result"] = document_result
                st.success("区域已添加。")
            except RuntimeError as exc:
                st.error(f"区域添加失败：{exc}")
    return document_result

def _download_panel(document_result: dict) -> None:
    exports = document_result.get("exports") or {}
    with st.expander("结果导出", expanded=False):
        json_path = Path(exports.get("json") or "")
        csv_path = Path(exports.get("regions_csv") or "")
        zip_path = Path(exports.get("zip") or "")
        if json_path.is_file():
            st.download_button("下载文档分析结果", json_path.read_bytes(), "document_result.json", "application/json")
        if csv_path.is_file():
            st.download_button("下载区域结果表", csv_path.read_bytes(), "regions.csv", "text/csv")
        if zip_path.is_file():
            st.caption(f"完整结果包已生成：{zip_path.name}（{zip_path.stat().st_size / 1024 / 1024:.2f} MB）")
            if st.button("准备完整结果包下载", key="prepare_document_zip_download"):
                st.session_state["document_zip_download_bytes"] = zip_path.read_bytes()
            if st.session_state.get("document_zip_download_bytes") is not None:
                st.download_button(
                    "下载完整结果包",
                    st.session_state["document_zip_download_bytes"],
                    "document_results.zip",
                    "application/zip",
                )

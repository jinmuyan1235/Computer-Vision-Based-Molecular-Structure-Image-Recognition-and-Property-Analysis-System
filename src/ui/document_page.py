"""PDF and multi-molecule document page."""

from __future__ import annotations

import base64
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

import streamlit as st
import streamlit.components.v1 as components

import config
from src.documents.input_loader import DocumentInputError, OptionalDependencyError
from src.documents.processor import DocumentOCSRProcessor
from src.storage.analysis_repository import record_result_payload
from src.ui.image_viewer import show_document_page
from src.ui.labels import REGION_TYPE_LABELS, localize_region_rows
from src.ui.records import render_records
from src.ui.state import current_runtime_key, get_document_processor, remember_backend_status, runtime_config_from_key
from src.ui.styles import page_intro
from src.runtime.job_manager import (
    extract_json_object,
    run_json_command,
    start_logged_process,
    terminate_process_tree,
)

PROCESS_MODE_DETECT = "仅检测分子区域（速度快，不执行结构识别）"
PROCESS_MODE_FULL = "检测并识别已确认分子区域（新检测结果需先人工确认）"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
AUDIT_REGION_TYPES = ["molecule", "text", "table", "reaction", "ignore"]
DOCUMENT_BBOX_QUERY_KEYS = (
    "document_region_editor_key",
    "doc_bbox_x1",
    "doc_bbox_y1",
    "doc_bbox_x2",
    "doc_bbox_y2",
    "doc_bbox_nonce",
)


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
        try:
            if backend != "demo":
                input_path = _persist_uploaded_document(upload)
                _start_document_job(input_path, backend, run_ocsr)
                st.session_state.pop("document_result", None)
                st.rerun()
            else:
                _run_demo_document(upload, backend, run_ocsr)
        except OptionalDependencyError as exc:
            st.error(str(exc))
        except (DocumentInputError, FileNotFoundError, RuntimeError, ValueError) as exc:
            st.error(str(exc))

    _render_document_job_status()
    if "document_result" in st.session_state:
        st.session_state["document_result"] = show_document_result(st.session_state["document_result"], backend)


def _run_demo_document(upload: Any, backend: str, run_ocsr: bool) -> None:
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
            result = get_document_processor(backend).process(
                temporary_path,
                run_ocsr=run_ocsr,
                progress_callback=progress_callback if run_ocsr else None,
            )
            record_result_payload(result, (result.get("exports") or {}).get("json"))
            st.session_state["document_result"] = result
            remember_backend_status(backend)
        progress_text.empty()
        progress_bar.empty()
    finally:
        temporary_path.unlink(missing_ok=True)


def _persist_uploaded_document(upload: Any) -> Path:
    job_dir = config.OUTPUT_DIR / "ui_jobs"
    job_dir.mkdir(parents=True, exist_ok=True)
    safe_name = Path(upload.name).name.replace(" ", "_")
    path = job_dir / f"document_{uuid4().hex}_{safe_name}"
    path.write_bytes(upload.getvalue())
    return path


def _document_command(input_path: Path, backend: str, run_ocsr: bool) -> list[str]:
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
    return command


def _start_document_job(input_path: Path, backend: str, run_ocsr: bool) -> None:
    job_dir = config.OUTPUT_DIR / "ui_jobs" / uuid4().hex
    job_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = job_dir / "stdout.log"
    stderr_path = job_dir / "stderr.log"
    env = os.environ.copy()
    env.setdefault("MOLSCRIBE_ISOLATED_SUBPROCESS", "true")
    env.setdefault("DECIMER_ISOLATED_SUBPROCESS", "true")
    process = start_logged_process(
        _document_command(input_path, backend, run_ocsr),
        cwd=PROJECT_ROOT,
        env=env,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )
    st.session_state["document_job"] = {
        "process": process,
        "backend": backend,
        "input_path": str(input_path),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "started_at": time.time(),
    }


def _render_document_job_status() -> None:
    job = st.session_state.get("document_job")
    if not job:
        return
    process = job["process"]
    elapsed = time.time() - float(job.get("started_at", time.time()))
    return_code = process.poll()
    if return_code is None:
        st.info(f"文档处理正在后台运行，已耗时 {elapsed:.1f} 秒。页面会自动刷新，期间不会阻塞 Streamlit。")
        st.caption(f"后台日志：{job.get('stdout_path')} / {job.get('stderr_path')}")
        st.progress(min(elapsed / 120.0, 0.95))
        if st.button("取消文档后台任务", key="cancel_document_job"):
            terminate_process_tree(process)
            input_path = Path(str(job.get("input_path") or ""))
            if input_path.exists():
                input_path.unlink(missing_ok=True)
            st.session_state.pop("document_job", None)
            st.warning("已取消文档处理任务并终止后台进程。")
            return
        time.sleep(2)
        st.rerun()

    stdout_path = Path(job["stdout_path"])
    stderr_path = Path(job["stderr_path"])
    stdout = stdout_path.read_text(encoding="utf-8", errors="replace") if stdout_path.is_file() else ""
    stderr = stderr_path.read_text(encoding="utf-8", errors="replace") if stderr_path.is_file() else ""
    payload = _extract_json_object(stdout)
    input_path = Path(str(job.get("input_path") or ""))
    if input_path.exists():
        input_path.unlink(missing_ok=True)
    st.session_state.pop("document_job", None)

    if return_code != 0:
        detail = (stderr or stdout or "").strip().splitlines()
        message = detail[-1] if detail else f"文档处理子进程退出码 {return_code}"
        if payload and payload.get("message"):
            message = str(payload["message"])
        st.error(f"文档处理失败：{message}")
        return
    if not payload or not payload.get("result_path"):
        st.error("文档处理完成，但没有返回结果文件路径。")
        return
    result_path = Path(str(payload["result_path"]))
    if not result_path.is_file():
        st.error(f"文档处理结果文件不存在：{result_path}")
        return
    result = json.loads(result_path.read_text(encoding="utf-8"))
    record_result_payload(result, result_path)
    st.session_state["document_result"] = result
    st.session_state["document_job_logs"] = {
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
    }
    st.success("文档处理完成。")
    st.rerun()


def _apply_document_edits_subprocess(
    document_result: dict,
    backend: str,
    edits: list[dict],
    rerun_ocsr: bool,
) -> dict:
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
    completed = run_json_command(command, cwd=PROJECT_ROOT, env=env, timeout=900)
    payload = completed.payload
    if completed.timed_out:
        raise RuntimeError("文档编辑子进程超时，已终止后台进程。")
    if completed.returncode != 0:
        message = completed.last_output_line() or f"文档编辑子进程退出码 {completed.returncode}"
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
        updated = processor.apply_edits(document_result, edits, rerun_ocsr=rerun_ocsr)
    else:
        updated = _apply_document_edits_subprocess(document_result, backend, edits, rerun_ocsr)
    record_result_payload(updated, (updated.get("exports") or {}).get("json"))
    return updated


def _extract_json_object(text: str) -> dict | None:
    return extract_json_object(text)


def show_document_result(document_result: dict, backend: str) -> dict:
    summary = document_result.get("summary") or {}
    st.subheader("文档区域识别结果")
    metrics = st.columns(6)
    metrics[0].metric("页数", summary.get("page_count", 0))
    metrics[1].metric("检测区域", summary.get("region_count", 0))
    metrics[2].metric("已确认区域", summary.get("confirmed_region_count", 0))
    metrics[3].metric("分子候选区域", summary.get("molecule_region_count", 0))
    metrics[4].metric("识别成功", summary.get("recognized_region_count", 0))
    metrics[5].metric("入审核队列", summary.get("review_queue_count", 0))
    processing = document_result.get("processing") or {}
    if processing.get("total_time_ms") is not None:
        st.caption(f"文档处理总耗时：{processing.get('total_time_ms')} ms")
    if document_result.get("detection_errors"):
        with st.expander("检测提示", expanded=False):
            st.json(document_result.get("detection_errors"))

    annotated = [item for item in (document_result.get("exports", {}).get("annotated_pages") or "").split(",") if item]
    if annotated and st.checkbox("显示区域标注预览", value=False, key="show_document_annotation_preview"):
        st.subheader("区域标注预览")
        page_names = [Path(path).name for path in annotated]
        selected_name = st.selectbox("选择页码", page_names, key="annotated_page_select")
        selected_path = annotated[page_names.index(selected_name)]
        show_document_page(selected_path, selected_name)

    rows = DocumentOCSRProcessor.region_rows(document_result)
    if not rows:
        st.info("未检测到分子候选区域。可以在下方手动添加区域。")
    elif st.checkbox("显示区域结果表", value=False, key="show_document_region_table"):
        important = [
            "page_number",
            "region_id",
            "region_type",
            "detection_confidence",
            "status",
            "confirmed",
            "message",
            "final_smiles",
            "valid",
            "review_queued",
            "inference_time_ms",
            "processing_time_ms",
            "screening_reason",
        ]
        display_rows = [{key: row.get(key) for key in important} for row in rows]
        render_records(
            localize_region_rows(display_rows),
            title_keys=("区域 ID",),
            summary_keys=("页码", "区域类型", "状态", "最终 SMILES", "推理耗时(ms)"),
        )
        if st.checkbox("显示完整字段", value=False, key="show_document_full_table"):
            render_records(
                localize_region_rows(rows),
                title_keys=("区域 ID",),
                summary_keys=("页码", "区域类型", "状态", "说明", "最终 SMILES"),
                max_records=100,
            )

    if st.checkbox("显示区域编辑工具", value=False, key="show_document_region_editor"):
        document_result = _region_editor(document_result, backend)

    logs = st.session_state.get("document_job_logs") or {}
    if logs:
        st.caption(f"最近一次后台日志：{logs.get('stdout_path')} / {logs.get('stderr_path')}")

    if st.checkbox("显示结果导出", value=False, key="show_document_downloads"):
        _download_panel(document_result)
    return document_result


def _region_editor(document_result: dict, backend: str) -> dict:
    active = [region for region in document_result.get("regions", []) if region.get("status") != "deleted"]
    if active:
        st.subheader("审核和编辑检测区域")
        region_ids = [region["region_id"] for region in active]
        selected_id = st.selectbox("区域", region_ids, key="document_region_select")
        selected = next(region for region in active if region["region_id"] == selected_id)
        _consume_document_bbox_query(selected_id)
        bbox = selected.get("bbox") or [0, 0, 1, 1]
        for key, value in zip(("x1", "y1", "x2", "y2"), bbox):
            st.session_state.setdefault(f"edit_{key}_{selected_id}", int(value))

        page = _page_for_region(document_result, selected)
        if page:
            preview_bbox = [
                int(st.session_state.get(f"edit_x1_{selected_id}", bbox[0])),
                int(st.session_state.get(f"edit_y1_{selected_id}", bbox[1])),
                int(st.session_state.get(f"edit_x2_{selected_id}", bbox[2])),
                int(st.session_state.get(f"edit_y2_{selected_id}", bbox[3])),
            ]
            _render_bbox_dragger(page, active, selected_id, preview_bbox)

        first_row = st.columns(2)
        x1 = first_row[0].number_input("x1", min_value=0, value=int(bbox[0]), key=f"edit_x1_{selected_id}")
        y1 = first_row[1].number_input("y1", min_value=0, value=int(bbox[1]), key=f"edit_y1_{selected_id}")
        second_row = st.columns(2)
        x2 = second_row[0].number_input("x2", min_value=1, value=int(bbox[2]), key=f"edit_x2_{selected_id}")
        y2 = second_row[1].number_input("y2", min_value=1, value=int(bbox[3]), key=f"edit_y2_{selected_id}")
        allowed = list(AUDIT_REGION_TYPES)
        current = _audit_region_type(selected.get("region_type"))
        region_type = st.selectbox(
            "区域类型",
            allowed,
            index=allowed.index(current),
            format_func=lambda value: REGION_TYPE_LABELS[value],
            key=f"edit_type_{selected_id}",
        )
        confirmed = bool(selected.get("confirmed"))
        status_text = "已确认" if confirmed else "待确认"
        st.caption(f"当前审核状态：{status_text}；只有已确认的分子区域会被送入 OCSR。")
        actions = st.columns(4)
        if actions[0].button("保存区域", key=f"update_region_{selected_id}"):
            try:
                document_result = _apply_document_edits(
                    document_result,
                    backend,
                    [{
                        "action": "update",
                        "region_id": selected_id,
                        "bbox": [x1, y1, x2, y2],
                        "region_type": region_type,
                        "confirmed": confirmed,
                    }],
                    rerun_ocsr=False,
                )
                st.session_state["document_result"] = document_result
                st.success("区域已保存。")
            except RuntimeError as exc:
                st.error(f"区域更新失败：{exc}")
        if actions[1].button("确认并识别", key=f"confirm_region_{selected_id}", type="primary"):
            try:
                document_result = _apply_document_edits(
                    document_result,
                    backend,
                    [{
                        "action": "update",
                        "region_id": selected_id,
                        "bbox": [x1, y1, x2, y2],
                        "region_type": region_type,
                        "confirmed": True,
                    }],
                    rerun_ocsr=region_type == "molecule",
                )
                st.session_state["document_result"] = document_result
                st.success("区域已确认；分子区域已重新识别。")
            except RuntimeError as exc:
                st.error(f"区域确认失败：{exc}")
        if actions[2].button("删除区域", key=f"delete_region_{selected_id}"):
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
        if actions[3].button("标记忽略", key=f"mark_region_{selected_id}"):
            try:
                document_result = _apply_document_edits(
                    document_result,
                    backend,
                    [{"action": "mark", "region_id": selected_id, "region_type": "ignore", "confirmed": True}],
                    rerun_ocsr=False,
                )
                st.session_state["document_result"] = document_result
                st.success("区域已标记为忽略。")
            except RuntimeError as exc:
                st.error(f"区域标记失败：{exc}")

        page_numbers = sorted({int(region["page_number"]) for region in active})
        page_number_for_batch = st.selectbox("批量确认页码", page_numbers, key="confirm_page_number")
        batch_actions = st.columns(2)
        if batch_actions[0].button("确认本页全部区域", key=f"confirm_page_{page_number_for_batch}"):
            try:
                document_result = _apply_document_edits(
                    document_result,
                    backend,
                    [{"action": "confirm_page", "page_number": page_number_for_batch, "note": "页面批量确认。"}],
                    rerun_ocsr=False,
                )
                st.session_state["document_result"] = document_result
                st.success("本页区域已批量确认。")
            except RuntimeError as exc:
                st.error(f"本页确认失败：{exc}")
        if batch_actions[1].button("确认本页并识别分子", key=f"confirm_page_ocsr_{page_number_for_batch}"):
            try:
                document_result = _apply_document_edits(
                    document_result,
                    backend,
                    [{"action": "confirm_page", "page_number": page_number_for_batch, "note": "页面批量确认并识别。"}],
                    rerun_ocsr=True,
                )
                st.session_state["document_result"] = document_result
                st.success("本页区域已确认，已确认分子区域已识别。")
            except RuntimeError as exc:
                st.error(f"本页识别失败：{exc}")

        with st.expander("合并两个或多个区域", expanded=False):
            merge_ids = st.multiselect("待合并区域", region_ids, default=[selected_id], key="merge_region_ids")
            merge_type = st.selectbox(
                "合并后类型",
                AUDIT_REGION_TYPES,
                index=AUDIT_REGION_TYPES.index(current),
                format_func=lambda value: REGION_TYPE_LABELS[value],
                key="merge_region_type",
            )
            merge_confirmed = st.checkbox("合并后立即确认", value=False, key="merge_confirmed")
            if st.button("合并区域", key="merge_regions", disabled=len(merge_ids) < 2):
                try:
                    document_result = _apply_document_edits(
                        document_result,
                        backend,
                        [{
                            "action": "merge",
                            "region_ids": merge_ids,
                            "region_type": merge_type,
                            "confirmed": merge_confirmed,
                        }],
                        rerun_ocsr=merge_confirmed and merge_type == "molecule",
                    )
                    st.session_state["document_result"] = document_result
                    st.success("区域已合并。")
                except RuntimeError as exc:
                    st.error(f"区域合并失败：{exc}")

        with st.expander("拆分当前区域", expanded=False):
            split_direction_label = st.radio("拆分方向", ["左右拆分", "上下拆分"], horizontal=True, key=f"split_direction_{selected_id}")
            split_ratio = st.slider("拆分位置", 0.1, 0.9, 0.5, 0.05, key=f"split_ratio_{selected_id}")
            split_confirmed = st.checkbox("拆分后立即确认", value=False, key=f"split_confirmed_{selected_id}")
            if st.button("拆分区域", key=f"split_region_{selected_id}"):
                try:
                    document_result = _apply_document_edits(
                        document_result,
                        backend,
                        [{
                            "action": "split",
                            "region_id": selected_id,
                            "direction": "vertical" if split_direction_label == "左右拆分" else "horizontal",
                            "split_at": float(split_ratio),
                            "region_type": region_type,
                            "confirmed": split_confirmed,
                        }],
                        rerun_ocsr=split_confirmed and region_type == "molecule",
                    )
                    st.session_state["document_result"] = document_result
                    st.success("区域已拆分。")
                except RuntimeError as exc:
                    st.error(f"区域拆分失败：{exc}")

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
            AUDIT_REGION_TYPES,
            format_func=lambda value: REGION_TYPE_LABELS[value],
            key="add_type",
        )
        add_confirmed = st.checkbox("添加后立即确认", value=True, key="add_confirmed")
        if st.button("添加区域", key="add_region"):
            try:
                document_result = _apply_document_edits(
                    document_result,
                    backend,
                    [{
                        "action": "add",
                        "page_number": page_number,
                        "bbox": [add_x1, add_y1, add_x2, add_y2],
                        "region_type": add_type,
                        "confirmed": add_confirmed,
                    }],
                    rerun_ocsr=add_confirmed and add_type == "molecule",
                )
                st.session_state["document_result"] = document_result
                st.success("区域已添加。")
            except RuntimeError as exc:
                st.error(f"区域添加失败：{exc}")
    return document_result


def _audit_region_type(region_type: str | None) -> str:
    value = str(region_type or "ignore")
    if value in AUDIT_REGION_TYPES:
        return value
    if value in {"reaction_arrow", "reaction_condition", "reaction_like"}:
        return "reaction"
    if value in {"text", "table", "molecule"}:
        return value
    return "ignore"


def _page_for_region(document_result: dict, region: dict) -> dict | None:
    page_number = int(region.get("page_number", 0))
    return next((page for page in document_result.get("pages", []) if int(page.get("page_number", 0)) == page_number), None)


def _consume_document_bbox_query(region_id: str) -> None:
    params = st.query_params
    if params.get("document_region_editor_key") != region_id:
        return
    try:
        nonce = str(params.get("doc_bbox_nonce") or "")
        if not nonce or st.session_state.get(f"doc_bbox_nonce_{region_id}") == nonce:
            return
        values = {}
        for coord in ("x1", "y1", "x2", "y2"):
            values[coord] = int(float(str(params.get(f"doc_bbox_{coord}") or "0")))
        for coord, value in values.items():
            st.session_state[f"edit_{coord}_{region_id}"] = max(0, int(value))
        st.session_state[f"doc_bbox_nonce_{region_id}"] = nonce
    finally:
        for key in DOCUMENT_BBOX_QUERY_KEYS:
            try:
                if key in params:
                    del params[key]
            except KeyError:
                continue


def _render_bbox_dragger(page: dict, regions: list[dict], selected_id: str, bbox: list[int]) -> None:
    image_path = Path(str(page.get("image_path") or ""))
    if not image_path.is_file():
        st.warning(f"页面图片不存在：{image_path}")
        return
    width = int(page.get("width") or 1)
    height = int(page.get("height") or 1)
    display_width = min(860, width)
    display_height = max(160, int(display_width * height / max(width, 1)))
    mime = "image/jpeg" if image_path.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    overlay_regions = [
        {
            "region_id": region.get("region_id"),
            "bbox": region.get("bbox"),
            "region_type": region.get("region_type"),
            "confirmed": bool(region.get("confirmed")),
        }
        for region in regions
        if int(region.get("page_number", 0)) == int(page.get("page_number", 0))
    ]
    payload = {
        "key": selected_id,
        "src": f"data:{mime};base64,{encoded}",
        "width": width,
        "height": height,
        "bbox": [int(value) for value in bbox],
        "regions": overlay_regions,
    }
    html = f"""
    <div style="font: 14px system-ui, sans-serif; color: #163232;">
      <div style="margin-bottom: 6px;">拖动选中框移动区域，拖动角点缩放；松手后坐标会回填到下方输入框。</div>
      <div style="position: relative; display: inline-block; max-width: 100%;">
        <img id="doc-region-image" src="{payload['src']}" style="width: min(100%, {display_width}px); display: block; cursor: crosshair; border: 1px solid #9ab8b8; border-radius: 6px;" />
        <canvas id="doc-region-overlay" style="position:absolute; inset:0;"></canvas>
      </div>
    </div>
    <script>
      const payload = {json.dumps(payload, ensure_ascii=False)};
      const image = document.getElementById("doc-region-image");
      const canvas = document.getElementById("doc-region-overlay");
      const ctx = canvas.getContext("2d");
      let bbox = payload.bbox.slice();
      let drag = null;

      function scale() {{
        return {{ sx: canvas.width / payload.width, sy: canvas.height / payload.height }};
      }}
      function clampBox(box) {{
        box[0] = Math.max(0, Math.min(payload.width - 1, Math.round(box[0])));
        box[1] = Math.max(0, Math.min(payload.height - 1, Math.round(box[1])));
        box[2] = Math.max(1, Math.min(payload.width, Math.round(box[2])));
        box[3] = Math.max(1, Math.min(payload.height, Math.round(box[3])));
        if (box[2] <= box[0]) box[2] = Math.min(payload.width, box[0] + 1);
        if (box[3] <= box[1]) box[3] = Math.min(payload.height, box[1] + 1);
        return box;
      }}
      function point(event) {{
        const rect = image.getBoundingClientRect();
        return {{
          x: (event.clientX - rect.left) * payload.width / rect.width,
          y: (event.clientY - rect.top) * payload.height / rect.height,
        }};
      }}
      function hitHandle(p) {{
        const handles = [
          ["nw", bbox[0], bbox[1]], ["ne", bbox[2], bbox[1]],
          ["sw", bbox[0], bbox[3]], ["se", bbox[2], bbox[3]],
        ];
        for (const [name, x, y] of handles) {{
          if (Math.abs(p.x - x) <= 14 && Math.abs(p.y - y) <= 14) return name;
        }}
        if (p.x >= bbox[0] && p.x <= bbox[2] && p.y >= bbox[1] && p.y <= bbox[3]) return "move";
        return null;
      }}
      function draw() {{
        const rect = image.getBoundingClientRect();
        canvas.width = Math.max(1, Math.round(rect.width));
        canvas.height = Math.max(1, Math.round(rect.height));
        canvas.style.width = rect.width + "px";
        canvas.style.height = rect.height + "px";
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        const s = scale();
        for (const region of payload.regions || []) {{
          const box = region.bbox || [];
          if (box.length !== 4) continue;
          const selected = region.region_id === payload.key;
          ctx.strokeStyle = selected ? "#0f766e" : (region.confirmed ? "#2563eb" : "#8a8f98");
          ctx.lineWidth = selected ? 3 : 1.5;
          ctx.setLineDash(selected ? [] : [5, 4]);
          ctx.strokeRect(box[0] * s.sx, box[1] * s.sy, (box[2] - box[0]) * s.sx, (box[3] - box[1]) * s.sy);
          ctx.setLineDash([]);
        }}
        ctx.fillStyle = "rgba(15, 118, 110, 0.13)";
        ctx.strokeStyle = "#0f766e";
        ctx.lineWidth = 3;
        ctx.fillRect(bbox[0] * s.sx, bbox[1] * s.sy, (bbox[2] - bbox[0]) * s.sx, (bbox[3] - bbox[1]) * s.sy);
        ctx.strokeRect(bbox[0] * s.sx, bbox[1] * s.sy, (bbox[2] - bbox[0]) * s.sx, (bbox[3] - bbox[1]) * s.sy);
        for (const [x, y] of [[bbox[0], bbox[1]], [bbox[2], bbox[1]], [bbox[0], bbox[3]], [bbox[2], bbox[3]]]) {{
          ctx.fillStyle = "#0f766e";
          ctx.fillRect(x * s.sx - 5, y * s.sy - 5, 10, 10);
          ctx.strokeStyle = "white";
          ctx.lineWidth = 2;
          ctx.strokeRect(x * s.sx - 5, y * s.sy - 5, 10, 10);
        }}
      }}
      function submitBox() {{
        const params = new URLSearchParams(window.top.location.search);
        params.set("document_region_editor_key", payload.key);
        params.set("doc_bbox_x1", String(bbox[0]));
        params.set("doc_bbox_y1", String(bbox[1]));
        params.set("doc_bbox_x2", String(bbox[2]));
        params.set("doc_bbox_y2", String(bbox[3]));
        params.set("doc_bbox_nonce", String(Date.now()));
        window.top.location.href = window.top.location.pathname + "?" + params.toString();
      }}
      canvas.addEventListener("mousedown", (event) => {{
        const p = point(event);
        const mode = hitHandle(p);
        if (!mode) return;
        drag = {{ mode, start: p, bbox: bbox.slice() }};
        event.preventDefault();
      }});
      canvas.addEventListener("mousemove", (event) => {{
        if (!drag) return;
        const p = point(event);
        const dx = p.x - drag.start.x;
        const dy = p.y - drag.start.y;
        const next = drag.bbox.slice();
        if (drag.mode === "move") {{
          next[0] += dx; next[2] += dx; next[1] += dy; next[3] += dy;
          const w = next[2] - next[0], h = next[3] - next[1];
          if (next[0] < 0) {{ next[0] = 0; next[2] = w; }}
          if (next[1] < 0) {{ next[1] = 0; next[3] = h; }}
          if (next[2] > payload.width) {{ next[2] = payload.width; next[0] = payload.width - w; }}
          if (next[3] > payload.height) {{ next[3] = payload.height; next[1] = payload.height - h; }}
        }} else {{
          if (drag.mode.includes("w")) next[0] += dx;
          if (drag.mode.includes("e")) next[2] += dx;
          if (drag.mode.includes("n")) next[1] += dy;
          if (drag.mode.includes("s")) next[3] += dy;
        }}
        bbox = clampBox(next);
        draw();
      }});
      window.addEventListener("mouseup", () => {{
        if (!drag) return;
        drag = null;
        submitBox();
      }});
      image.addEventListener("load", draw);
      window.addEventListener("resize", draw);
      draw();
    </script>
    """
    components.html(html, height=display_height + 48)


def _download_panel(document_result: dict) -> None:
    exports = document_result.get("exports") or {}
    st.subheader("结果导出")
    json_path = Path(exports.get("json") or "")
    csv_path = Path(exports.get("regions_csv") or "")
    detection_annotations_path = Path(exports.get("detection_annotations_json") or "")
    zip_path = Path(exports.get("zip") or "")
    if json_path.is_file():
        st.download_button("下载文档分析结果", json_path.read_bytes(), "document_result.json", "application/json")
    if csv_path.is_file():
        st.download_button("下载区域结果表", csv_path.read_bytes(), "regions.csv", "text/csv")
    if detection_annotations_path.is_file():
        st.download_button(
            "下载检测训练标注",
            detection_annotations_path.read_bytes(),
            "detection_annotations.json",
            "application/json",
        )
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

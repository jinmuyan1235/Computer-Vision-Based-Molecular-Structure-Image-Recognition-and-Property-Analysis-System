"""PDF and multi-molecule document page."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

import streamlit as st

import config
from src.documents.input_loader import DocumentInputError, OptionalDependencyError
from src.documents.processor import DocumentOCSRProcessor
from src.ui.image_viewer import show_document_page
from src.ui.labels import REGION_TYPE_LABELS, localize_region_rows
from src.ui.records import render_records
from src.ui.state import current_runtime_key, get_document_processor, remember_backend_status, runtime_config_from_key
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
            st.session_state["document_result"] = get_document_processor(backend).process(
                temporary_path,
                run_ocsr=run_ocsr,
                progress_callback=progress_callback if run_ocsr else None,
            )
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
    with stdout_path.open("w", encoding="utf-8") as stdout_file, stderr_path.open("w", encoding="utf-8") as stderr_file:
        process = subprocess.Popen(
            _document_command(input_path, backend, run_ocsr),
            cwd=PROJECT_ROOT,
            env=env,
            stdout=stdout_file,
            stderr=stderr_file,
            text=True,
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
    process: subprocess.Popen = job["process"]
    elapsed = time.time() - float(job.get("started_at", time.time()))
    return_code = process.poll()
    if return_code is None:
        st.info(f"文档处理正在后台运行，已耗时 {elapsed:.1f} 秒。页面会自动刷新，期间不会阻塞 Streamlit。")
        st.caption(f"后台日志：{job.get('stdout_path')} / {job.get('stderr_path')}")
        st.progress(min(elapsed / 120.0, 0.95))
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
    st.session_state["document_result"] = json.loads(result_path.read_text(encoding="utf-8"))
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
    completed = subprocess.run(command, cwd=PROJECT_ROOT, env=env, capture_output=True, text=True, timeout=900)
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
            "message",
            "final_smiles",
            "valid",
            "inference_time_ms",
            "processing_time_ms",
            "screening_reason",
        ]
        display_rows = [{key: row.get(key) for key in important} for row in rows]
        render_records(localize_region_rows(display_rows), title_keys=("区域 ID", "页码"))
        if st.checkbox("显示完整字段", value=False, key="show_document_full_table"):
            render_records(localize_region_rows(rows), title_keys=("区域 ID", "页码"), max_records=100)

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
    st.subheader("结果导出")
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

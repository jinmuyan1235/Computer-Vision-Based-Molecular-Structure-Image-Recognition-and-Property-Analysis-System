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
import cv2
import numpy as np

import config
from src.documents.input_loader import DocumentInputError, OptionalDependencyError
from src.documents.processor import DocumentOCSRProcessor
from src.documents.region_review import (
    apply_canvas_event,
    background_failure_reason,
    canvas_event_from_query,
    persist_document_result_atomic,
    save_region_selection,
)
from src.storage.analysis_repository import record_result_payload
from src.ui.image_viewer import show_document_page
from src.ui.labels import REGION_TYPE_LABELS, localize_region_rows, region_type_label, status_label
from src.ui.records import render_records
from src.ui.state import current_runtime_key, get_document_processor, remember_backend_status, runtime_config_from_key
from src.ui.styles import page_intro
from src.runtime.job_manager import (
    extract_json_object,
    run_json_command,
    start_logged_process,
    terminate_process_tree,
)

DOCUMENT_WORKFLOW_LABEL = "全文检测与审核识别"
DOCUMENT_PROGRESS_MARKER = "DOCUMENT_PROGRESS_JSON="
DOCUMENT_RESULT_MARKER = "DOCUMENT_RESULT_JSON="
PROJECT_ROOT = Path(__file__).resolve().parents[2]
AUDIT_REGION_TYPES = ["molecule", "text", "table", "reaction", "ignore"]
DOCUMENT_BBOX_QUERY_KEYS = (
    "document_region_editor_key",
    "doc_bbox_action",
    "doc_bbox_region_id",
    "doc_bbox_page",
    "doc_bbox_x1",
    "doc_bbox_y1",
    "doc_bbox_x2",
    "doc_bbox_y2",
    "doc_canvas_width",
    "doc_canvas_height",
    "doc_bbox_nonce",
)


def render_document_page(backend: str) -> None:
    page_intro("PDF/多分子文档", "一个入口处理整篇论文：全文分页、区域检测、人工审核和后台 OCSR 识别。")
    upload = st.file_uploader(
        "上传 PDF / 页面图片 / ZIP",
        type=["pdf", "png", "jpg", "jpeg", "zip"],
        key="document_upload",
    )
    st.info(
        "综合流程：先检测整篇文档的分子候选框，再在同一审核台中修正，并在确认后启动识别；"
        "新检测框不会在人工确认前自动送入 OCSR。"
    )
    st.caption(
        f"当前支持一次处理最多 {config.DOCUMENT_MAX_PAGES} 页、{config.DOCUMENT_MAX_REGIONS} 个区域、"
        f"{config.DOCUMENT_MAX_FILE_SIZE_MB:g} MB 文件；上限均可通过环境变量调整。"
    )

    if upload is not None and st.button(f"开始{DOCUMENT_WORKFLOW_LABEL}", type="primary", key="process_document"):
        try:
            if backend != "demo":
                input_path = _persist_uploaded_document(upload)
                _start_document_job(input_path, backend, False)
                st.session_state.pop("document_result", None)
                st.rerun()
            else:
                _run_demo_document(upload, backend, False)
        except OptionalDependencyError as exc:
            st.error(str(exc))
        except (DocumentInputError, FileNotFoundError, RuntimeError, ValueError) as exc:
            st.error(str(exc))

    _render_document_job_status()
    if "document_result" in st.session_state:
        st.session_state["document_result"] = show_document_result(st.session_state["document_result"], backend)
    elif not st.session_state.get("document_job"):
        _render_document_recovery()


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

        def document_progress(stage: str, current: int, total: int, detail: str) -> None:
            progress_text.info(detail)
            if stage == "rendered":
                progress_bar.progress(0.2)
            elif stage == "detecting":
                progress_bar.progress(0.2 + 0.75 * current / max(total, 1))

        with st.spinner("正在渲染页面并检测区域……"):
            result = get_document_processor(backend).process(
                temporary_path,
                run_ocsr=run_ocsr,
                progress_callback=progress_callback if run_ocsr else None,
                document_progress_callback=document_progress,
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
        stdout_path = Path(str(job.get("stdout_path") or ""))
        stdout = stdout_path.read_text(encoding="utf-8", errors="replace") if stdout_path.is_file() else ""
        progress = _extract_document_progress(stdout)
        if progress:
            current = int(progress.get("current") or 0)
            total = max(1, int(progress.get("total") or 1))
            stage = str(progress.get("stage") or "")
            if stage == "rendered":
                fraction = 0.2
            elif stage == "detecting":
                fraction = 0.2 + 0.75 * current / total
            else:
                fraction = min(0.95, current / total)
            detail = str(progress.get("detail") or f"{current}/{total}")
            st.info(f"{DOCUMENT_WORKFLOW_LABEL}正在后台运行：{detail}")
            st.progress(min(0.95, fraction))
        else:
            st.info(f"正在渲染整篇文档，已耗时 {elapsed:.1f} 秒。完成分页后将自动开始逐页区域检测。")
            st.progress(min(0.15, elapsed / 180.0))
        st.caption("页面会自动刷新；全文任务在后台运行，不会让 Streamlit 无反馈等待。")
        st.caption(f"后台日志：{job.get('stdout_path')} / {job.get('stderr_path')}")
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
    output_dir = Path(document_result.get("output_dir") or tempfile.gettempdir()).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    current_path = output_dir / "document_result_for_edit.json"
    current_path.write_text(json.dumps(document_result, ensure_ascii=False, indent=2), encoding="utf-8")
    command = _document_edit_command(current_path, backend, edits, rerun_ocsr)

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


def _document_edit_command(current_path: Path, backend: str, edits: list[dict], rerun_ocsr: bool) -> list[str]:
    runtime = runtime_config_from_key(current_runtime_key())
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
    return command


def _start_region_ocsr_job(document_result: dict, backend: str, region_id: str, bbox: list[int]) -> dict:
    current_job = st.session_state.get("document_region_job")
    if current_job and current_job.get("process") and current_job["process"].poll() is None:
        raise RuntimeError("已有区域识别任务正在运行，请等待其完成。")
    saved = save_region_selection(document_result, region_id, bbox, recognize=True)
    current_path = persist_document_result_atomic(saved)
    record_result_payload(saved, current_path)
    selected = next(region for region in saved.get("regions", []) if str(region.get("region_id")) == str(region_id))
    job_dir = config.OUTPUT_DIR / "ui_jobs" / f"region_{uuid4().hex}"
    job_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = job_dir / "stdout.log"
    stderr_path = job_dir / "stderr.log"
    env = os.environ.copy()
    env.setdefault("MOLSCRIBE_ISOLATED_SUBPROCESS", "true")
    env.setdefault("DECIMER_ISOLATED_SUBPROCESS", "true")
    process = start_logged_process(
        _document_edit_command(
            current_path,
            backend,
            [{"action": "recognize", "region_id": region_id, "note": "Background OCSR requested by reviewer."}],
            True,
        ),
        cwd=PROJECT_ROOT,
        env=env,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )
    st.session_state["document_result"] = saved
    st.session_state["document_region_job"] = {
        "process": process,
        "region_id": str(region_id),
        "page_number": int(selected.get("page_number") or 0),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "started_at": time.time(),
    }
    return saved


def _start_confirmed_region_batch_job(
    document_result: dict,
    backend: str,
    region_ids: list[str],
    *,
    scope_label: str,
    page_number: int,
) -> dict:
    current_job = st.session_state.get("document_region_job")
    if current_job and current_job.get("process") and current_job["process"].poll() is None:
        raise RuntimeError("已有区域识别任务正在运行，请等待其完成。")
    requested = {str(region_id) for region_id in region_ids}
    targets = [
        region for region in document_result.get("regions", [])
        if str(region.get("region_id")) in requested
        and region.get("status") != "deleted"
        and region.get("region_type") == "molecule"
        and bool(region.get("confirmed"))
    ]
    if not targets:
        raise ValueError("当前范围没有已确认的分子区域。请先逐个核对并保存结构框。")
    target_ids = [str(region.get("region_id")) for region in targets]
    current_path = persist_document_result_atomic(document_result)
    record_result_payload(document_result, current_path)
    job_dir = config.OUTPUT_DIR / "ui_jobs" / f"region_batch_{uuid4().hex}"
    job_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = job_dir / "stdout.log"
    stderr_path = job_dir / "stderr.log"
    env = os.environ.copy()
    env.setdefault("MOLSCRIBE_ISOLATED_SUBPROCESS", "true")
    env.setdefault("DECIMER_ISOLATED_SUBPROCESS", "true")
    process = start_logged_process(
        _document_edit_command(
            current_path,
            backend,
            [
                {"action": "recognize", "region_id": region_id, "note": f"{scope_label} background OCSR."}
                for region_id in target_ids
            ],
            True,
        ),
        cwd=PROJECT_ROOT,
        env=env,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )
    restore_id = str(st.session_state.get("document_region_select") or target_ids[0])
    st.session_state["document_region_job"] = {
        "process": process,
        "region_id": restore_id,
        "region_ids": target_ids,
        "scope_label": scope_label,
        "page_number": int(page_number),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "started_at": time.time(),
    }
    return document_result


@st.fragment(run_every="2s")
def _render_region_ocsr_job_status(inline_region_id: str | None = None) -> None:
    job = st.session_state.get("document_region_job")
    if not job:
        return
    process = job.get("process")
    if process is None:
        st.session_state.pop("document_region_job", None)
        return
    elapsed = time.time() - float(job.get("started_at", time.time()))
    return_code = process.poll()
    region_id = str(job.get("region_id") or "")
    region_ids = [str(value) for value in (job.get("region_ids") or [region_id])]
    scope_label = str(job.get("scope_label") or f"区域 {region_id}")
    if return_code is None:
        target_note = "当前区域" if len(region_ids) == 1 and str(inline_region_id or "") in region_ids else scope_label
        with st.status(f"{target_note}正在后台识别…", state="running", expanded=True):
            st.write(f"✅ 步骤 1/2：已保存并锁定 {len(region_ids)} 个已确认分子区域。")
            st.write("⏳ 步骤 2/2：正在加载 OCSR 并识别结构，请稍候。")
            st.progress(min(0.95, max(0.08, elapsed / 90.0)))
            st.caption(f"已运行 {elapsed:.1f} 秒。识别按钮已锁定，完成后会自动显示候选结构。")
        return

    stdout_path = Path(str(job.get("stdout_path") or ""))
    stderr_path = Path(str(job.get("stderr_path") or ""))
    stdout = stdout_path.read_text(encoding="utf-8", errors="replace") if stdout_path.is_file() else ""
    stderr = stderr_path.read_text(encoding="utf-8", errors="replace") if stderr_path.is_file() else ""
    payload = _extract_json_object(stdout)
    st.session_state.pop("document_region_job", None)
    st.session_state["document_region_restore"] = {
        "region_id": region_id,
        "page_number": str(job.get("page_number") or "全部"),
    }
    st.session_state["document_region_job_logs"] = {
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
    }
    if return_code != 0:
        reason = background_failure_reason(return_code, payload, stdout, stderr)
        st.session_state["document_region_notice"] = {"level": "error", "message": f"{scope_label}失败：{reason}"}
        st.rerun()
        return
    result_path_value = (payload or {}).get("result_path")
    result_path = Path(str(result_path_value or ""))
    if not result_path_value or not result_path.is_file():
        st.session_state["document_region_notice"] = {
            "level": "error",
            "message": f"{scope_label}完成，但结果文件缺失。请查看后台日志。",
        }
        st.rerun()
        return
    updated = json.loads(result_path.read_text(encoding="utf-8"))
    st.session_state["document_result"] = updated
    record_result_payload(updated, result_path)
    result_regions = [item for item in updated.get("regions", []) if str(item.get("region_id")) in set(region_ids)]
    recognized_count = sum(str(item.get("status")) == "recognized" for item in result_regions)
    if len(region_ids) > 1:
        failed_count = len(region_ids) - recognized_count
        message = f"{scope_label}完成：成功 {recognized_count} 个，未成功 {failed_count} 个。"
        level = "success" if recognized_count else "error"
    else:
        region = result_regions[0] if result_regions else {}
        if str(region.get("status")) == "recognized":
            message = f"区域 {region_id} 识别成功，候选 SMILES 和重绘结构已更新。"
            level = "success"
        else:
            message = f"区域 {region_id} 识别未成功：{region.get('message') or '模型未返回可用结构。'}"
            level = "error"
    st.session_state["document_region_notice"] = {"level": level, "message": message}
    st.rerun()


def _apply_document_edits(document_result: dict, backend: str, edits: list[dict], rerun_ocsr: bool) -> dict:
    if backend == "demo":
        processor = get_document_processor(backend)
        updated = processor.apply_edits(document_result, edits, rerun_ocsr=rerun_ocsr)
    else:
        updated = _apply_document_edits_subprocess(document_result, backend, edits, rerun_ocsr)
    record_result_payload(updated, (updated.get("exports") or {}).get("json"))
    return updated


def _extract_json_object(text: str) -> dict | None:
    marked = extract_json_object(text, marker=DOCUMENT_RESULT_MARKER)
    if marked is not None and marked.get("result_path"):
        return marked
    decoder = json.JSONDecoder()
    fallback: dict | None = None
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if not isinstance(value, dict):
            continue
        if value.get("result_path"):
            return value
        fallback = value
    return fallback or extract_json_object(text)


def _extract_document_progress(text: str) -> dict | None:
    for line in reversed(text.splitlines()):
        if not line.startswith(DOCUMENT_PROGRESS_MARKER):
            continue
        try:
            payload = json.loads(line[len(DOCUMENT_PROGRESS_MARKER) :])
        except json.JSONDecodeError:
            return None
        return payload if isinstance(payload, dict) else None
    return None


def _latest_recoverable_document_result(jobs_root: str | Path) -> tuple[dict, Path] | None:
    root = Path(jobs_root)
    if not root.is_dir():
        return None
    logs = sorted(root.rglob("stdout.log"), key=lambda path: path.stat().st_mtime, reverse=True)
    for log_path in logs[:20]:
        text = log_path.read_text(encoding="utf-8", errors="replace")
        payload = _extract_json_object(text)
        result_path_value = (payload or {}).get("result_path")
        if not result_path_value:
            continue
        result_path = Path(str(result_path_value))
        if result_path.is_file():
            return payload or {}, result_path
    return None


def _render_document_recovery() -> None:
    recoverable = _latest_recoverable_document_result(config.OUTPUT_DIR / "ui_jobs")
    if recoverable is None:
        return
    payload, result_path = recoverable
    summary = payload.get("summary") or {}
    with st.expander("恢复最近完成的全文任务", expanded=False):
        st.caption(
            f"检测到可恢复结果：{summary.get('page_count', '-')} 页、"
            f"{summary.get('region_count', '-')} 个区域。无需重新处理 PDF。"
        )
        if st.button("恢复到审核台", key="recover_latest_document_result"):
            result = json.loads(result_path.read_text(encoding="utf-8"))
            st.session_state["document_result"] = result
            record_result_payload(result, result_path)
            st.success("最近完成的全文任务已恢复。")
            st.rerun()


def show_document_result(document_result: dict, backend: str) -> dict:
    notice = st.session_state.pop("document_region_notice", None)
    if notice:
        renderer = st.success if notice.get("level") == "success" else st.error
        renderer(str(notice.get("message") or "区域任务状态已更新。"))
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
    document_result = _document_workbench(document_result, backend, rows)

    logs = st.session_state.get("document_job_logs") or {}
    if logs:
        st.caption(f"最近一次后台日志：{logs.get('stdout_path')} / {logs.get('stderr_path')}")

    return document_result


def _document_workbench(document_result: dict, backend: str, rows: list[dict]) -> dict:
    restore = st.session_state.pop("document_region_restore", None)
    if restore:
        st.session_state["document_region_select"] = str(restore.get("region_id") or "")
        st.session_state["document_current_page"] = int(restore.get("page_number") or 1)
    document_result = _consume_document_canvas_event(document_result)
    pages = sorted(document_result.get("pages") or [], key=lambda page: int(page.get("page_number", 0)))
    if not pages:
        st.warning("文档结果中没有可显示的页面。")
        return document_result
    active = [region for region in document_result.get("regions", []) if region.get("status") != "deleted"]
    page_numbers = [int(page.get("page_number", 0)) for page in pages]
    current_page = st.session_state.get("document_current_page")
    if current_page not in page_numbers:
        selected_region_id = str(st.session_state.get("document_region_select") or "")
        selected_region = next((region for region in active if str(region.get("region_id")) == selected_region_id), None)
        st.session_state["document_current_page"] = int((selected_region or {}).get("page_number") or page_numbers[0])
    page_number = st.selectbox(
        f"论文页码（全文共 {len(page_numbers)} 页）",
        page_numbers,
        key="document_current_page",
    )
    page = next(item for item in pages if int(item.get("page_number", 0)) == int(page_number))
    st.caption(f"正在审核第 {page_numbers.index(int(page_number)) + 1} / {len(page_numbers)} 页；所有页面均已保留在当前任务中。")
    page_regions = [region for region in active if int(region.get("page_number", 0)) == int(page_number)]

    show_non_molecule = st.checkbox(
        "显示文本、表格、反应等非分子区域",
        value=False,
        key="document_show_non_molecule_regions",
    )
    visible = page_regions if show_non_molecule else [region for region in page_regions if region.get("region_type") == "molecule"]
    if not visible:
        if page_regions:
            st.info("本页没有分子候选框。勾选上方选项可查看本页文本、表格等其他区域，也可直接拖画新框。")
        else:
            st.info("本页未检测到区域；仍可直接在页面上拖画新的分子候选框。")
        _render_region_ocsr_job_status(None)
        _render_document_toolbar(document_result, backend, rows, None)
        _render_bbox_dragger(page, [], "", [0, 0, 1, 1])
        return document_result

    selected_id = _selected_region_id(visible)
    selected = next((region for region in visible if region["region_id"] == selected_id), visible[0])
    document_result = _render_document_toolbar(document_result, backend, rows, selected)
    bbox = selected.get("bbox") or [0, 0, 1, 1]
    _sync_region_bbox_state(selected)

    region_list, canvas, inspector = st.columns([0.22, 0.53, 0.25])
    with region_list:
        selected = _render_region_list(visible, selected["region_id"])
        _sync_region_bbox_state(selected)
    with canvas:
        preview_bbox = [
            int(st.session_state.get(f"edit_x1_{selected['region_id']}", (selected.get("bbox") or [0, 0, 1, 1])[0])),
            int(st.session_state.get(f"edit_y1_{selected['region_id']}", (selected.get("bbox") or [0, 0, 1, 1])[1])),
            int(st.session_state.get(f"edit_x2_{selected['region_id']}", (selected.get("bbox") or [0, 0, 1, 1])[2])),
            int(st.session_state.get(f"edit_y2_{selected['region_id']}", (selected.get("bbox") or [0, 0, 1, 1])[3])),
        ]
        _render_bbox_dragger(page, visible, selected["region_id"], preview_bbox)
        _render_region_crop_preview(page, preview_bbox)
    with inspector:
        _render_region_inspector(document_result, backend, selected)
    return document_result


def _render_create_only_canvas(document_result: dict) -> None:
    pages = document_result.get("pages") or []
    if not pages:
        return
    page_numbers = [int(page.get("page_number", 0)) for page in pages]
    page_number = st.selectbox("创建区域的页码", page_numbers, key="document_empty_canvas_page")
    page = next(page for page in pages if int(page.get("page_number", 0)) == page_number)
    _render_bbox_dragger(page, [], "", [0, 0, 1, 1])


def _selected_region_id(active: list[dict]) -> str:
    active_ids = [str(region["region_id"]) for region in active]
    current = str(st.session_state.get("document_region_select") or active_ids[0])
    return current if current in active_ids else active_ids[0]


def _render_region_list(active: list[dict], selected_id: str) -> dict:
    st.subheader("区域列表")
    type_options = ["全部", *AUDIT_REGION_TYPES]
    selected_type = st.selectbox(
        "类型",
        type_options,
        key="document_filter_type",
        format_func=lambda value: "全部" if value == "全部" else REGION_TYPE_LABELS.get(value, value),
    )
    status_options = ["全部", "待确认", "已确认", "识别成功", "识别失败", "已跳过"]
    selected_status = st.selectbox("状态", status_options, key="document_filter_status")
    filtered = [
        region
        for region in active
        if (selected_type == "全部" or _audit_region_type(region.get("region_type")) == selected_type)
        and _region_matches_status(region, selected_status)
    ]
    if not filtered:
        st.info("没有匹配区域。")
        filtered = active
    ids = [str(region["region_id"]) for region in filtered]
    if selected_id not in ids:
        selected_id = ids[0]
    if str(st.session_state.get("document_region_select") or "") not in ids:
        st.session_state["document_region_select"] = selected_id
    choice = st.radio(
        "当前区域",
        ids,
        index=ids.index(selected_id),
        key="document_region_select",
        format_func=lambda region_id: _region_option_label(next(region for region in filtered if str(region["region_id"]) == region_id)),
        label_visibility="collapsed",
    )
    return next(region for region in filtered if str(region["region_id"]) == str(choice))


def _region_matches_status(region: dict, selected_status: str) -> bool:
    if selected_status == "全部":
        return True
    if selected_status == "待确认":
        return not bool(region.get("confirmed"))
    if selected_status == "已确认":
        return bool(region.get("confirmed"))
    mapping = {
        "识别成功": "recognized",
        "识别失败": "failed",
        "已跳过": "skipped",
    }
    return str(region.get("status") or "") == mapping.get(selected_status)


def _region_option_label(region: dict) -> str:
    status = "已确认" if region.get("confirmed") else status_label(region.get("status") or "detected")
    return (
        f"第 {region.get('page_number')} 页 · "
        f"{region.get('region_id')} · "
        f"{region_type_label(region.get('region_type'))} · {status}"
    )


def _render_region_inspector(document_result: dict, backend: str, selected: dict) -> None:
    selected_id = str(selected["region_id"])
    st.subheader("当前区域")
    st.caption(f"页码：{selected.get('page_number')}；ID：{selected_id}")
    st.write(f"**类型：** {region_type_label(selected.get('region_type'))}")
    st.write(f"**状态：** {status_label(selected.get('status'))}")
    st.write(f"**置信度：** {selected.get('detection_confidence') if selected.get('detection_confidence') is not None else '-'}")
    candidate_smiles = _region_candidate_smiles(selected)
    if candidate_smiles:
        st.write("**候选 SMILES：**")
        st.code(candidate_smiles, language=None)
        redrawn = str((((selected.get("report") or {}).get("images") or {}).get("redrawn_molecule")) or "")
        if redrawn and Path(redrawn).is_file():
            st.image(redrawn, caption="候选结构重绘", width=300)
    elif str(selected.get("status") or "") == "failed":
        st.error(f"识别失败：{selected.get('message') or '模型未返回可用结构。'}")

    bbox = selected.get("bbox") or [0, 0, 1, 1]
    with st.popover("高级信息（坐标与类型）"):
        first_row = st.columns(2)
        x1 = first_row[0].number_input("x1", min_value=0, value=int(st.session_state.get(f"edit_x1_{selected_id}", bbox[0])), key=f"edit_x1_{selected_id}")
        y1 = first_row[1].number_input("y1", min_value=0, value=int(st.session_state.get(f"edit_y1_{selected_id}", bbox[1])), key=f"edit_y1_{selected_id}")
        second_row = st.columns(2)
        x2 = second_row[0].number_input("x2", min_value=1, value=int(st.session_state.get(f"edit_x2_{selected_id}", bbox[2])), key=f"edit_x2_{selected_id}")
        y2 = second_row[1].number_input("y2", min_value=1, value=int(st.session_state.get(f"edit_y2_{selected_id}", bbox[3])), key=f"edit_y2_{selected_id}")
        allowed = list(AUDIT_REGION_TYPES)
        current = _audit_region_type(selected.get("region_type"))
        region_type = st.selectbox(
            "区域类型",
            allowed,
            index=allowed.index(current),
            format_func=lambda value: REGION_TYPE_LABELS[value],
            key=f"edit_type_{selected_id}",
        )
    job_running = _region_job_is_running()
    st.caption("“仅保存框选”不会运行模型；“保存并识别”会将类型设为分子并确认后启动后台 OCSR。")
    _render_region_ocsr_job_status(selected_id)
    actions = st.columns(2)
    if actions[0].button("仅保存框选", key=f"save_region_{selected_id}", disabled=job_running):
        try:
            updated = save_region_selection(
                document_result,
                selected_id,
                [x1, y1, x2, y2],
                recognize=False,
                region_type=region_type,
            )
            result_path = persist_document_result_atomic(updated)
            record_result_payload(updated, result_path)
            st.session_state["document_result"] = updated
            st.session_state["document_region_select"] = selected_id
            st.session_state["document_region_notice"] = {"level": "success", "message": "框选已保存，尚未启动识别。"}
            st.rerun()
        except (OSError, RuntimeError, ValueError) as exc:
            st.error(f"框选保存失败：{exc}")
    if actions[1].button("保存并识别", key=f"recognize_region_{selected_id}", type="primary", disabled=job_running):
        try:
            _start_region_ocsr_job(document_result, backend, selected_id, [x1, y1, x2, y2])
            st.session_state["document_region_select"] = selected_id
            st.rerun()
        except (OSError, RuntimeError, ValueError) as exc:
            st.error(f"后台识别启动失败：{exc}")
    minor = st.columns(2)
    if minor[0].button("标记忽略", key=f"mark_region_{selected_id}", disabled=job_running):
        _apply_edits_with_notice(
            document_result,
            backend,
            [{"action": "mark", "region_id": selected_id, "region_type": "ignore", "confirmed": True}],
            rerun_ocsr=False,
            message="区域已标记为忽略。",
        )
    with minor[1].popover("删除"):
        st.warning("删除后该区域会从当前工作台中移除。")
        if st.button("确认删除", key=f"delete_region_confirm_{selected_id}", disabled=job_running):
            _apply_edits_with_notice(
                document_result,
                backend,
                [{"action": "delete", "region_id": selected_id, "note": "用户在界面删除区域。"}],
                rerun_ocsr=False,
                message="区域已删除。",
            )


def _region_candidate_smiles(region: dict) -> str | None:
    report = region.get("report") or {}
    final = report.get("final") or region.get("final_result") or {}
    validation = report.get("validation") or {}
    ocsr = report.get("ocsr") or region.get("ocsr") or {}
    value = (
        region.get("final_smiles")
        or final.get("smiles")
        or final.get("canonical_smiles")
        or validation.get("canonical_smiles")
        or ocsr.get("smiles")
    )
    return str(value) if value else None


def _region_job_is_running() -> bool:
    job = st.session_state.get("document_region_job") or {}
    process = job.get("process")
    return bool(process is not None and process.poll() is None)


def _render_document_toolbar(document_result: dict, backend: str, rows: list[dict], selected: dict | None) -> dict:
    toolbar = st.columns([0.12, 0.12, 0.14, 0.16, 0.12, 0.34])
    with toolbar[0].popover("新增区域"):
        st.caption("请直接在页面空白处按住鼠标并拖动创建分子框。")
    with toolbar[1].popover("合并"):
        _render_merge_popover(document_result, backend, selected)
    with toolbar[2].popover("拆分"):
        _render_split_popover(document_result, backend, selected)
    with toolbar[3].popover("批量操作"):
        _render_batch_region_popover(document_result, backend)
    with toolbar[4].popover("导出"):
        _download_panel(document_result)
    with toolbar[5].popover("结果表"):
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
            max_records=100,
        )
    return document_result


def _render_add_region_popover(document_result: dict, backend: str) -> None:
    page_numbers = [int(page["page_number"]) for page in document_result.get("pages", [])]
    if not page_numbers:
        st.info("没有可添加区域的页面。")
        return
    page_number = st.selectbox("页码", page_numbers, key="add_region_page")
    row1 = st.columns(2)
    add_x1 = row1[0].number_input("新区域 x1", min_value=0, value=0, key="add_x1")
    add_y1 = row1[1].number_input("新区域 y1", min_value=0, value=0, key="add_y1")
    row2 = st.columns(2)
    add_x2 = row2[0].number_input("新区域 x2", min_value=1, value=200, key="add_x2")
    add_y2 = row2[1].number_input("新区域 y2", min_value=1, value=200, key="add_y2")
    add_type = st.selectbox("新区域类型", AUDIT_REGION_TYPES, format_func=lambda value: REGION_TYPE_LABELS[value], key="add_type")
    add_confirmed = st.checkbox("添加后立即确认", value=True, key="add_confirmed")
    if st.button("添加区域", key="add_region"):
        _apply_edits_with_notice(
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
            message="区域已添加。",
        )


def _render_merge_popover(document_result: dict, backend: str, selected: dict | None) -> None:
    active = [region for region in document_result.get("regions", []) if region.get("status") != "deleted"]
    if len(active) < 2:
        st.info("至少需要两个区域才能合并。")
        return
    region_ids = [str(region["region_id"]) for region in active]
    default = [str(selected["region_id"])] if selected else []
    merge_ids = st.multiselect("待合并区域", region_ids, default=default, key="merge_region_ids")
    current = _audit_region_type((selected or {}).get("region_type"))
    merge_type = st.selectbox(
        "合并后类型",
        AUDIT_REGION_TYPES,
        index=AUDIT_REGION_TYPES.index(current),
        format_func=lambda value: REGION_TYPE_LABELS[value],
        key="merge_region_type",
    )
    merge_confirmed = st.checkbox("合并后立即确认", value=False, key="merge_confirmed")
    if st.button("合并区域", key="merge_regions", disabled=len(merge_ids) < 2):
        _apply_edits_with_notice(
            document_result,
            backend,
            [{
                "action": "merge",
                "region_ids": merge_ids,
                "region_type": merge_type,
                "confirmed": merge_confirmed,
            }],
            rerun_ocsr=merge_confirmed and merge_type == "molecule",
            message="区域已合并。",
        )


def _render_split_popover(document_result: dict, backend: str, selected: dict | None) -> None:
    if not selected:
        st.info("请先选择一个区域。")
        return
    selected_id = str(selected["region_id"])
    split_direction_label = st.radio("拆分方向", ["左右拆分", "上下拆分"], horizontal=True, key=f"split_direction_{selected_id}")
    split_ratio = st.slider("拆分位置", 0.1, 0.9, 0.5, 0.05, key=f"split_ratio_{selected_id}")
    current = _audit_region_type(selected.get("region_type"))
    split_confirmed = st.checkbox("拆分后立即确认", value=False, key=f"split_confirmed_{selected_id}")
    if st.button("拆分区域", key=f"split_region_{selected_id}"):
        _apply_edits_with_notice(
            document_result,
            backend,
            [{
                "action": "split",
                "region_id": selected_id,
                "direction": "vertical" if split_direction_label == "左右拆分" else "horizontal",
                "split_at": float(split_ratio),
                "region_type": current,
                "confirmed": split_confirmed,
            }],
            rerun_ocsr=split_confirmed and current == "molecule",
            message="区域已拆分。",
        )


def _render_batch_region_popover(document_result: dict, backend: str) -> None:
    active = [region for region in document_result.get("regions", []) if region.get("status") != "deleted"]
    page_numbers = sorted({int(page.get("page_number", 0)) for page in document_result.get("pages", [])})
    if not page_numbers:
        st.info("没有可批量操作的区域。")
        return
    preferred_page = int(st.session_state.get("document_current_page") or page_numbers[0])
    if preferred_page not in page_numbers:
        preferred_page = page_numbers[0]
    page_number_for_batch = st.selectbox(
        "批量识别页码",
        page_numbers,
        index=page_numbers.index(preferred_page),
        key="recognize_page_number",
    )
    confirmed_molecules = [
        region for region in active
        if region.get("region_type") == "molecule" and bool(region.get("confirmed"))
    ]
    page_ids = [
        str(region.get("region_id"))
        for region in confirmed_molecules
        if int(region.get("page_number", 0)) == int(page_number_for_batch)
    ]
    all_ids = [str(region.get("region_id")) for region in confirmed_molecules]
    st.caption(f"本页已确认分子：{len(page_ids)}；全文已确认分子：{len(all_ids)}。未确认框不会进入 OCSR。")
    running = _region_job_is_running()
    if st.button(
        "识别本页已确认区域",
        key=f"recognize_confirmed_page_{page_number_for_batch}",
        disabled=running or not page_ids,
    ):
        try:
            _start_confirmed_region_batch_job(
                document_result,
                backend,
                page_ids,
                scope_label=f"第 {page_number_for_batch} 页批量识别",
                page_number=page_number_for_batch,
            )
            st.rerun()
        except (OSError, RuntimeError, ValueError) as exc:
            st.error(f"本页批量识别启动失败：{exc}")
    if st.button(
        "识别全文全部已确认区域",
        key="recognize_all_confirmed_regions",
        type="primary",
        disabled=running or not all_ids,
    ):
        try:
            _start_confirmed_region_batch_job(
                document_result,
                backend,
                all_ids,
                scope_label="全文批量识别",
                page_number=page_number_for_batch,
            )
            st.rerun()
        except (OSError, RuntimeError, ValueError) as exc:
            st.error(f"全文批量识别启动失败：{exc}")


def _apply_edits_with_notice(
    document_result: dict,
    backend: str,
    edits: list[dict],
    *,
    rerun_ocsr: bool,
    message: str,
) -> None:
    if _region_job_is_running():
        st.error("区域识别任务运行中，请等待完成后再编辑其他区域。")
        return
    try:
        updated = _apply_document_edits(document_result, backend, edits, rerun_ocsr=rerun_ocsr)
        st.session_state["document_result"] = updated
        st.success(message)
        st.rerun()
    except RuntimeError as exc:
        st.error(str(exc))


def _region_editor(document_result: dict, backend: str) -> dict:
    active = [region for region in document_result.get("regions", []) if region.get("status") != "deleted"]
    if active:
        st.subheader("审核和编辑检测区域")
        region_ids = [region["region_id"] for region in active]
        selected_id = st.selectbox("区域", region_ids, key="document_region_select")
        selected = next(region for region in active if region["region_id"] == selected_id)
        # Legacy editor retained for old callers; the active workbench consumes canvas events above.
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
        if actions[0].button("仅保存框选", key=f"update_region_{selected_id}"):
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
        if actions[1].button("保存并识别", key=f"confirm_region_{selected_id}", type="primary"):
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
                    [{"action": "confirm_page", "page_number": page_number_for_batch, "note": "页面批量保存后识别。"}],
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


def _sync_region_bbox_state(region: dict) -> None:
    region_id = str(region.get("region_id") or "")
    bbox = [int(value) for value in (region.get("bbox") or [0, 0, 1, 1])]
    identity = tuple(bbox)
    if st.session_state.get(f"edit_bbox_identity_{region_id}") == identity:
        return
    st.session_state[f"edit_bbox_identity_{region_id}"] = identity
    for coord, value in zip(("x1", "y1", "x2", "y2"), bbox):
        st.session_state[f"edit_{coord}_{region_id}"] = value


def _consume_document_canvas_event(document_result: dict) -> dict:
    params = st.query_params
    if not params.get("document_region_editor_key"):
        return document_result
    try:
        nonce = str(params.get("doc_bbox_nonce") or "")
        if not nonce or st.session_state.get("document_canvas_event_nonce") == nonce:
            return document_result
        page_number = int(float(str(params.get("doc_bbox_page") or "0")))
        page = next(
            (item for item in document_result.get("pages", []) if int(item.get("page_number", 0)) == page_number),
            None,
        )
        if page is None:
            raise ValueError(f"找不到画布对应的第 {page_number} 页。")
        event = canvas_event_from_query(params, page)
        if event is None:
            return document_result
        updated, selected_id = apply_canvas_event(document_result, event)
        if updated is not document_result:
            result_path = persist_document_result_atomic(updated)
            record_result_payload(updated, result_path)
            st.session_state["document_result"] = updated
            action_label = {"create": "新框已创建", "update": "框选已自动保存", "delete": "区域已删除"}.get(event["action"], "区域已更新")
            st.success(f"{action_label}（原始页面坐标）。")
        if selected_id:
            st.session_state["document_region_select"] = str(selected_id)
        elif event.get("action") == "delete":
            remaining = [
                region for region in updated.get("regions", [])
                if region.get("status") != "deleted" and int(region.get("page_number", 0)) == page_number
            ]
            if remaining:
                st.session_state["document_region_select"] = str(remaining[0].get("region_id"))
        st.session_state["document_current_page"] = int(page_number)
        st.session_state["document_canvas_event_nonce"] = nonce
        return updated
    except (TypeError, ValueError, OSError) as exc:
        st.error(f"区域框选保存失败：{exc}")
        return document_result
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
        "key": selected_id or f"page-{int(page.get('page_number', 0))}-new",
        "selected_region_id": selected_id,
        "has_selection": bool(selected_id),
        "locked": _region_job_is_running(),
        "page_number": int(page.get("page_number", 0)),
        "src": f"data:{mime};base64,{encoded}",
        "width": width,
        "height": height,
        "bbox": [int(value) for value in bbox],
        "regions": overlay_regions,
    }
    html = f"""
    <div style="font: 14px system-ui, sans-serif; color: #163232;">
      <div style="display:flex; align-items:center; justify-content:space-between; gap:12px; margin-bottom:6px;">
        <span>空白处拖动画新框；点击框选择，拖动框移动，拖动角点缩放。松手后自动保存。</span>
        <button id="delete-region" type="button" style="white-space:nowrap; border:1px solid #b42318; color:#b42318; background:white; border-radius:6px; padding:5px 9px;">删除选中框</button>
      </div>
      <div style="position: relative; display: inline-block; max-width: 100%;">
        <img id="doc-region-image" src="{payload['src']}" style="width: min(100%, {display_width}px); display: block; cursor: crosshair; border: 1px solid #9ab8b8; border-radius: 6px;" />
        <canvas id="doc-region-overlay" tabindex="0" style="position:absolute; inset:0; outline:none;"></canvas>
      </div>
    </div>
    <script>
      const payload = {json.dumps(payload, ensure_ascii=False)};
      const image = document.getElementById("doc-region-image");
      const canvas = document.getElementById("doc-region-overlay");
      const deleteButton = document.getElementById("delete-region");
      const ctx = canvas.getContext("2d");
      let bbox = payload.bbox.slice();
      let drag = null;
      deleteButton.style.display = payload.has_selection && !payload.locked ? "inline-block" : "none";

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
        if (!payload.has_selection) return null;
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
      function hitRegion(p) {{
        const regions = (payload.regions || []).slice().reverse();
        return regions.find((region) => {{
          const box = region.bbox || [];
          return box.length === 4 && p.x >= box[0] && p.x <= box[2] && p.y >= box[1] && p.y <= box[3];
        }}) || null;
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
        if (payload.has_selection || (drag && drag.mode === "create")) {{
          ctx.fillStyle = "rgba(15, 118, 110, 0.13)";
          ctx.strokeStyle = "#0f766e";
          ctx.lineWidth = 3;
          ctx.fillRect(bbox[0] * s.sx, bbox[1] * s.sy, (bbox[2] - bbox[0]) * s.sx, (bbox[3] - bbox[1]) * s.sy);
          ctx.strokeRect(bbox[0] * s.sx, bbox[1] * s.sy, (bbox[2] - bbox[0]) * s.sx, (bbox[3] - bbox[1]) * s.sy);
        }}
        if (payload.has_selection) {{
          for (const [x, y] of [[bbox[0], bbox[1]], [bbox[2], bbox[1]], [bbox[0], bbox[3]], [bbox[2], bbox[3]]]) {{
            ctx.fillStyle = "#0f766e";
            ctx.fillRect(x * s.sx - 5, y * s.sy - 5, 10, 10);
            ctx.strokeStyle = "white";
            ctx.lineWidth = 2;
            ctx.strokeRect(x * s.sx - 5, y * s.sy - 5, 10, 10);
          }}
        }}
      }}
      function submitEvent(action, regionId, box) {{
        const params = new URLSearchParams(window.top.location.search);
        params.set("document_region_editor_key", payload.key);
        params.set("doc_bbox_action", action);
        params.set("doc_bbox_region_id", regionId || "");
        params.set("doc_bbox_page", String(payload.page_number));
        if (box) {{
          const s = scale();
          params.set("doc_bbox_x1", String(box[0] * s.sx));
          params.set("doc_bbox_y1", String(box[1] * s.sy));
          params.set("doc_bbox_x2", String(box[2] * s.sx));
          params.set("doc_bbox_y2", String(box[3] * s.sy));
          params.set("doc_canvas_width", String(canvas.width));
          params.set("doc_canvas_height", String(canvas.height));
        }}
        params.set("doc_bbox_nonce", String(Date.now()));
        window.top.location.href = window.top.location.pathname + "?" + params.toString();
      }}
      canvas.addEventListener("mousedown", (event) => {{
        if (payload.locked) return;
        const p = point(event);
        const mode = hitHandle(p);
        if (mode) {{
          drag = {{ mode, start: p, bbox: bbox.slice() }};
        }} else {{
          const region = hitRegion(p);
          if (region && region.region_id !== payload.selected_region_id) {{
            submitEvent("select", region.region_id, null);
            return;
          }}
          bbox = [p.x, p.y, p.x + 1, p.y + 1];
          drag = {{ mode: "create", start: p, bbox: bbox.slice() }};
          draw();
        }}
        canvas.focus();
        event.preventDefault();
      }});
      canvas.addEventListener("mousemove", (event) => {{
        if (!drag) return;
        const p = point(event);
        const dx = p.x - drag.start.x;
        const dy = p.y - drag.start.y;
        const next = drag.bbox.slice();
        if (drag.mode === "create") {{
          next[0] = Math.min(drag.start.x, p.x);
          next[1] = Math.min(drag.start.y, p.y);
          next[2] = Math.max(drag.start.x, p.x);
          next[3] = Math.max(drag.start.y, p.y);
        }} else if (drag.mode === "move") {{
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
        const action = drag.mode === "create" ? "create" : "update";
        drag = null;
        if ((bbox[2] - bbox[0]) < 3 || (bbox[3] - bbox[1]) < 3) {{ draw(); return; }}
        submitEvent(action, action === "update" ? payload.selected_region_id : null, bbox);
      }});
      deleteButton.addEventListener("click", () => {{
        if (window.confirm("确认删除当前区域框？")) submitEvent("delete", payload.selected_region_id, null);
      }});
      canvas.addEventListener("keydown", (event) => {{
        if ((event.key === "Delete" || event.key === "Backspace") && window.confirm("确认删除当前区域框？")) {{
          event.preventDefault();
          submitEvent("delete", payload.selected_region_id, null);
        }}
      }});
      image.addEventListener("load", draw);
      window.addEventListener("resize", draw);
      draw();
    </script>
    """
    components.html(html, height=display_height + 48)


def _render_region_crop_preview(page: dict, bbox: list[int]) -> None:
    image_path = Path(str(page.get("image_path") or ""))
    if not image_path.is_file():
        return
    image = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        st.warning("无法生成当前框选的裁剪预览。")
        return
    height, width = image.shape[:2]
    x1, y1, x2, y2 = [int(value) for value in bbox]
    x1, x2 = sorted((max(0, min(width - 1, x1)), max(1, min(width, x2))))
    y1, y2 = sorted((max(0, min(height - 1, y1)), max(1, min(height, y2))))
    if x2 <= x1 or y2 <= y1:
        st.warning("当前框选为空，无法生成裁剪预览。")
        return
    success, encoded = cv2.imencode(".png", image[y1:y2, x1:x2])
    if success:
        st.image(encoded.tobytes(), caption=f"裁剪预览 · 原始页坐标 [{x1}, {y1}, {x2}, {y2}]", width=420)


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

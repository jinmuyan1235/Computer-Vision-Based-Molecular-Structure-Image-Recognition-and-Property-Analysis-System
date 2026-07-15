"""Single-image recognition page."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import streamlit as st

from src.analysis.molecule_report import MoleculeReportGenerator
from src.runtime.run_store import ImageRun, create_image_run_from_bytes, save_run_report, write_runtime_metadata
from src.ui.image_viewer import show_upload_preview
from src.ui.report_view import show_correction_panel, show_report
from src.ui.state import (
    current_runtime_key,
    remember_backend_status,
    runtime_config_from_key,
)
from src.ui.styles import page_intro
from src.runtime.job_manager import extract_json_object, run_json_command

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def render_image_page(backend: str, show_preprocessing: bool, export_pdf: bool) -> None:
    page_intro("图片识别", "上传单张分子结构图，执行 OCSR 识别、RDKit 校验、性质计算和人工纠错。")
    uploaded = st.file_uploader("上传 PNG/JPG/JPEG 分子结构图", type=["png", "jpg", "jpeg"], key="single_upload")
    if uploaded is not None:
        show_upload_preview(uploaded, f"上传原图：{uploaded.name}")
        if st.button("开始识别与分析", type="primary", key="analyze_image"):
            progress = st.empty()
            progress.info("正在执行图像预处理、OCSR 与 RDKit 分析……")
            image_run = create_image_run_from_bytes(uploaded.getvalue(), uploaded.name)
            try:
                if backend == "demo":
                    report = MoleculeReportGenerator(backend, image_run.run_dir).generate(
                        image_path=image_run.input_path,
                        analysis_id=image_run.analysis_id,
                    )
                    save_run_report(report, image_run)
                else:
                    report = _process_image_subprocess(image_run, backend)
                st.session_state["image_report"] = report
                remember_backend_status(backend)
                progress.empty()
            except RuntimeError as exc:
                write_runtime_metadata(image_run, {"status": "failed", "message": str(exc)})
                progress.empty()
                st.error(str(exc))
    if "image_report" in st.session_state:
        active_report = show_correction_panel(st.session_state["image_report"])
        show_report(active_report, show_preprocessing, export_pdf, f"image_{active_report.get('analysis_id', 'report')[:8]}")


def _process_image_subprocess(image_run: ImageRun, backend: str) -> dict:
    """Run real OCSR outside Streamlit so native crashes do not kill the UI server."""
    runtime = runtime_config_from_key(current_runtime_key())
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "process_image.py"),
        "--input",
        str(image_run.input_path),
        "--backend",
        backend,
        "--original-filename",
        image_run.original_filename,
        "--analysis-id",
        image_run.analysis_id,
        "--run-dir",
        str(image_run.run_dir),
    ]
    if runtime.get("molscribe_device"):
        command.extend(["--molscribe-device", str(runtime["molscribe_device"])])
    if runtime.get("decimer_device"):
        command.extend(["--decimer-device", str(runtime["decimer_device"])])
    if runtime.get("visible_gpu_index") is not None:
        command.extend(["--visible-gpu-index", str(runtime["visible_gpu_index"])])

    env = os.environ.copy()
    env.setdefault("MOLSCRIBE_ISOLATED_SUBPROCESS", "true")
    env.setdefault("DECIMER_ISOLATED_SUBPROCESS", "true")
    completed = run_json_command(
        command,
        cwd=PROJECT_ROOT,
        env=env,
        timeout=900,
    )
    payload = completed.payload
    if completed.timed_out:
        raise RuntimeError("图像识别子进程超时，已终止后台进程。")
    if completed.returncode != 0:
        message = completed.last_output_line() or f"图像识别子进程退出码 {completed.returncode}"
        if payload and payload.get("message"):
            message = str(payload["message"])
        raise RuntimeError(f"图像识别子进程失败：{message}")
    if not payload or not payload.get("result_path"):
        raise RuntimeError("图像识别子进程未返回结果文件路径。")
    result_path = Path(str(payload["result_path"]))
    if not result_path.is_file():
        raise RuntimeError(f"图像识别结果文件不存在：{result_path}")
    return json.loads(result_path.read_text(encoding="utf-8"))


def _extract_json_object(text: str) -> dict | None:
    """Extract a JSON object from stdout that may also contain native-library logs."""
    return extract_json_object(text)

"""Single-image recognition page."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import streamlit as st

from src.analysis.correction import sha256_file
from src.analysis.molecule_report import MoleculeReportGenerator
from src.runtime.run_store import ImageRun, create_image_run_from_bytes, save_run_report, write_runtime_metadata
from src.storage.analysis_repository import record_report
from src.ui.image_editor import render_image_editor
from src.ui.image_viewer import show_upload_preview
from src.ui.report_view import show_correction_panel, show_report
from src.ui.state import (
    current_runtime_key,
    remember_backend_status,
    runtime_config_from_key,
)
from src.ui.styles import page_intro
from src.runtime.job_manager import extract_json_object, run_json_command
from src.utils.file_utils import safe_stem

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def render_image_page(backend: str, show_preprocessing: bool, export_pdf: bool) -> None:
    page_intro("图片识别", "上传单张分子结构图，执行 OCSR 识别、RDKit 校验、性质计算和人工纠错。")
    uploaded = st.file_uploader("上传 PNG/JPG/JPEG 分子结构图", type=["png", "jpg", "jpeg"], key="single_upload")
    if uploaded is not None:
        uploaded_bytes = uploaded.getvalue()
        show_upload_preview(uploaded, f"上传原图：{uploaded.name}")
        user_preprocessing, adjusted_bytes, has_adjustments = render_image_editor(
            uploaded_bytes,
            uploaded.name,
            key_prefix=f"single_{safe_stem(uploaded.name)}",
        )
        if st.button("开始识别与分析", type="primary", key="analyze_image"):
            progress = st.empty()
            progress.info("正在执行图像预处理、OCSR 与 RDKit 分析……")
            image_run = create_image_run_from_bytes(uploaded_bytes, uploaded.name)
            try:
                effective_input = _prepare_effective_input(image_run, uploaded.name, adjusted_bytes, user_preprocessing, has_adjustments)
                if backend == "demo":
                    report = MoleculeReportGenerator(backend, image_run.run_dir).generate(
                        image_path=effective_input,
                        analysis_id=image_run.analysis_id,
                    )
                    _attach_user_preprocessing(report, user_preprocessing, effective_input, image_run.input_path, has_adjustments)
                    result_path = save_run_report(report, image_run)
                    record_report(report, result_path)
                else:
                    report = _process_image_subprocess(image_run, backend, effective_input, user_preprocessing, has_adjustments)
                    record_report(report, (report.get("run") or {}).get("report_path"))
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


def _process_image_subprocess(
    image_run: ImageRun,
    backend: str,
    effective_input: str | Path | None = None,
    user_preprocessing: dict | None = None,
    user_adjusted: bool = False,
) -> dict:
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
    if effective_input is not None and Path(effective_input).resolve() != image_run.input_path.resolve():
        command.extend(["--effective-input", str(effective_input)])
    if user_preprocessing is not None:
        payload = dict(user_preprocessing)
        payload["applied"] = bool(user_adjusted)
        command.extend(["--user-preprocessing-json", json.dumps(payload, ensure_ascii=False)])
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


def _prepare_effective_input(
    image_run: ImageRun,
    original_filename: str,
    adjusted_bytes: bytes,
    user_preprocessing: dict,
    has_adjustments: bool,
) -> Path:
    if not has_adjustments:
        write_runtime_metadata(image_run, {"user_preprocessing": {**user_preprocessing, "applied": False}})
        return image_run.input_path
    stem = safe_stem(Path(original_filename).stem, "image")
    adjusted_path = image_run.input_dir / f"{stem}_user_adjusted.png"
    adjusted_path.write_bytes(adjusted_bytes)
    payload = {
        **user_preprocessing,
        "applied": True,
        "adjusted_image_path": str(adjusted_path.resolve()),
        "adjusted_image_sha256": sha256_file(adjusted_path),
    }
    user_preprocessing.clear()
    user_preprocessing.update(payload)
    write_runtime_metadata(image_run, {"user_preprocessing": payload})
    return adjusted_path


def _attach_user_preprocessing(
    report: dict,
    user_preprocessing: dict,
    effective_input: str | Path,
    original_input: str | Path,
    has_adjustments: bool,
) -> None:
    effective_path = Path(effective_input).expanduser().resolve()
    original_path = Path(original_input).expanduser().resolve()
    payload = {
        **user_preprocessing,
        "applied": bool(has_adjustments),
        "effective_image_path": str(effective_path),
        "effective_image_sha256": sha256_file(effective_path),
    }
    report["user_preprocessing"] = payload
    report.setdefault("input", {})
    report["input"]["effective_path"] = str(effective_path)
    report["input"]["effective_image_sha256"] = payload["effective_image_sha256"]
    if has_adjustments:
        report.setdefault("images", {}).setdefault("preprocessing", {})
        report["images"]["preprocessing"]["uploaded_original"] = str(original_path)
        report["images"]["preprocessing"]["user_adjusted"] = str(effective_path)

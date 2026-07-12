"""Single-image recognition page."""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from src.ui.image_viewer import show_upload_preview
from src.ui.report_view import show_correction_panel, show_report
from src.ui.state import get_report_generator, remember_backend_status
from src.ui.styles import page_intro


def render_image_page(backend: str, show_preprocessing: bool, export_pdf: bool) -> None:
    page_intro("图片识别", "上传单张分子结构图，执行 OCSR 识别、RDKit 校验、性质计算和人工纠错。")
    uploaded = st.file_uploader("上传 PNG/JPG/JPEG 分子结构图", type=["png", "jpg", "jpeg"], key="single_upload")
    if uploaded is not None:
        show_upload_preview(uploaded, f"上传原图：{uploaded.name}")
        if st.button("开始识别与分析", type="primary", key="analyze_image"):
            progress = st.empty()
            progress.info("正在执行图像预处理、OCSR 与 RDKit 分析……")
            suffix = Path(uploaded.name).suffix.lower()
            prefix = Path(uploaded.name).stem + "_"
            with tempfile.NamedTemporaryFile(prefix=prefix, suffix=suffix, delete=False) as temporary:
                temporary.write(uploaded.getvalue())
                temporary_path = Path(temporary.name)
            try:
                st.session_state["image_report"] = get_report_generator(backend).generate(image_path=temporary_path)
                st.session_state["image_report"]["input"]["filename"] = uploaded.name
                remember_backend_status(backend)
                progress.empty()
            finally:
                temporary_path.unlink(missing_ok=True)
    if "image_report" in st.session_state:
        active_report = show_correction_panel(st.session_state["image_report"])
        show_report(active_report, show_preprocessing, export_pdf, f"image_{active_report.get('analysis_id', 'report')[:8]}")

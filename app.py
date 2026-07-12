"""Streamlit entry point for the molecule recognition application."""

from __future__ import annotations

import streamlit as st

from src.ui.about_page import render_about_page
from src.ui.batch_page import render_batch_page
from src.ui.document_page import render_document_page
from src.ui.image_page import render_image_page
from src.ui.report_view import show_report
from src.ui.sidebar import render_sidebar
from src.ui.smiles_page import render_smiles_page
from src.ui.styles import apply_styles


st.set_page_config(page_title="分子结构识别与性质分析", page_icon="🧪", layout="wide")
apply_styles()

st.title("分子结构识别与性质分析")
st.caption("图片/PDF → OpenCV 区域处理 → OCSR → SMILES → RDKit 校验与报告")

selected_backend, show_preprocessing, export_pdf = render_sidebar()

image_tab, document_tab, smiles_tab, batch_tab, about_tab = st.tabs(
    ["图片识别", "PDF/多分子文档", "SMILES 分析", "批量处理", "项目说明"]
)

with image_tab:
    render_image_page(selected_backend, show_preprocessing, export_pdf)

with document_tab:
    render_document_page(selected_backend)

with smiles_tab:
    render_smiles_page(export_pdf)

with batch_tab:
    render_batch_page(selected_backend)

with about_tab:
    render_about_page()

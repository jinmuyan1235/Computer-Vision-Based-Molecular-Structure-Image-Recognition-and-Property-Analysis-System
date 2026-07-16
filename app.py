"""Streamlit entry point for the molecule recognition application."""

from __future__ import annotations

import streamlit as st

import config
from src.runtime.health import image_workflows_enabled, run_production_health_check
from src.runtime.run_store import cleanup_runs_if_due
from src.ui.about_page import render_about_page
from src.ui.batch_page import render_batch_page
from src.ui.document_page import render_document_page
from src.ui.health_page import render_blocked_workflow, render_health_banner, render_health_page
from src.ui.history_page import render_history_page
from src.ui.image_page import render_image_page
from src.ui.report_view import show_report
from src.ui.review_queue_page import render_review_queue_page
from src.ui.sidebar import render_sidebar
from src.ui.smiles_page import render_smiles_page
from src.ui.state import current_runtime_key, runtime_config_from_key
from src.ui.styles import apply_styles


config.initialize_directories()
st.set_page_config(page_title="分子结构识别与性质分析", page_icon="🧪", layout="wide")
apply_styles()

st.title("分子结构识别与性质分析")
st.caption(f"图片/PDF → OpenCV 区域处理 → OCSR → SMILES → RDKit 校验与报告；当前模式：{config.APP_MODE}")
st.session_state["run_cleanup"] = cleanup_runs_if_due()

selected_backend, show_preprocessing, export_pdf = render_sidebar()
health_force_refresh = bool(st.session_state.pop("health_force_refresh", False))
production_health = run_production_health_check(
    selected_backend,
    runtime_config=runtime_config_from_key(current_runtime_key()),
    production=config.IS_PRODUCTION_MODE,
    warmup=config.PRODUCTION_HEALTH_WARMUP if config.IS_PRODUCTION_MODE else False,
    load_model=config.PRODUCTION_HEALTH_LOAD_MODEL if config.IS_PRODUCTION_MODE else False,
    force=health_force_refresh,
    use_cache=config.PRODUCTION_HEALTH_CACHE_ENABLED,
)
st.session_state["production_health"] = production_health
if config.IS_PRODUCTION_MODE:
    render_health_banner(production_health)
real_ocsr_enabled = image_workflows_enabled(production_health)

image_tab, document_tab, smiles_tab, batch_tab, history_tab, review_tab, health_tab, about_tab = st.tabs(
    ["图片识别", "PDF/多分子文档", "SMILES 分析", "批量处理", "分析历史", "审核队列", "健康检查", "项目说明"]
)

with image_tab:
    if real_ocsr_enabled:
        render_image_page(selected_backend, show_preprocessing, export_pdf)
    else:
        render_blocked_workflow(production_health, "图片识别")

with document_tab:
    if real_ocsr_enabled:
        render_document_page(selected_backend)
    else:
        render_blocked_workflow(production_health, "文档识别")

with smiles_tab:
    render_smiles_page(export_pdf)

with batch_tab:
    if real_ocsr_enabled:
        render_batch_page(selected_backend)
    else:
        render_blocked_workflow(production_health, "批量识别")

with history_tab:
    render_history_page(selected_backend, show_preprocessing, export_pdf)

with review_tab:
    render_review_queue_page()

with health_tab:
    render_health_page(production_health)

with about_tab:
    render_about_page()

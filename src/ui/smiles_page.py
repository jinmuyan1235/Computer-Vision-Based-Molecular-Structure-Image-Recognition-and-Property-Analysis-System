"""Manual SMILES analysis page."""

from __future__ import annotations

import streamlit as st

from config import OUTPUT_DIR
from src.export.json_exporter import save_json
from src.storage.analysis_repository import record_report
from src.ui.report_view import show_report
from src.ui.state import get_report_generator
from src.ui.styles import page_intro


def render_smiles_page(export_pdf: bool) -> None:
    page_intro("SMILES 分析", "该功能不使用图片识别模型，只对你输入的 SMILES 进行 RDKit 校验、标准化和性质计算。")
    smiles_input = st.text_input("输入 SMILES", value="CCO", placeholder="例如：CCO")
    st.caption("长 SMILES 可以直接粘贴到输入框；分析结果会在下方以可复制代码框显示。")
    if st.button("分析 SMILES", type="primary", key="analyze_smiles"):
        with st.spinner("正在进行 RDKit 校验与性质计算……"):
            report = get_report_generator("manual").generate(smiles=smiles_input)
            report_path = save_json(report, OUTPUT_DIR / "smiles_history" / f"{report['analysis_id']}.json")
            record_report(report, report_path)
            st.session_state["smiles_report"] = report
    if "smiles_report" in st.session_state:
        show_report(st.session_state["smiles_report"], False, export_pdf, "smiles")

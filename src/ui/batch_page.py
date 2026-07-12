"""Batch image processing page."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from src.export.json_exporter import to_json_text
from src.ui.labels import BATCH_COLUMN_LABELS, localize_batch_rows
from src.ui.state import get_batch_analyzer, remember_backend_status
from src.ui.styles import page_intro


DEFAULT_COLUMNS = [
    "filename",
    "status",
    "backend",
    "final_smiles",
    "valid",
    "confidence",
    "inference_time_ms",
    "message",
]


def render_batch_page(backend: str) -> None:
    page_intro("批量处理", "批量处理服务器文件夹或一次上传的多张图片；单张失败不会中止整批任务。")
    folder_path = st.text_input("输入文件夹路径（可选）", value="")
    uploaded_files = st.file_uploader(
        "批量上传图片",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
        key="batch_upload",
    )
    if st.button("开始批量处理", type="primary", key="analyze_batch"):
        with st.spinner("正在逐张处理并生成汇总……"):
            try:
                if uploaded_files:
                    with tempfile.TemporaryDirectory() as temp_dir:
                        for item in uploaded_files:
                            (Path(temp_dir) / Path(item.name).name).write_bytes(item.getvalue())
                        st.session_state["batch_result"] = get_batch_analyzer(backend).analyze_folder(temp_dir)
                elif folder_path.strip():
                    st.session_state["batch_result"] = get_batch_analyzer(backend).analyze_folder(folder_path.strip())
                else:
                    st.warning("请上传至少一张图片或填写输入文件夹路径。")
                remember_backend_status(backend)
            except Exception as exc:
                st.error(f"批量处理失败：{exc}")

    if "batch_result" not in st.session_state:
        return
    batch_result = st.session_state["batch_result"]
    summary = batch_result["summary"]
    metrics = st.columns(4)
    metrics[0].metric("总图片", summary["total"])
    metrics[1].metric("识别成功", summary["successful"])
    metrics[2].metric("有效 SMILES", summary["valid_smiles"])
    metrics[3].metric("成功率", f"{summary['success_rate']:.1%}")

    rows = batch_result["rows"]
    default_rows = [{key: row.get(key) for key in DEFAULT_COLUMNS} for row in rows]
    st.dataframe(pd.DataFrame(localize_batch_rows(default_rows)), use_container_width=True, hide_index=True)
    with st.expander("查看完整字段", expanded=False):
        st.dataframe(pd.DataFrame(localize_batch_rows(rows)), use_container_width=True, hide_index=True)

    chart = batch_result["exports"]["summary_chart"]
    if Path(chart).is_file():
        st.image(chart, caption="批量结果统计", width=640)

    with st.expander("结果下载", expanded=True):
        csv_bytes = Path(batch_result["exports"]["csv"]).read_bytes()
        st.download_button("下载区域/批量结果表 CSV", csv_bytes, "batch_results.csv", "text/csv", key="batch_csv")
        st.download_button(
            "下载完整 JSON",
            to_json_text({"summary": summary, "results": batch_result["reports"]}),
            "batch_results.json",
            "application/json",
            key="batch_json",
        )


def default_batch_columns_chinese() -> list[str]:
    return [BATCH_COLUMN_LABELS[column] for column in DEFAULT_COLUMNS]

"""Batch image processing page."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from src.export.json_exporter import to_json_text
from src.ui.labels import BATCH_COLUMN_LABELS, localize_batch_rows
from src.ui.state import (
    current_runtime_key,
    get_batch_analyzer,
    remember_backend_status,
    runtime_config_from_key,
)
from src.ui.streamlit_compat import dataframe_stretch
from src.ui.styles import page_intro

PROJECT_ROOT = Path(__file__).resolve().parents[2]

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
                        st.session_state["batch_result"] = _run_batch(temp_dir, backend)
                elif folder_path.strip():
                    st.session_state["batch_result"] = _run_batch(folder_path.strip(), backend)
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
    dataframe_stretch(pd.DataFrame(localize_batch_rows(default_rows)), hide_index=True)
    with st.expander("查看完整字段", expanded=False):
        dataframe_stretch(pd.DataFrame(localize_batch_rows(rows)), hide_index=True)

    chart = batch_result["exports"]["summary_chart"]
    if Path(chart).is_file():
        st.image(chart, caption="批量结果统计", width=640)

    with st.expander("结果下载", expanded=True):
        csv_bytes = Path(batch_result["exports"]["csv"]).read_bytes()
        st.download_button("下载批量结果表 CSV", csv_bytes, "batch_results.csv", "text/csv", key="batch_csv")
        st.download_button(
            "下载完整 JSON",
            to_json_text({"summary": summary, "results": batch_result["reports"]}),
            "batch_results.json",
            "application/json",
            key="batch_json",
        )


def _run_batch(input_dir: str | Path, backend: str) -> dict:
    if backend == "demo":
        return get_batch_analyzer(backend).analyze_folder(input_dir)
    return _run_batch_subprocess(input_dir, backend)


def _run_batch_subprocess(input_dir: str | Path, backend: str) -> dict:
    runtime = runtime_config_from_key(current_runtime_key())
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "process_batch.py"),
        "--input",
        str(input_dir),
        "--backend",
        backend,
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
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    payload = _extract_json_object(completed.stdout)
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip().splitlines()
        message = detail[-1] if detail else f"批量处理子进程退出码 {completed.returncode}"
        if payload and payload.get("message"):
            message = str(payload["message"])
        raise RuntimeError(message)
    if not payload or not payload.get("result_path"):
        raise RuntimeError("批量处理子进程未返回结果文件路径。")
    result_path = Path(str(payload["result_path"]))
    if not result_path.is_file():
        raise RuntimeError(f"批量处理结果文件不存在：{result_path}")
    return json.loads(result_path.read_text(encoding="utf-8"))


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


def default_batch_columns_chinese() -> list[str]:
    return [BATCH_COLUMN_LABELS[column] for column in DEFAULT_COLUMNS]

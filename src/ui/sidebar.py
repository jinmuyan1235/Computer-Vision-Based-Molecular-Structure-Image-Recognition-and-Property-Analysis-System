"""Sidebar backend selection and settings."""

from __future__ import annotations

from typing import Any

import streamlit as st

import config
from src.runtime.gpu_manager import default_gpu_selection, gpu_selection_options
from src.ui.labels import (
    BACKEND_DESCRIPTIONS,
    backend_label,
    default_backend,
    runnable_backends,
    unavailable_backends,
)
from src.ui.state import current_runtime_key, get_backend_statuses, merged_backend_status, runtime_config_from_key


def _query_gpu_selection() -> str | None:
    try:
        value = st.query_params.get("gpu_device")
    except Exception:
        return None
    if isinstance(value, list):
        return str(value[0]) if value else None
    return str(value) if value else None


def render_sidebar() -> tuple[str, bool, bool]:
    """Render sidebar controls and return selected backend and display switches."""
    with st.sidebar:
        st.header("运行设置")
        gpu_options = gpu_selection_options()
        gpu_values = [option["value"] for option in gpu_options]
        gpu_labels = {option["value"]: option["label"] for option in gpu_options}
        current_gpu = st.session_state.get("gpu_device_selection") or _query_gpu_selection() or default_gpu_selection(gpu_options)
        if current_gpu not in gpu_values:
            current_gpu = default_gpu_selection(gpu_options)
        selected_gpu = st.selectbox(
            "本机推理设备",
            gpu_values,
            index=gpu_values.index(current_gpu),
            format_func=lambda value: gpu_labels.get(value, value),
            key="gpu_device_selection",
        )
        try:
            st.query_params["gpu_device"] = selected_gpu
        except Exception:
            pass
        runtime = runtime_config_from_key(current_runtime_key())
        st.caption(
            "实际传入："
            f"MolScribe={runtime.get('molscribe_device')}，"
            f"DECIMER={runtime.get('decimer_device')}，"
            f"GPU索引={runtime.get('visible_gpu_index') or '自动/未指定'}"
        )
        if st.session_state.get("gpu_device_selection") not in {"auto", "cpu"}:
            st.caption("DECIMER/TensorFlow 已加载后再切换 GPU，建议重启 Streamlit 以确保显存绑定生效。")

        statuses = get_backend_statuses()
        if "selected_backend" not in st.session_state:
            st.session_state["selected_backend"] = default_backend(statuses, config.OCSR_BACKEND)

        show_demo = st.session_state.get("show_demo_backend", False)
        options = runnable_backends(statuses, include_demo=show_demo)
        if st.session_state["selected_backend"] not in options:
            st.session_state["selected_backend"] = default_backend(statuses, config.OCSR_BACKEND)

        selected = st.selectbox(
            "当前识别后端",
            options,
            index=options.index(st.session_state["selected_backend"]),
            format_func=backend_label,
            key="selected_backend",
        )
        status = merged_backend_status(selected)
        if status.get("available"):
            st.success("后端可用")
        else:
            st.error("后端不可用")
        if selected == "demo":
            st.warning("演示模式只识别内置样例文件名，不是真实 AI 图像识别。")

        show_preprocessing = st.checkbox("显示 OpenCV 预处理过程", value=True)
        export_pdf = st.checkbox("启用 PDF 报告", value=False)

        with st.expander("高级设置", expanded=False):
            st.checkbox("显示演示模式", value=show_demo, key="show_demo_backend")
            st.caption("SMILES 分析页不调用图片识别模型；该设置只影响图片、文档和批处理。")

        with st.expander("识别后端说明", expanded=False):
            for backend, description in BACKEND_DESCRIPTIONS.items():
                st.markdown(f"**{backend_label(backend)}**  \n{description}")

        unavailable = unavailable_backends(statuses)
        if unavailable:
            with st.expander("未配置的识别后端", expanded=False):
                for backend in unavailable:
                    item = statuses.get(backend, {})
                    st.caption(f"{backend_label(backend)}：{item.get('message') or '未配置'}")

        with st.expander("技术信息", expanded=False):
            st.write(f"**当前设备选择：** {gpu_labels.get(st.session_state.get('gpu_device_selection', 'auto'), '自动')}")
            _render_technical_status(status)

    return selected, show_preprocessing, export_pdf


def _render_technical_status(status: dict[str, Any]) -> None:
    rows = {
        "内部后端": status.get("backend"),
        "模型": status.get("model_name") or status.get("model_path") or "无",
        "设备": status.get("device") or status.get("requested_device") or "未指定",
        "包版本": status.get("package_version") or "未安装/未提供",
        "输入策略": status.get("image_strategy") or "默认",
        "最近推理耗时": (
            f"{status.get('last_inference_time_ms')} ms"
            if status.get("last_inference_time_ms") is not None
            else "暂无"
        ),
    }
    for key, value in rows.items():
        st.write(f"**{key}：** {value}")
    children = status.get("child_statuses") or []
    for child in children:
        st.caption(f"{child.get('backend')}：{child.get('message') or ''}")

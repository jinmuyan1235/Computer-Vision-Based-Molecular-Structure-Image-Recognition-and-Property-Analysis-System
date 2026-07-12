"""Sidebar backend selection and settings."""

from __future__ import annotations

from typing import Any

import streamlit as st

import config
from src.runtime.gpu_manager import environment_status
from src.ui.labels import (
    BACKEND_DESCRIPTIONS,
    backend_label,
    default_backend,
    runnable_backends,
    unavailable_backends,
)
from src.ui.state import get_backend_statuses, merged_backend_status


def render_sidebar() -> tuple[str, bool, bool]:
    """Render sidebar controls and return selected backend and display switches."""
    with st.sidebar:
        st.header("运行设置")
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
            _render_technical_status(status)

    return selected, show_preprocessing, export_pdf


def _render_technical_status(status: dict[str, Any]) -> None:
    runtime = environment_status(run_matrix_test=False)
    nvidia = runtime.get("nvidia_smi", {})
    first_gpu = (nvidia.get("gpus") or [{}])[0]
    rows = {
        "GPU": first_gpu.get("name") or "未检测到",
        "PyTorch CUDA": "可用" if (runtime.get("torch") or {}).get("cuda_available") else "不可用",
        "TensorFlow GPU": "可用" if (runtime.get("tensorflow") or {}).get("gpu_available") else "不可用",
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

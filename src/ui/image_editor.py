"""Lightweight single-image preprocessing editor widgets."""

from __future__ import annotations

from typing import Any

import streamlit as st

from src.preprocess.user_adjustments import (
    apply_user_adjustments,
    encode_png,
    has_user_adjustments,
    image_dimensions,
    normalize_user_adjustments,
)


OUTPUT_STAGE_LABELS = {
    "original": "原图",
    "grayscale": "灰度",
    "normalized": "归一化",
    "binary": "二值化",
}


def render_image_editor(image_bytes: bytes, filename: str, key_prefix: str = "single_image") -> tuple[dict[str, Any], bytes, bool]:
    """Render lightweight preprocessing controls and return adjustments plus preview bytes."""
    dimensions = image_dimensions(image_bytes)
    default_bbox = [0, 0, dimensions["width"], dimensions["height"]]
    with st.expander("单图预处理编辑器", expanded=False):
        st.caption(f"{filename} | {dimensions['width']} × {dimensions['height']}")
        controls = st.columns(4)
        use_crop = controls[0].checkbox("裁剪", value=False, key=f"{key_prefix}_crop_enabled")
        trim_whitespace = controls[1].checkbox("去除白边", value=False, key=f"{key_prefix}_trim")
        invert = controls[2].checkbox("黑白反转", value=False, key=f"{key_prefix}_invert")
        output_stage = controls[3].selectbox(
            "输出版本",
            list(OUTPUT_STAGE_LABELS),
            index=0,
            format_func=lambda value: OUTPUT_STAGE_LABELS[value],
            key=f"{key_prefix}_stage",
        )

        crop_bbox: list[int] = []
        if use_crop:
            crop_cols = st.columns(4)
            x1 = crop_cols[0].number_input("x1", min_value=0, max_value=dimensions["width"], value=default_bbox[0], key=f"{key_prefix}_x1")
            y1 = crop_cols[1].number_input("y1", min_value=0, max_value=dimensions["height"], value=default_bbox[1], key=f"{key_prefix}_y1")
            x2 = crop_cols[2].number_input("x2", min_value=0, max_value=dimensions["width"], value=default_bbox[2], key=f"{key_prefix}_x2")
            y2 = crop_cols[3].number_input("y2", min_value=0, max_value=dimensions["height"], value=default_bbox[3], key=f"{key_prefix}_y2")
            crop_bbox = [int(x1), int(y1), int(x2), int(y2)]

        rotate_cols = st.columns(3)
        right_angle = rotate_cols[0].selectbox("旋转 90°", [0, 90, 180, 270], index=0, key=f"{key_prefix}_right_angle")
        fine_rotation = rotate_cols[1].slider("小角度旋转", -15.0, 15.0, 0.0, 0.5, key=f"{key_prefix}_fine_rotation")
        contrast = rotate_cols[2].slider("对比度", 0.5, 2.0, 1.0, 0.05, key=f"{key_prefix}_contrast")

        adjustments = normalize_user_adjustments(
            {
                "crop_bbox": crop_bbox,
                "rotation": float(right_angle) + float(fine_rotation),
                "invert": invert,
                "contrast": float(contrast),
                "trim_whitespace": trim_whitespace,
                "output_stage": output_stage,
            }
        )
        try:
            adjusted_image = apply_user_adjustments(image_bytes, adjustments)
            adjusted_bytes = encode_png(adjusted_image)
            st.image(adjusted_bytes, caption="调整后预览", width=600)
        except Exception as exc:
            st.warning(f"预处理预览失败：{exc}")
            adjusted_bytes = image_bytes
        st.json({"user_preprocessing": adjustments})
    return adjustments, adjusted_bytes, has_user_adjustments(adjustments)

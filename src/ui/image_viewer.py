"""Consistent image preview helpers for Streamlit pages."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

UPLOAD_PREVIEW_WIDTH = 600
STRUCTURE_PREVIEW_WIDTH = 480
PREPROCESS_PREVIEW_WIDTH = 260
DOCUMENT_PREVIEW_WIDTH = 900


def show_upload_preview(image: Any, caption: str | None = None) -> None:
    st.image(image, caption=caption, width=UPLOAD_PREVIEW_WIDTH)


def show_structure(image_path: str | Path | None, caption: str = "分子结构图") -> None:
    if image_path and Path(image_path).is_file():
        st.image(str(image_path), caption=caption, width=STRUCTURE_PREVIEW_WIDTH)


def show_preprocess_thumbnail(image_path: str | Path, caption: str) -> None:
    st.image(str(image_path), caption=caption, width=PREPROCESS_PREVIEW_WIDTH)


def show_document_page(image_path: str | Path, caption: str) -> None:
    st.image(str(image_path), caption=caption, width=DOCUMENT_PREVIEW_WIDTH)
    with st.expander("查看大图"):
        st.image(str(image_path), caption=caption, use_container_width=True)

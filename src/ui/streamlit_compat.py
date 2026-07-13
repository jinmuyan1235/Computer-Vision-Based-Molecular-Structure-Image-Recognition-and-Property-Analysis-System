"""Small Streamlit compatibility helpers."""

from __future__ import annotations

from typing import Any

import streamlit as st


def dataframe_stretch(data: Any, **kwargs: Any) -> Any:
    """Render a dataframe full-width on new Streamlit with a legacy fallback."""
    try:
        return st.dataframe(data, width="stretch", **kwargs)
    except TypeError:
        return st.dataframe(data, use_container_width=True, **kwargs)


def image_stretch(image: Any, **kwargs: Any) -> Any:
    """Render an image full-width on new Streamlit with a legacy fallback."""
    try:
        return st.image(image, width="stretch", **kwargs)
    except TypeError:
        return st.image(image, use_container_width=True, **kwargs)

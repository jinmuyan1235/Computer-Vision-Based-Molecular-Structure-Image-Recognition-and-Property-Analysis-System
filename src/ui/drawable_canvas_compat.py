"""Compatibility wrapper for streamlit-drawable-canvas on recent Streamlit.

The component's published Python wrapper uses a private Streamlit image API whose
signature changed in Streamlit 1.59.  Passing the background as an inline PNG keeps
the component self-contained and avoids depending on that private API.
"""

from __future__ import annotations

import base64
import io
from copy import deepcopy

import numpy as np
from PIL import Image
from streamlit_drawable_canvas import CanvasResult, _component_func, _data_url_to_image


def image_data_url(image: Image.Image) -> str:
    """Return a browser-ready PNG data URL without Streamlit internals."""

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def st_canvas_compat(
    *,
    fill_color: str = "#eee",
    stroke_width: int = 20,
    stroke_color: str = "black",
    background_color: str = "",
    background_image: Image.Image | None = None,
    update_streamlit: bool = True,
    height: int = 400,
    width: int = 600,
    drawing_mode: str = "freedraw",
    initial_drawing: dict | None = None,
    display_toolbar: bool = True,
    point_display_radius: int = 3,
    key: str | None = None,
) -> CanvasResult:
    """Render the drawable canvas with a Streamlit-1.59-safe background."""

    background_image_url = image_data_url(background_image) if background_image else None
    if background_image_url:
        background_color = ""
    drawing = deepcopy(initial_drawing) if initial_drawing is not None else {"version": "4.4.0"}
    drawing["background"] = background_color
    component_value = _component_func(
        fillColor=fill_color,
        strokeWidth=stroke_width,
        strokeColor=stroke_color,
        backgroundColor=background_color,
        backgroundImageURL=background_image_url,
        realtimeUpdateStreamlit=update_streamlit and drawing_mode != "polygon",
        canvasHeight=height,
        canvasWidth=width,
        drawingMode=drawing_mode,
        initialDrawing=drawing,
        displayToolbar=display_toolbar,
        displayRadius=point_display_radius,
        key=key,
        default=None,
    )
    if component_value is None:
        return CanvasResult()
    return CanvasResult(
        np.asarray(_data_url_to_image(component_value["data"])),
        component_value["raw"],
    )

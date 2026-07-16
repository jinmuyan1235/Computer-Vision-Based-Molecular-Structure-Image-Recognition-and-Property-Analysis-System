"""Tests for lightweight user-controlled single-image preprocessing."""

from pathlib import Path

import cv2
import numpy as np

from src.preprocess.image_preprocessor import ImagePreprocessor
from src.preprocess.user_adjustments import (
    apply_user_adjustments,
    has_user_adjustments,
    normalize_user_adjustments,
    save_user_adjusted_image,
)
from src.runtime.run_store import create_image_run_from_bytes
from src.ui.image_editor import crop_bbox_from_points
from src.ui.image_page import _attach_user_preprocessing, _prepare_effective_input


def _synthetic_structure() -> np.ndarray:
    image = np.full((80, 120, 3), 255, dtype=np.uint8)
    cv2.line(image, (20, 40), (100, 40), (0, 0, 0), 3)
    cv2.putText(image, "OH", (48, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
    return image


def _encode_png(image: np.ndarray) -> bytes:
    success, encoded = cv2.imencode(".png", image)
    assert success
    return encoded.tobytes()


def test_normalize_user_adjustments_clamps_crop_and_values() -> None:
    image = _synthetic_structure()

    normalized = normalize_user_adjustments(
        {
            "crop_bbox": [-10, 5, 90, 999],
            "rotation": "not-a-number",
            "contrast": 9,
            "output_stage": "unsupported",
        },
        image.shape,
    )

    assert normalized["crop_bbox"] == [0, 5, 90, 80]
    assert normalized["rotation"] == 0
    assert normalized["contrast"] == 4
    assert normalized["output_stage"] == "original"
    assert has_user_adjustments(normalized) is True


def test_visual_crop_points_create_clamped_bbox() -> None:
    dimensions = {"width": 120, "height": 80}

    assert crop_bbox_from_points([(90, 70), (10, 20)], dimensions) == [10, 20, 90, 70]
    assert crop_bbox_from_points([(-10, 5), (150, 999)], dimensions) == [0, 5, 120, 80]
    assert crop_bbox_from_points([(20, 20), (20, 60)], dimensions) == []


def test_apply_user_adjustments_outputs_requested_versions(tmp_path: Path) -> None:
    image = _synthetic_structure()

    cropped = apply_user_adjustments(image, {"crop_bbox": [20, 20, 110, 60], "contrast": 1.3})
    binary = apply_user_adjustments(image, {"output_stage": "binary", "invert": True})
    normalized = apply_user_adjustments(image, {"output_stage": "normalized"}, default_size=(64, 64))
    trimmed = apply_user_adjustments(image, {"trim_whitespace": True})

    assert cropped.shape[0] == 40
    assert cropped.shape[1] == 90
    assert binary.ndim == 2
    assert binary.dtype == np.uint8
    assert set(np.unique(binary)).issubset({0, 255})
    assert normalized.shape == (64, 64)
    assert trimmed.shape[0] < image.shape[0]
    assert trimmed.shape[1] < image.shape[1]

    saved = Path(save_user_adjusted_image(image, {"rotation": 90}, tmp_path / "adjusted.png"))
    assert saved.is_file()
    assert cv2.imread(str(saved)) is not None


def test_preprocess_pipeline_records_uploaded_and_adjusted_stages(tmp_path: Path) -> None:
    image_path = tmp_path / "molecule.png"
    image_path.write_bytes(_encode_png(_synthetic_structure()))

    stages = ImagePreprocessor(default_size=(64, 64)).preprocess_pipeline(
        image_path,
        user_adjustments={"crop_bbox": [10, 10, 110, 70], "output_stage": "grayscale"},
    )

    assert {"uploaded_original", "user_adjusted", "original", "normalized"} <= stages.keys()
    assert stages["uploaded_original"].shape == (80, 120, 3)
    assert stages["user_adjusted"].ndim == 2
    assert stages["normalized"].shape == (64, 64)


def test_image_page_persists_adjusted_input_and_report_metadata(tmp_path: Path) -> None:
    payload = _encode_png(_synthetic_structure())
    image_run = create_image_run_from_bytes(payload, "aspirin.png", runs_root=tmp_path / "runs", analysis_id="analysis-1")
    adjustments = {
        "crop_bbox": [5, 5, 100, 70],
        "rotation": 2.5,
        "invert": False,
        "contrast": 1.1,
        "trim_whitespace": True,
        "output_stage": "normalized",
    }

    adjusted_input = _prepare_effective_input(image_run, "aspirin.png", payload, adjustments, True)
    report = {"input": {"type": "image"}, "images": {"preprocessing": {}}}
    _attach_user_preprocessing(report, adjustments, adjusted_input, image_run.input_path, True)

    assert adjusted_input.is_file()
    assert adjusted_input.name == "aspirin_user_adjusted.png"
    assert report["user_preprocessing"]["applied"] is True
    assert report["user_preprocessing"]["effective_image_sha256"]
    assert report["input"]["effective_path"] == str(adjusted_input.resolve())
    assert report["images"]["preprocessing"]["uploaded_original"] == str(image_run.input_path.resolve())
    assert report["images"]["preprocessing"]["user_adjusted"] == str(adjusted_input.resolve())

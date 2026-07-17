"""Shared OpenCV candidate screening for interactive documents and dataset collection."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np


ScreeningProfile = Literal["baseline", "candidate"]


@dataclass(frozen=True)
class CandidateFeatureThresholds:
    """Profile-independent feature extraction and decision thresholds."""

    text_row_ink_ratio: float = 0.012
    text_min_run_pixels: int = 2
    text_min_run_height_ratio: float = 0.006
    component_min_area: int = 6
    small_component_max_area: int = 80
    long_line_min_pixels: int = 32
    long_line_min_dimension_ratio: float = 0.10
    horizontal_tolerance_pixels: int = 4
    horizontal_tolerance_height_ratio: float = 0.08
    hough_min_line_pixels: int = 24
    hough_max_line_gap: int = 8
    cluster_min_width: int = 32
    cluster_min_height: int = 24
    skeletal_min_long_lines: int = 8
    skeletal_max_components: int = 55
    skeletal_max_ink_ratio: float = 0.12
    skeletal_max_small_component_ratio: float = 0.35
    skeletal_max_projection_ratio: float = 0.55
    table_min_aspect: float = 0.5
    table_max_aspect: float = 6.0
    mixed_arrow_min_clusters: int = 2
    mixed_arrow_min_components: int = 6
    mixed_arrow_min_text_lines: int = 2
    arrow_max_text_lines: int = 3
    reaction_condition_max_aspect: float = 8.0
    reaction_condition_min_ink_ratio: float = 0.008
    reaction_condition_max_ink_ratio: float = 0.20
    reaction_condition_min_text_lines: int = 1
    reaction_condition_max_text_lines: int = 3
    reaction_condition_min_components: int = 3
    reaction_condition_max_components: int = 28
    reaction_condition_min_small_ratio: float = 0.25
    dense_text_min_components: int = 35
    dense_text_max_ink_ratio: float = 0.26
    short_text_max_height: int = 90
    short_text_min_aspect: float = 1.7
    short_text_min_components: int = 2
    short_text_max_ink_ratio: float = 0.24
    body_text_min_components: int = 12
    body_text_min_aspect: float = 1.3
    body_text_max_height: int = 240
    body_text_max_ink_ratio: float = 0.22
    score_ink_min: float = 0.012
    score_ink_max: float = 0.24
    score_edge_min: float = 0.02
    score_component_min: int = 3
    score_aspect_min: float = 0.25
    score_aspect_max: float = 4.5


DEFAULT_FEATURE_THRESHOLDS = CandidateFeatureThresholds()


@dataclass(frozen=True)
class CandidateScreeningConfig:
    """Named thresholds for candidate formation and post-detection classification."""

    name: ScreeningProfile
    dilation_kernel: tuple[int, int]
    bbox_padding: int
    merge_overlap_ratio: float
    min_width: int
    min_height: int
    blank_ink_ratio: float
    dense_ink_ratio: float
    molecule_score_threshold: float
    reclassify_non_molecule: bool
    table_projection_ratio: float
    arrow_min_aspect: float
    arrow_min_width: int
    arrow_max_height: int
    arrow_max_ink_ratio: float
    arrow_hough_threshold: int
    arrow_min_length_ratio: float
    reaction_condition_min_aspect: float
    reaction_condition_max_height: int
    text_min_lines: int
    text_min_components: int
    text_small_component_ratio: float
    multi_cluster_kernel: tuple[int, int]
    multi_cluster_min_area_ratio: float
    multi_cluster_min_count: int
    dense_figure_edge_ratio: float
    promotion_min_ink_ratio: float
    promotion_max_ink_ratio: float
    promotion_min_components: int
    promotion_max_components: int
    promotion_min_small_component_ratio: float
    promotion_max_small_component_ratio: float
    promotion_max_text_lines: int
    promotion_max_aspect: float
    sparse_arrow_max_components: int
    sparse_arrow_max_ink_ratio: float
    sparse_arrow_min_horizontal_ratio: float
    merged_min_components: int
    merged_min_long_lines: int
    high_ink_uncertain_ratio: float
    existing_molecule_min_ink_ratio: float
    existing_molecule_min_components: int
    existing_molecule_min_aspect: float
    existing_molecule_max_aspect: float
    features: CandidateFeatureThresholds = DEFAULT_FEATURE_THRESHOLDS


BASELINE_SCREENING_CONFIG = CandidateScreeningConfig(
    name="baseline",
    dilation_kernel=(23, 17), bbox_padding=8, merge_overlap_ratio=0.72,
    min_width=70, min_height=55, blank_ink_ratio=0.006, dense_ink_ratio=0.38,
    molecule_score_threshold=0.68, reclassify_non_molecule=False,
    table_projection_ratio=0.65,
    arrow_min_aspect=2.5, arrow_min_width=180, arrow_max_height=190,
    arrow_max_ink_ratio=0.18, arrow_hough_threshold=40, arrow_min_length_ratio=0.25,
    reaction_condition_min_aspect=1.7, reaction_condition_max_height=145,
    text_min_lines=4, text_min_components=18, text_small_component_ratio=0.72,
    multi_cluster_kernel=(15, 11), multi_cluster_min_area_ratio=0.08, multi_cluster_min_count=3,
    dense_figure_edge_ratio=0.025,
    promotion_min_ink_ratio=0.03, promotion_max_ink_ratio=0.06,
    promotion_min_components=12, promotion_max_components=35,
    promotion_min_small_component_ratio=0.25, promotion_max_small_component_ratio=0.65,
    promotion_max_text_lines=3, promotion_max_aspect=3.0,
    sparse_arrow_max_components=2, sparse_arrow_max_ink_ratio=0.05,
    sparse_arrow_min_horizontal_ratio=0.40,
    merged_min_components=16, merged_min_long_lines=6, high_ink_uncertain_ratio=0.16,
    existing_molecule_min_ink_ratio=0.012, existing_molecule_min_components=3,
    existing_molecule_min_aspect=0.20, existing_molecule_max_aspect=4.5,
)

CANDIDATE_SCREENING_CONFIG = CandidateScreeningConfig(
    name="candidate",
    # Smaller dilation/padding and stricter overlap avoid joining adjacent Scheme structures.
    dilation_kernel=(17, 11), bbox_padding=6, merge_overlap_ratio=0.82,
    min_width=64, min_height=50, blank_ink_ratio=0.006, dense_ink_ratio=0.34,
    molecule_score_threshold=0.72, reclassify_non_molecule=True,
    table_projection_ratio=0.58,
    # The development confusion has molecule->reaction errors, so mixed arrow boxes are routed early.
    arrow_min_aspect=1.8, arrow_min_width=120, arrow_max_height=230,
    arrow_max_ink_ratio=0.24, arrow_hough_threshold=32, arrow_min_length_ratio=0.22,
    reaction_condition_min_aspect=1.45, reaction_condition_max_height=165,
    text_min_lines=3, text_min_components=14, text_small_component_ratio=0.62,
    multi_cluster_kernel=(11, 7), multi_cluster_min_area_ratio=0.055, multi_cluster_min_count=2,
    dense_figure_edge_ratio=0.022,
    promotion_min_ink_ratio=0.03, promotion_max_ink_ratio=0.065,
    promotion_min_components=10, promotion_max_components=40,
    promotion_min_small_component_ratio=0.25, promotion_max_small_component_ratio=0.70,
    promotion_max_text_lines=4, promotion_max_aspect=3.2,
    sparse_arrow_max_components=3, sparse_arrow_max_ink_ratio=0.06,
    sparse_arrow_min_horizontal_ratio=0.34,
    merged_min_components=12, merged_min_long_lines=4, high_ink_uncertain_ratio=0.10,
    existing_molecule_min_ink_ratio=0.012, existing_molecule_min_components=3,
    existing_molecule_min_aspect=0.20, existing_molecule_max_aspect=4.5,
)

SCREENING_CONFIGS = {
    "baseline": BASELINE_SCREENING_CONFIG,
    "candidate": CANDIDATE_SCREENING_CONFIG,
}


def get_screening_config(config: ScreeningProfile | CandidateScreeningConfig) -> CandidateScreeningConfig:
    if isinstance(config, CandidateScreeningConfig):
        return config
    try:
        return SCREENING_CONFIGS[config]
    except KeyError as exc:
        raise ValueError(f"Unknown candidate-screening config: {config}") from exc


@dataclass(frozen=True)
class CandidateScreeningResult:
    recommended_region_type: str
    molecule_candidate: bool
    screening_score: float
    reason_codes: tuple[str, ...]
    diagnostics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "recommended_region_type": self.recommended_region_type,
            "molecule_candidate": self.molecule_candidate,
            "screening_score": self.screening_score,
            "reason_codes": list(self.reason_codes),
            "diagnostics": self.diagnostics,
        }


def _read_image(page_image: str | Path | np.ndarray) -> np.ndarray:
    if isinstance(page_image, np.ndarray):
        return page_image
    path = Path(page_image).expanduser().resolve()
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unable to decode candidate page image: {path}")
    return image


def _normalize_initial(region_type: str) -> str:
    value = str(region_type or "uncertain").lower()
    if value in {"reaction", "reaction_like", "reaction_arrow", "reaction_condition"}:
        return "reaction"
    if value in {"unknown", "ignore", "non_molecule", "uncertain_visual"}:
        return "uncertain"
    return value


def _foreground_binary(crop: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)


def _text_line_count(binary: np.ndarray) -> int:
    height, width = binary.shape[:2]
    thresholds = DEFAULT_FEATURE_THRESHOLDS
    active = np.sum(binary > 0, axis=1) / max(width, 1) > thresholds.text_row_ink_ratio
    minimum_run = max(thresholds.text_min_run_pixels, int(height * thresholds.text_min_run_height_ratio))
    lines = run = 0
    for value in active:
        if value:
            run += 1
        else:
            lines += int(run >= minimum_run)
            run = 0
    return lines + int(run >= minimum_run)


def _line_diagnostics(binary: np.ndarray, config: CandidateScreeningConfig) -> dict[str, Any]:
    height, width = binary.shape[:2]
    edges = cv2.Canny(binary, 60, 180)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=config.arrow_hough_threshold,
        minLineLength=max(config.features.hough_min_line_pixels, int(width * config.arrow_min_length_ratio)),
        maxLineGap=config.features.hough_max_line_gap,
    )
    total = long_segments = long_horizontal = 0
    longest_horizontal_ratio = 0.0
    if lines is not None:
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            length = float(((int(x2) - int(x1)) ** 2 + (int(y2) - int(y1)) ** 2) ** 0.5)
            total += 1
            thresholds = config.features
            long_segments += int(length >= max(
                thresholds.long_line_min_pixels,
                min(width, height) * thresholds.long_line_min_dimension_ratio,
            ))
            if abs(int(y2) - int(y1)) <= max(
                thresholds.horizontal_tolerance_pixels,
                height * thresholds.horizontal_tolerance_height_ratio,
            ):
                ratio = length / max(width, 1)
                if ratio >= config.arrow_min_length_ratio:
                    long_horizontal += 1
                    longest_horizontal_ratio = max(longest_horizontal_ratio, ratio)
    return {
        "edge_ratio": float(np.count_nonzero(edges) / max(width * height, 1)),
        "line_count": total, "long_line_count": long_segments,
        "long_horizontal_line_count": long_horizontal,
        "longest_horizontal_ratio": longest_horizontal_ratio,
    }


def _structure_cluster_count(binary: np.ndarray, config: CandidateScreeningConfig) -> int:
    height, width = binary.shape[:2]
    merged = cv2.dilate(
        binary,
        cv2.getStructuringElement(cv2.MORPH_RECT, config.multi_cluster_kernel),
        iterations=1,
    )
    contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    crop_area = max(width * height, 1)
    clusters = 0
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        del x, y
        if w < config.features.cluster_min_width or h < config.features.cluster_min_height:
            continue
        if (w * h) / crop_area < config.multi_cluster_min_area_ratio:
            continue
        clusters += 1
    return clusters


def screen_region_candidate(
    page_image: str | Path | np.ndarray,
    bbox: tuple[int, int, int, int] | list[int],
    initial_region_type: str,
    initial_detector_confidence: float | None,
    *,
    config: ScreeningProfile | CandidateScreeningConfig = "baseline",
) -> CandidateScreeningResult:
    """Classify one proposed page region without invoking any OCSR model."""
    settings = get_screening_config(config)
    image = _read_image(page_image)
    page_height, page_width = image.shape[:2]
    if len(bbox) != 4:
        raise ValueError("bbox must contain four coordinates.")
    x1, y1, x2, y2 = [int(value) for value in bbox]
    x1, x2 = sorted((max(0, x1), min(page_width, x2)))
    y1, y2 = sorted((max(0, y1), min(page_height, y2)))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("bbox must define a non-empty region.")
    crop = image[y1:y2, x1:x2]
    height, width = crop.shape[:2]
    binary = _foreground_binary(crop)
    area = max(width * height, 1)
    ink_ratio = float(np.count_nonzero(binary) / area)
    aspect = width / max(height, 1)
    component_count, _, stats, _ = cv2.connectedComponentsWithStats((binary > 0).astype(np.uint8), 8)
    component_areas = [int(stats[index, cv2.CC_STAT_AREA]) for index in range(1, component_count)]
    thresholds = settings.features
    significant = [value for value in component_areas if value >= thresholds.component_min_area]
    small_ratio = sum(
        thresholds.component_min_area <= value <= thresholds.small_component_max_area
        for value in component_areas
    ) / max(len(component_areas), 1)
    text_lines = _text_line_count(binary)
    horizontal_projection = float(np.max(np.sum(binary > 0, axis=1)) / max(width, 1))
    vertical_projection = float(np.max(np.sum(binary > 0, axis=0)) / max(height, 1))
    line_info = _line_diagnostics(binary, settings)
    clusters = _structure_cluster_count(binary, settings)
    page_area_ratio = area / max(page_width * page_height, 1)
    skeletal = (
        line_info["long_line_count"] >= thresholds.skeletal_min_long_lines
        and len(significant) <= thresholds.skeletal_max_components
        and ink_ratio < thresholds.skeletal_max_ink_ratio
        and small_ratio < thresholds.skeletal_max_small_component_ratio
        and horizontal_projection < thresholds.skeletal_max_projection_ratio
        and vertical_projection < thresholds.skeletal_max_projection_ratio
    )
    diagnostics = {
        "config": settings.name, "config_values": asdict(settings),
        "width": width, "height": height, "aspect": round(aspect, 4),
        "ink_ratio": round(ink_ratio, 6), "component_count": max(component_count - 1, 0),
        "significant_component_count": len(significant), "small_component_ratio": round(small_ratio, 4),
        "text_line_count": text_lines, "horizontal_projection": round(horizontal_projection, 4),
        "vertical_projection": round(vertical_projection, 4), "page_area_ratio": round(page_area_ratio, 6),
        "structure_cluster_count": clusters, "skeletal_linework": skeletal,
        "initial_region_type": initial_region_type,
        "initial_detector_confidence": initial_detector_confidence,
        **{key: round(value, 4) if isinstance(value, float) else value for key, value in line_info.items()},
    }
    initial = _normalize_initial(initial_region_type)

    def result(label: str, score: float, *reasons: str) -> CandidateScreeningResult:
        return CandidateScreeningResult(
            recommended_region_type=label,
            molecule_candidate=label == "molecule",
            screening_score=round(max(0.0, min(score, 1.0)), 4),
            reason_codes=tuple(dict.fromkeys(reasons)), diagnostics=diagnostics,
        )

    if ink_ratio < settings.blank_ink_ratio:
        return result("blank", 1.0 - ink_ratio, "blank")
    if width < settings.min_width or height < settings.min_height:
        return result("text", 0.75, "too_small", "text_like")
    table_like = (
        thresholds.table_min_aspect <= aspect <= thresholds.table_max_aspect
        and horizontal_projection > settings.table_projection_ratio
        and vertical_projection > settings.table_projection_ratio
    )
    if table_like:
        return result("table", 0.88, "table_like")
    sparse_arrow = (
        initial in {"molecule", "reaction"}
        and len(significant) <= settings.sparse_arrow_max_components
        and ink_ratio <= settings.sparse_arrow_max_ink_ratio
        and line_info["long_horizontal_line_count"] >= 1
        and line_info["longest_horizontal_ratio"] >= settings.sparse_arrow_min_horizontal_ratio
    )
    arrow_like = sparse_arrow or (
        initial in {"molecule", "reaction"}
        and
        aspect >= settings.arrow_min_aspect and width >= settings.arrow_min_width
        and height <= settings.arrow_max_height and ink_ratio <= settings.arrow_max_ink_ratio
        and line_info["long_horizontal_line_count"] >= 1
        and text_lines <= thresholds.arrow_max_text_lines
    )
    mixed_arrow = arrow_like and (
        clusters >= thresholds.mixed_arrow_min_clusters
        or len(significant) >= thresholds.mixed_arrow_min_components
        or text_lines >= thresholds.mixed_arrow_min_text_lines
    )
    if arrow_like:
        return result(
            "reaction", 0.90 if mixed_arrow else 0.84, "reaction_arrow",
            *(("multiple_or_merged_region",) if mixed_arrow else ()),
        )
    merged_region = (
        settings.name == "candidate" and initial == "molecule"
        and clusters >= settings.multi_cluster_min_count
        and (
            len(significant) >= settings.merged_min_components
            or line_info["long_line_count"] >= settings.merged_min_long_lines
        )
    )
    if merged_region:
        return result("multiple_molecules", 0.82, "multiple_or_merged_region")
    strong_single_structure = (
        settings.reclassify_non_molecule and clusters == 1
        and settings.promotion_min_ink_ratio <= ink_ratio <= settings.promotion_max_ink_ratio
        and settings.promotion_min_components <= len(significant) <= settings.promotion_max_components
        and settings.promotion_min_small_component_ratio <= small_ratio <= settings.promotion_max_small_component_ratio
        and text_lines <= settings.promotion_max_text_lines and aspect <= settings.promotion_max_aspect
    )
    if strong_single_structure:
        return result("molecule", 0.91, "possible_molecule")
    existing_single_structure = (
        settings.name == "candidate" and initial == "molecule" and clusters == 1
        and settings.existing_molecule_min_ink_ratio <= ink_ratio <= settings.promotion_max_ink_ratio
        and settings.existing_molecule_min_components <= len(significant) <= settings.promotion_max_components
        and settings.existing_molecule_min_aspect <= aspect <= settings.existing_molecule_max_aspect
    )
    if existing_single_structure:
        return result("molecule", 0.89, "possible_molecule")
    reaction_condition = (
        settings.reaction_condition_min_aspect <= aspect <= thresholds.reaction_condition_max_aspect
        and settings.min_height <= height <= settings.reaction_condition_max_height
        and thresholds.reaction_condition_min_ink_ratio <= ink_ratio <= thresholds.reaction_condition_max_ink_ratio
        and thresholds.reaction_condition_min_text_lines <= text_lines <= thresholds.reaction_condition_max_text_lines
        and thresholds.reaction_condition_min_components <= len(significant) <= thresholds.reaction_condition_max_components
        and small_ratio >= thresholds.reaction_condition_min_small_ratio
    )
    if reaction_condition and initial == "reaction":
        return result("reaction", 0.78, "reaction_condition")
    text_like = (
        (text_lines >= settings.text_min_lines and len(significant) >= settings.text_min_components and not skeletal)
        or (
            len(significant) >= thresholds.dense_text_min_components
            and small_ratio > settings.text_small_component_ratio
            and ink_ratio < thresholds.dense_text_max_ink_ratio
        )
        or (
            height <= thresholds.short_text_max_height and aspect > thresholds.short_text_min_aspect
            and len(significant) >= thresholds.short_text_min_components
            and ink_ratio < thresholds.short_text_max_ink_ratio
        )
        or (
            len(significant) >= thresholds.body_text_min_components and aspect > thresholds.body_text_min_aspect
            and height < thresholds.body_text_max_height and ink_ratio < thresholds.body_text_max_ink_ratio
            and not skeletal
        )
    )
    if text_like:
        return result("text", 0.86, "text_like")
    if ink_ratio > settings.dense_ink_ratio and line_info["edge_ratio"] > settings.dense_figure_edge_ratio:
        return result("figure", 0.76, "dense_figure")

    molecule_score = 0.24
    molecule_score += 0.18 if thresholds.score_ink_min <= ink_ratio <= thresholds.score_ink_max else 0.0
    molecule_score += 0.14 if line_info["edge_ratio"] > thresholds.score_edge_min else 0.0
    molecule_score += 0.12 if len(significant) >= thresholds.score_component_min else 0.0
    molecule_score += 0.18 if skeletal else 0.0
    molecule_score += 0.10 if thresholds.score_aspect_min <= aspect <= thresholds.score_aspect_max else 0.0
    molecule_score += 0.06 if clusters == 1 else 0.0
    molecule_score += float(initial_detector_confidence or 0.0) * 0.08
    diagnostics["molecule_score"] = round(molecule_score, 4)
    if (
        settings.name == "candidate" and initial == "molecule"
        and ink_ratio >= settings.high_ink_uncertain_ratio and not skeletal
    ):
        return result("uncertain", molecule_score, "uncertain")
    if molecule_score >= settings.molecule_score_threshold and initial == "molecule":
        return result("molecule", molecule_score, "possible_molecule")
    if not settings.reclassify_non_molecule and initial != "molecule":
        return result(initial, float(initial_detector_confidence or 0.5), "uncertain")
    if initial != "molecule":
        return result(initial, molecule_score, "uncertain")
    return result("uncertain", molecule_score, "uncertain")

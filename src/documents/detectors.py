"""Molecule-region detection interfaces and lightweight OpenCV fallback."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import config
from src.documents.models import DocumentPage, DocumentRegion


class BaseMoleculeRegionDetector(ABC):
    """Detector interface for future trainable molecule-region detectors."""

    name = "base"

    @abstractmethod
    def detect(self, page: DocumentPage) -> list[DocumentRegion]:
        """Return detected regions for one rendered page."""
        raise NotImplementedError


def page_quality(image: np.ndarray) -> dict[str, Any]:
    """Compute simple page-quality diagnostics before region detection."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    foreground = gray < 245
    height, width = gray.shape[:2]
    contrast = float(np.percentile(gray, 95) - np.percentile(gray, 5))
    ink_ratio = float(np.mean(foreground))
    return {
        "width": int(width),
        "height": int(height),
        "pixel_count": int(width * height),
        "contrast": round(contrast, 3),
        "ink_ratio": round(ink_ratio, 5),
        "blank": bool(ink_ratio < 0.0005 or (contrast < 5.0 and ink_ratio < 0.002)),
        "too_large": bool(width * height > config.DOCUMENT_MAX_PIXELS),
    }


class HeuristicMoleculeRegionDetector(BaseMoleculeRegionDetector):
    """Detect molecule-like drawing regions without a large ML model."""

    name = "heuristic-opencv"

    def __init__(
        self,
        min_area: int = config.DOCUMENT_MIN_REGION_AREA,
        max_area_ratio: float = config.DOCUMENT_MAX_REGION_AREA_RATIO,
        max_regions: int = config.DOCUMENT_MAX_REGIONS,
    ) -> None:
        self.min_area = min_area
        self.max_area_ratio = max_area_ratio
        self.max_regions = max_regions

    def detect(self, page: DocumentPage) -> list[DocumentRegion]:
        image = cv2.imdecode(np.fromfile(str(Path(page.image_path)), dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Unable to decode page image: {page.image_path}")
        quality = page_quality(image)
        page.quality = quality
        if quality["blank"] or quality["too_large"]:
            return []
        binary = self._foreground_binary(image)
        contours = self._candidate_contours(binary, image.shape[1], image.shape[0])
        regions: list[DocumentRegion] = []
        for bbox in contours:
            region_type, confidence, message = self._classify(binary, bbox, image.shape[1], image.shape[0])
            if region_type == "unknown" and confidence < 0.2:
                continue
            region_id = f"p{page.page_number:03d}_r{len(regions) + 1:03d}"
            regions.append(DocumentRegion(
                document_id=page.document_id,
                page_number=page.page_number,
                region_id=region_id,
                bbox=bbox,
                region_type=region_type,
                detection_confidence=round(confidence, 3),
                detector_name=self.name,
                message=message,
            ))
            if len(regions) >= self.max_regions:
                break
        if not regions:
            fallback = self._whole_page_region(page, binary, image.shape[1], image.shape[0])
            if fallback is not None:
                regions.append(fallback)
        return regions

    @staticmethod
    def _foreground_binary(image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)
        return binary

    def _candidate_contours(self, binary: np.ndarray, width: int, height: int) -> list[tuple[int, int, int, int]]:
        # A moderate dilation joins bonds, atom labels, and nearby ring strokes into one region.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (23, 17))
        merged = cv2.dilate(binary, kernel, iterations=1)
        contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates: list[tuple[int, int, int, int]] = []
        page_area = width * height
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            if area < self.min_area:
                continue
            if area / max(page_area, 1) > self.max_area_ratio:
                continue
            if w < 35 or h < 25:
                continue
            if w / max(h, 1) > 9 and not (w >= 180 and h >= 18):
                continue
            if h / max(w, 1) > 6:
                continue
            padding = 8
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(width, x + w + padding)
            y2 = min(height, y + h + padding)
            candidates.append((x1, y1, x2, y2))
        return sorted(candidates, key=lambda item: (item[1], item[0]))

    def _classify(
        self,
        binary: np.ndarray,
        bbox: tuple[int, int, int, int],
        page_width: int,
        page_height: int,
    ) -> tuple[str, float, str]:
        x1, y1, x2, y2 = bbox
        crop = binary[y1:y2, x1:x2]
        width, height = x2 - x1, y2 - y1
        area = max(width * height, 1)
        ink_ratio = float(np.count_nonzero(crop) / area)
        component_count, labels, stats, _ = cv2.connectedComponentsWithStats((crop > 0).astype(np.uint8), 8)
        component_areas = [int(stats[index, cv2.CC_STAT_AREA]) for index in range(1, component_count)]
        significant_components = [value for value in component_areas if value >= 6]
        edges = cv2.Canny(crop, 60, 180)
        edge_ratio = float(np.count_nonzero(edges) / area)
        aspect = width / max(height, 1)
        horizontal_projection = np.max(np.sum(crop > 0, axis=1)) / max(width, 1)
        vertical_projection = np.max(np.sum(crop > 0, axis=0)) / max(height, 1)
        page_area_ratio = area / max(page_width * page_height, 1)

        if self._looks_like_table(crop, aspect, horizontal_projection, vertical_projection):
            return "table", 0.55, "Grid-like line structure; not sent to single-molecule OCSR by default."
        if self._looks_like_reaction(crop, aspect, width, height):
            return "reaction_like", 0.62, "Wide arrow/plus-like region; reaction parsing is not supported yet."
        if self._looks_like_text(width, height, aspect, ink_ratio, significant_components):
            return "text", 0.5, "Text-like compact components; not treated as a molecule."

        confidence = 0.25
        if 0.01 <= ink_ratio <= 0.28:
            confidence += 0.2
        if edge_ratio > 0.02:
            confidence += 0.15
        if len(significant_components) >= 3:
            confidence += 0.15
        if 0.25 <= aspect <= 4.5:
            confidence += 0.12
        if 0.003 <= page_area_ratio <= 0.55:
            confidence += 0.08
        confidence = min(confidence, 0.95)
        if confidence >= 0.55:
            return "molecule", confidence, "Detected by OpenCV line/foreground-density fallback."
        return "unknown", confidence, "Region did not meet molecule confidence threshold."

    def _whole_page_region(
        self,
        page: DocumentPage,
        binary: np.ndarray,
        page_width: int,
        page_height: int,
    ) -> DocumentRegion | None:
        coordinates = cv2.findNonZero((binary > 0).astype(np.uint8))
        if coordinates is None:
            return None
        x, y, width, height = cv2.boundingRect(coordinates)
        padding = 16
        bbox = (
            max(0, x - padding),
            max(0, y - padding),
            min(page_width, x + width + padding),
            min(page_height, y + height + padding),
        )
        region_type, confidence, message = self._classify(binary, bbox, page_width, page_height)
        if region_type != "molecule" or confidence < 0.55:
            return None
        return DocumentRegion(
            document_id=page.document_id,
            page_number=page.page_number,
            region_id=f"p{page.page_number:03d}_r001",
            bbox=bbox,
            region_type="molecule",
            detection_confidence=round(min(confidence, 0.82), 3),
            detector_name=self.name,
            message=message + " Whole-page fallback was used.",
        )

    @staticmethod
    def _looks_like_text(
        width: int,
        height: int,
        aspect: float,
        ink_ratio: float,
        significant_components: list[int],
    ) -> bool:
        if height <= 45 and aspect > 2.2 and len(significant_components) >= 3:
            return True
        if aspect > 5 and ink_ratio < 0.18 and len(significant_components) >= 5:
            return True
        if width < 140 and height < 60 and len(significant_components) >= 2:
            return True
        return False

    @staticmethod
    def _looks_like_table(crop: np.ndarray, aspect: float, horizontal_projection: float, vertical_projection: float) -> bool:
        if aspect < 0.5 or aspect > 6:
            return False
        horizontal_lines = horizontal_projection > 0.65
        vertical_lines = vertical_projection > 0.65
        return bool(horizontal_lines and vertical_lines)

    @staticmethod
    def _looks_like_reaction(crop: np.ndarray, aspect: float, width: int, height: int) -> bool:
        if aspect < 2.5 or width < 180:
            return False
        lines = cv2.HoughLinesP(crop, 1, np.pi / 180, threshold=40, minLineLength=max(60, width // 4), maxLineGap=8)
        if lines is None:
            return False
        long_horizontal = 0
        for line in lines.reshape(-1, 4):
            x1, y1, x2, y2 = [int(value) for value in line]
            length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
            if length > width * 0.25 and abs(y2 - y1) <= max(4, height * 0.08):
                long_horizontal += 1
        return long_horizontal >= 1

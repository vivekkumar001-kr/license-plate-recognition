"""Face localisation with OpenCV.

A frontal Haar cascade carries most of the load; a profile cascade is run as
a second opinion so heads turned away from the camera are not silently
dropped. Results are merged with NMS, exactly as in the plate ensemble.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..config import FaceGeometry
from .base import Detection, Detector, clamp_box, non_max_suppression

__all__ = ["FaceDetector"]


class _CascadeDetector(Detector):
    """Thin, reusable wrapper around a single Haar cascade file."""

    def __init__(
        self,
        cascade_file: str,
        name: str,
        base_score: float,
        min_size: tuple[int, int] = (40, 40),
    ) -> None:
        self.name = name
        self.base_score = base_score
        self.min_size = min_size
        self._cascade = cv2.CascadeClassifier(cv2.data.haarcascades + cascade_file)
        self.available = not self._cascade.empty()

    def detect(self, image: np.ndarray) -> list[Detection]:
        if not self.available:
            return []

        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        boxes = self._cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=self.min_size,
        )

        detections = []
        for box in boxes:
            x, y, width, height = clamp_box(tuple(int(v) for v in box), image.shape)
            detections.append(
                Detection(x, y, width, height, score=self.base_score, source=self.name)
            )
        return detections


class FaceDetector(Detector):
    """Frontal + profile Haar ensemble."""

    name = "face"

    def __init__(
        self,
        geometry: FaceGeometry | None = None,
        iou_threshold: float = 0.35,
        allow_whole_image_fallback: bool = True,
    ) -> None:
        self.geometry = geometry or FaceGeometry()
        self.iou_threshold = iou_threshold
        self.allow_whole_image_fallback = allow_whole_image_fallback
        self.strategies = [
            _CascadeDetector("haarcascade_frontalface_default.xml", "haar-frontal", 0.90),
            _CascadeDetector("haarcascade_profileface.xml", "haar-profile", 0.70),
        ]

    def _passes_shape_filter(self, detection: Detection) -> bool:
        geometry = self.geometry
        if detection.area < geometry.min_area:
            return False
        return geometry.min_aspect_ratio <= detection.aspect_ratio <= geometry.max_aspect_ratio

    def detect(self, image: np.ndarray) -> list[Detection]:
        if image is None or image.size == 0:
            return []

        candidates: list[Detection] = []
        for strategy in self.strategies:
            candidates.extend(strategy.detect(image))
        candidates = [item for item in candidates if self._passes_shape_filter(item)]

        if not candidates:
            if not self.allow_whole_image_fallback:
                return []
            height, width = image.shape[:2]
            fallback = Detection(0, 0, width, height, score=0.25, source="whole-image")
            return [fallback] if self._passes_shape_filter(fallback) else []

        return non_max_suppression(candidates, self.iou_threshold)

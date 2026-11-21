"""License-plate localisation with OpenCV.

Two complementary classical strategies are used and their votes merged:

``HaarPlateDetector``
    A Viola-Jones cascade. Fast and precise on frontal, well-lit plates;
    blind to unusual angles.
``ContourPlateDetector``
    Blackhat morphology plus a Scharr gradient to isolate the dense
    horizontal stroke texture that characterises plate glyphs, then contour
    filtering on aspect ratio and fill. Angle-tolerant but noisier.

Neither is good enough alone. Together, with NMS on top, they cover for each
other's blind spots -- a cheap ensemble that needs no training data.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..config import PlateGeometry
from .base import Detection, Detector, clamp_box, non_max_suppression

__all__ = ["HaarPlateDetector", "ContourPlateDetector", "PlateDetector"]

_IDEAL_ASPECT_RATIO = 3.6


def _aspect_plausibility(aspect_ratio: float, geometry: PlateGeometry) -> float:
    """Score in [0, 1] for how plate-shaped a box is."""
    if not geometry.min_aspect_ratio <= aspect_ratio <= geometry.max_aspect_ratio:
        return 0.0
    spread = max(
        _IDEAL_ASPECT_RATIO - geometry.min_aspect_ratio,
        geometry.max_aspect_ratio - _IDEAL_ASPECT_RATIO,
    )
    return float(1.0 - abs(aspect_ratio - _IDEAL_ASPECT_RATIO) / spread)


class HaarPlateDetector(Detector):
    """Viola-Jones cascade shipped with OpenCV."""

    name = "haar-plate"
    cascade_file = "haarcascade_russian_plate_number.xml"

    def __init__(self, geometry: PlateGeometry | None = None) -> None:
        self.geometry = geometry or PlateGeometry()
        path = cv2.data.haarcascades + self.cascade_file
        self._cascade = cv2.CascadeClassifier(path)
        self.available = not self._cascade.empty()

    def detect(self, image: np.ndarray) -> list[Detection]:
        if not self.available:
            return []

        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        boxes, _, weights = self._cascade.detectMultiScale3(
            gray,
            scaleFactor=1.08,
            minNeighbors=4,
            minSize=(44, 14),
            outputRejectLevels=True,
        )

        detections: list[Detection] = []
        for box, weight in zip(boxes, weights):
            x, y, width, height = clamp_box(tuple(int(v) for v in box), image.shape)
            # Cascade weights are unbounded log-odds; squash to a usable score.
            confidence = float(1.0 / (1.0 + np.exp(-float(weight))))
            detections.append(
                Detection(x, y, width, height, score=0.55 + 0.45 * confidence, source=self.name)
            )
        return sorted(detections, key=lambda item: item.score, reverse=True)


class ContourPlateDetector(Detector):
    """Morphology-and-contour plate finder."""

    name = "contour-plate"

    def __init__(self, geometry: PlateGeometry | None = None, max_candidates: int = 12) -> None:
        self.geometry = geometry or PlateGeometry()
        self.max_candidates = max_candidates
        self._wide_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5))
        self._square_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    def _text_mask(self, gray: np.ndarray) -> np.ndarray:
        """Highlight regions with dense, horizontally-repeating strokes."""
        smoothed = cv2.bilateralFilter(gray, 7, 45, 45)
        blackhat = cv2.morphologyEx(smoothed, cv2.MORPH_BLACKHAT, self._wide_kernel)

        gradient = cv2.Sobel(blackhat, cv2.CV_32F, 1, 0, ksize=3)
        gradient = np.absolute(gradient)
        span = gradient.max() - gradient.min()
        if span < 1e-6:
            return np.zeros_like(gray)
        gradient = (255.0 * (gradient - gradient.min()) / span).astype(np.uint8)

        gradient = cv2.GaussianBlur(gradient, (5, 5), 0)
        gradient = cv2.morphologyEx(gradient, cv2.MORPH_CLOSE, self._wide_kernel)
        _, mask = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._wide_kernel)
        return cv2.erode(mask, self._square_kernel, iterations=1)

    def detect(self, image: np.ndarray) -> list[Detection]:
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mask = self._text_mask(gray)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        geometry = self.geometry
        detections: list[Detection] = []
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[
            : self.max_candidates
        ]:
            x, y, width, height = clamp_box(cv2.boundingRect(contour), image.shape)
            if width * height < geometry.min_area:
                continue

            plausibility = _aspect_plausibility(width / height, geometry)
            if plausibility <= 0.0:
                continue

            # Fill ratio rejects L-shaped blobs that merely bound a plate-ish box.
            fill = cv2.contourArea(contour) / float(width * height)
            if fill < 0.35:
                continue

            score = 0.45 * plausibility + 0.35 * fill + 0.20
            detections.append(Detection(x, y, width, height, score=score, source=self.name))

        return sorted(detections, key=lambda item: item.score, reverse=True)


class PlateDetector(Detector):
    """Ensemble of the Haar and contour detectors with NMS and a fallback."""

    name = "plate"

    def __init__(
        self,
        geometry: PlateGeometry | None = None,
        iou_threshold: float = 0.35,
        allow_whole_image_fallback: bool = True,
    ) -> None:
        self.geometry = geometry or PlateGeometry()
        self.iou_threshold = iou_threshold
        self.allow_whole_image_fallback = allow_whole_image_fallback
        self.strategies: list[Detector] = [
            HaarPlateDetector(self.geometry),
            ContourPlateDetector(self.geometry),
        ]

    def _whole_image(self, image: np.ndarray) -> list[Detection]:
        """Treat an already-cropped plate as its own detection.

        Users routinely feed in a tight crop; refusing to recognise it just
        because no *sub*-plate was found would be obtuse.
        """
        height, width = image.shape[:2]
        if not self.allow_whole_image_fallback:
            return []
        if _aspect_plausibility(width / height, self.geometry) <= 0.0:
            return []
        return [Detection(0, 0, width, height, score=0.30, source="whole-image")]

    def detect(self, image: np.ndarray) -> list[Detection]:
        if image is None or image.size == 0:
            return []

        candidates: list[Detection] = []
        for strategy in self.strategies:
            candidates.extend(strategy.detect(image))

        if not candidates:
            return self._whole_image(image)
        return non_max_suppression(candidates, self.iou_threshold)

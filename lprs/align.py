"""Geometric alignment applied between detection and preprocessing.

Alignment is kept separate from both the detector and the preprocessor
because it is the only stage that needs *landmark* knowledge. Keeping it
behind a tiny interface means the plate pipeline simply uses
:class:`NullAligner` and pays nothing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import cv2
import numpy as np

__all__ = ["Aligner", "NullAligner", "EyeAligner"]


class Aligner(ABC):
    """Warps a crop into a canonical pose."""

    @abstractmethod
    def align(self, crop: np.ndarray) -> np.ndarray:
        """Return the aligned crop (or the input unchanged if impossible)."""

    def __call__(self, crop: np.ndarray) -> np.ndarray:
        return self.align(crop)


class NullAligner(Aligner):
    """Does nothing, on purpose."""

    def align(self, crop: np.ndarray) -> np.ndarray:
        return crop


class EyeAligner(Aligner):
    """Rotates a face so the inter-ocular line is horizontal.

    This is the cheap stand-in for FaceNet's MTCNN five-point alignment:
    pose normalisation removes a large chunk of intra-identity variance, so
    the embedding space ends up measuring *who* rather than *how tilted*.
    """

    def __init__(self, max_degrees: float = 30.0) -> None:
        self.max_degrees = max_degrees
        self._eyes = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye.xml"
        )
        self.available = not self._eyes.empty()

    def _eye_centres(self, gray: np.ndarray) -> list[tuple[float, float]]:
        boxes = self._eyes.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6)
        centres = [(x + w / 2.0, y + h / 2.0) for x, y, w, h in boxes]
        # Keep the two largest-area eyes, ordered left to right.
        ordered = sorted(
            zip(centres, (w * h for _, _, w, h in boxes)),
            key=lambda pair: pair[1],
            reverse=True,
        )[:2]
        return sorted((centre for centre, _ in ordered), key=lambda point: point[0])

    def align(self, crop: np.ndarray) -> np.ndarray:
        if not self.available or crop is None or crop.size == 0:
            return crop

        gray = crop if crop.ndim == 2 else cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        centres = self._eye_centres(gray)
        if len(centres) < 2:
            return crop

        (left_x, left_y), (right_x, right_y) = centres
        angle = np.degrees(np.arctan2(right_y - left_y, right_x - left_x))
        if abs(angle) < 1.0 or abs(angle) > self.max_degrees:
            return crop

        height, width = crop.shape[:2]
        eye_midpoint = ((left_x + right_x) / 2.0, (left_y + right_y) / 2.0)
        matrix = cv2.getRotationMatrix2D(eye_midpoint, angle, 1.0)
        return cv2.warpAffine(
            crop,
            matrix,
            (width, height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )

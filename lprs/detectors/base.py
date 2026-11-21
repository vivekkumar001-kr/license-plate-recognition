"""Detector abstractions shared by the plate and face pipelines.

The rest of the system depends on the :class:`Detector` interface, never on a
concrete detector. Swapping Haar cascades for a YOLO head later is therefore a
one-line change in the factory -- the Dependency Inversion Principle earning
its keep.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

__all__ = ["Detection", "Detector", "non_max_suppression", "clamp_box"]


def clamp_box(
    box: tuple[int, int, int, int],
    image_shape: tuple[int, ...],
) -> tuple[int, int, int, int]:
    """Clip an (x, y, w, h) box so it stays inside the image."""
    height, width = image_shape[:2]
    x, y, box_width, box_height = box
    x = max(0, min(int(x), width - 1))
    y = max(0, min(int(y), height - 1))
    box_width = max(1, min(int(box_width), width - x))
    box_height = max(1, min(int(box_height), height - y))
    return x, y, box_width, box_height


@dataclass(frozen=True)
class Detection:
    """A single detected region of interest."""

    x: int
    y: int
    width: int
    height: int
    score: float = 1.0
    source: str = "unknown"

    @property
    def box(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.width, self.height)

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height if self.height else 0.0

    @property
    def xyxy(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.x + self.width, self.y + self.height)

    def intersection_over_union(self, other: "Detection") -> float:
        ax1, ay1, ax2, ay2 = self.xyxy
        bx1, by1, bx2, by2 = other.xyxy
        inter_width = max(0, min(ax2, bx2) - max(ax1, bx1))
        inter_height = max(0, min(ay2, by2) - max(ay1, by1))
        intersection = inter_width * inter_height
        if intersection == 0:
            return 0.0
        union = self.area + other.area - intersection
        return intersection / union if union else 0.0

    def crop(self, image: np.ndarray, margin: float = 0.0) -> np.ndarray:
        """Extract this region, optionally growing the box by a margin."""
        pad_x = int(round(self.width * margin))
        pad_y = int(round(self.height * margin))
        x, y, width, height = clamp_box(
            (self.x - pad_x, self.y - pad_y, self.width + 2 * pad_x, self.height + 2 * pad_y),
            image.shape,
        )
        return image[y : y + height, x : x + width]


def non_max_suppression(
    detections: list[Detection],
    iou_threshold: float = 0.35,
) -> list[Detection]:
    """Greedy NMS -- keeps the highest-scoring box in each overlapping cluster.

    Needed because we deliberately run several detection strategies and let
    them vote, which produces duplicates by design.
    """
    ordered = sorted(detections, key=lambda item: item.score, reverse=True)
    kept: list[Detection] = []
    for candidate in ordered:
        if all(
            candidate.intersection_over_union(existing) <= iou_threshold
            for existing in kept
        ):
            kept.append(candidate)
    return kept


class Detector(ABC):
    """Locates regions of interest in a full-frame BGR image."""

    #: Human-readable name used in logs and the web UI.
    name: str = "detector"

    @abstractmethod
    def detect(self, image: np.ndarray) -> list[Detection]:
        """Return detections sorted by descending confidence."""

    def detect_best(self, image: np.ndarray) -> Detection | None:
        detections = self.detect(image)
        return detections[0] if detections else None

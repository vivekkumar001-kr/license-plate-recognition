"""Detector implementations and the factory that selects one."""

from __future__ import annotations

from .base import Detection, Detector, clamp_box, non_max_suppression
from .face import FaceDetector
from .plate import ContourPlateDetector, HaarPlateDetector, PlateDetector

__all__ = [
    "Detection",
    "Detector",
    "FaceDetector",
    "ContourPlateDetector",
    "HaarPlateDetector",
    "PlateDetector",
    "clamp_box",
    "non_max_suppression",
    "build_detector",
]


def build_detector(domain: str) -> Detector:
    """Return the detector registered for ``domain`` ('plate' or 'face')."""
    builders: dict[str, type[Detector]] = {
        "plate": PlateDetector,
        "face": FaceDetector,
    }
    try:
        return builders[domain]()
    except KeyError:
        raise ValueError(
            f"Unknown detection domain {domain!r}; expected one of {sorted(builders)}"
        ) from None

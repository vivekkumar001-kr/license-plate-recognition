"""Training-free baseline embedders.

These exist so the CNN has something honest to be compared against. A deep
model that cannot beat a histogram of oriented gradients is not earning its
parameters, and without the baseline you would never find out.
"""

from __future__ import annotations

import cv2
import numpy as np

from .base import Embedder, l2_normalise

__all__ = ["RawPixelEmbedder", "HOGEmbedder"]


def _as_uint8_gray(image: np.ndarray) -> np.ndarray:
    """Coerce a float [0, 1] or uint8 crop to a single-channel uint8 image."""
    array = np.asarray(image)
    if array.dtype != np.uint8:
        array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    if array.ndim == 3:
        array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    return array


class RawPixelEmbedder(Embedder):
    """Flattened, downscaled pixels. The floor of the leaderboard."""

    name = "raw-pixel"

    def __init__(self, size: tuple[int, int] = (32, 32)) -> None:
        self.size = size

    @property
    def embedding_dim(self) -> int:
        return self.size[0] * self.size[1]

    def embed(self, batch: np.ndarray) -> np.ndarray:
        if len(batch) == 0:
            return np.empty((0, self.embedding_dim), dtype=np.float32)
        rows = [
            cv2.resize(_as_uint8_gray(image), self.size, interpolation=cv2.INTER_AREA).ravel()
            for image in batch
        ]
        return l2_normalise(np.asarray(rows, dtype=np.float32))


class HOGEmbedder(Embedder):
    """Histogram of Oriented Gradients -- a strong classical baseline.

    HOG encodes local edge orientation, which is exactly the signal that
    distinguishes one glyph sequence from another, so it is a genuinely
    competitive opponent on plates (far less so on faces).
    """

    name = "hog"

    def __init__(self, size: tuple[int, int]) -> None:
        # HOG needs the window to be a whole number of 8-pixel cells.
        self.size = (max(16, size[0] // 8 * 8), max(16, size[1] // 8 * 8))
        self._descriptor = cv2.HOGDescriptor(
            _winSize=self.size,
            _blockSize=(16, 16),
            _blockStride=(8, 8),
            _cellSize=(8, 8),
            _nbins=9,
        )
        self._dim = int(self._descriptor.getDescriptorSize())

    @property
    def embedding_dim(self) -> int:
        return self._dim

    def embed(self, batch: np.ndarray) -> np.ndarray:
        if len(batch) == 0:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        rows = []
        for image in batch:
            gray = cv2.resize(
                _as_uint8_gray(image), self.size, interpolation=cv2.INTER_AREA
            )
            rows.append(self._descriptor.compute(gray).ravel())
        return l2_normalise(np.asarray(rows, dtype=np.float32))

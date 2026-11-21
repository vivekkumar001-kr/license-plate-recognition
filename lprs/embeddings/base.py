"""Embedding abstractions.

An embedder maps a preprocessed crop to a fixed-length, L2-normalised vector.
Because every embedder guarantees unit norm, cosine similarity reduces to a
dot product and the matcher never needs to know which model produced the
vector -- the Liskov substitution property we actually rely on.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

__all__ = ["Embedder", "l2_normalise"]


def l2_normalise(vectors: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
    """Scale each row to unit length, tolerating all-zero rows."""
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim == 1:
        vectors = vectors[None, :]
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, epsilon)


class Embedder(ABC):
    """Turns preprocessed crops into unit-norm feature vectors."""

    name: str = "embedder"

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Dimensionality of the produced vectors."""

    @abstractmethod
    def embed(self, batch: np.ndarray) -> np.ndarray:
        """Embed a stacked batch of preprocessed crops.

        Args:
            batch: ``(N, H, W)`` or ``(N, H, W, C)`` float32 array in [0, 1].

        Returns:
            ``(N, embedding_dim)`` float32 array with unit-norm rows.
        """

    def embed_one(self, image: np.ndarray) -> np.ndarray:
        """Convenience wrapper for a single crop."""
        return self.embed(image[None, ...])[0]

"""Embedder implementations and the factory used to select one."""

from __future__ import annotations

from pathlib import Path

from .base import Embedder, l2_normalise
from .classical import HOGEmbedder, RawPixelEmbedder
from .cnn import EmbeddingNet, TorchEmbedder, to_nchw

__all__ = [
    "Embedder",
    "EmbeddingNet",
    "HOGEmbedder",
    "RawPixelEmbedder",
    "TorchEmbedder",
    "build_embedder",
    "l2_normalise",
    "to_nchw",
]


def build_embedder(
    kind: str,
    size: tuple[int, int],
    checkpoint: str | Path | None = None,
    device: str = "cpu",
) -> Embedder:
    """Construct an embedder by name.

    Args:
        kind: ``"cnn"``, ``"hog"`` or ``"raw"``.
        size: ``(width, height)`` of the preprocessed crop.
        checkpoint: Required for ``"cnn"`` -- path to trained weights.
        device: Torch device string.
    """
    if kind == "cnn":
        if checkpoint is None:
            raise ValueError("The 'cnn' embedder requires a trained checkpoint path")
        return TorchEmbedder.from_checkpoint(checkpoint, device=device)
    if kind == "hog":
        return HOGEmbedder(size)
    if kind == "raw":
        return RawPixelEmbedder()
    raise ValueError(f"Unknown embedder {kind!r}; expected 'cnn', 'hog' or 'raw'")

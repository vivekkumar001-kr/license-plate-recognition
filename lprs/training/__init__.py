"""Training loops for the embedder and the OCR head."""

from __future__ import annotations

from .train_embedder import EmbedderTrainingResult, train_embedder
from .train_ocr import OCRTrainingResult, train_ocr
from .triplet import BatchHardTripletLoss, pairwise_distances

__all__ = [
    "BatchHardTripletLoss",
    "EmbedderTrainingResult",
    "OCRTrainingResult",
    "pairwise_distances",
    "train_embedder",
    "train_ocr",
]

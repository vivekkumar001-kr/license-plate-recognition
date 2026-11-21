"""Similarity matching: embedding gallery plus text-level fallback."""

from __future__ import annotations

from .gallery import UNKNOWN_LABEL, Gallery, MatchResult
from .text import confusion_aware_distance, levenshtein, text_similarity

__all__ = [
    "Gallery",
    "MatchResult",
    "UNKNOWN_LABEL",
    "confusion_aware_distance",
    "levenshtein",
    "text_similarity",
]

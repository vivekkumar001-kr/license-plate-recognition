"""Text <-> index codec and CTC decoding.

Kept separate from the network so that decoding can be tested without
instantiating (or training) a model.
"""

from __future__ import annotations

import numpy as np

from ..config import ALPHABET, BLANK, CHAR_TO_INDEX, INDEX_TO_CHAR

__all__ = [
    "encode_text",
    "decode_indices",
    "greedy_decode",
    "collapse_repeats",
    "BLANK_INDEX",
]

BLANK_INDEX = CHAR_TO_INDEX[BLANK]


def encode_text(text: str) -> list[int]:
    """Map a plate string to class indices, silently dropping stray glyphs."""
    return [CHAR_TO_INDEX[char] for char in text.upper() if char in CHAR_TO_INDEX]


def decode_indices(indices) -> str:
    """Map class indices back to a string, skipping blanks."""
    return "".join(
        INDEX_TO_CHAR[int(index)]
        for index in indices
        if int(index) != BLANK_INDEX and int(index) in INDEX_TO_CHAR
    )


def collapse_repeats(indices) -> list[int]:
    """Apply the CTC collapse rule: merge runs, then drop blanks.

    ``[A, A, blank, A]`` decodes to ``"AA"``, not ``"A"`` -- the blank is what
    lets CTC represent genuine double letters.
    """
    collapsed: list[int] = []
    previous = None
    for raw in indices:
        index = int(raw)
        if index != previous:
            collapsed.append(index)
        previous = index
    return [index for index in collapsed if index != BLANK_INDEX]


def greedy_decode(log_probs: np.ndarray) -> tuple[str, float]:
    """Best-path decode of a ``(T, C)`` log-probability matrix.

    Returns the decoded string and a mean per-frame confidence, which is the
    cheapest useful proxy for "should I trust this read?".
    """
    array = np.asarray(log_probs)
    if array.ndim != 2:
        raise ValueError(f"Expected a (T, C) matrix, received shape {array.shape}")

    best_indices = array.argmax(axis=1)
    confidence = float(np.exp(array[np.arange(len(best_indices)), best_indices]).mean())
    return decode_indices(collapse_repeats(best_indices)), confidence


assert len(ALPHABET) == len(set(ALPHABET)), "Alphabet must not contain duplicates"

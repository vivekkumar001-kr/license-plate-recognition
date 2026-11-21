"""String similarity for plate text, with OCR-confusion awareness.

Plate OCR fails in *predictable* ways: ``0``/``O``, ``1``/``I``, ``5``/``S``,
``8``/``B``, ``2``/``Z``. A plain Levenshtein distance charges full price for
those swaps, which makes a near-perfect read look as wrong as a garbage one.
Weighting the known confusion pairs at a fraction of a full edit fixes that
and measurably improves registry lookups.
"""

from __future__ import annotations

from functools import lru_cache

__all__ = [
    "CONFUSION_PAIRS",
    "levenshtein",
    "confusion_aware_distance",
    "text_similarity",
]

#: Glyph pairs that optical character recognition routinely swaps.
CONFUSION_PAIRS: frozenset[frozenset[str]] = frozenset(
    frozenset(pair)
    for pair in ("0O", "1I", "5S", "8B", "2Z", "6G", "1L", "UV")
)

_CONFUSION_COST = 0.35


def levenshtein(left: str, right: str) -> int:
    """Classic edit distance with unit costs."""
    return int(round(_weighted_distance(left, right, confusion_aware=False)))


def confusion_aware_distance(left: str, right: str) -> float:
    """Edit distance where known OCR confusions cost less than a full edit."""
    return _weighted_distance(left, right, confusion_aware=True)


@lru_cache(maxsize=4096)
def _substitution_cost(left_char: str, right_char: str) -> float:
    if left_char == right_char:
        return 0.0
    if frozenset((left_char, right_char)) in CONFUSION_PAIRS:
        return _CONFUSION_COST
    return 1.0


def _weighted_distance(left: str, right: str, confusion_aware: bool) -> float:
    """Wagner-Fischer with a configurable substitution cost."""
    if left == right:
        return 0.0
    if not left:
        return float(len(right))
    if not right:
        return float(len(left))

    previous = [float(index) for index in range(len(right) + 1)]
    for row, left_char in enumerate(left, start=1):
        current = [float(row)]
        for column, right_char in enumerate(right, start=1):
            if confusion_aware:
                substitution = _substitution_cost(left_char, right_char)
            else:
                substitution = 0.0 if left_char == right_char else 1.0
            current.append(
                min(
                    previous[column] + 1.0,  # deletion
                    current[column - 1] + 1.0,  # insertion
                    previous[column - 1] + substitution,  # substitution
                )
            )
        previous = current
    return previous[-1]


def text_similarity(left: str, right: str, confusion_aware: bool = True) -> float:
    """Normalised similarity in [0, 1]; 1.0 means identical."""
    longest = max(len(left), len(right))
    if longest == 0:
        return 1.0
    distance = _weighted_distance(left, right, confusion_aware)
    return max(0.0, 1.0 - distance / longest)

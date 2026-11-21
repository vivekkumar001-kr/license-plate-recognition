"""Evaluation metrics for every stage of the pipeline.

Three different questions need three different families of metric, and
conflating them is the classic way to report a number that means nothing:

* **Detection** -- did we find the object? (IoU, detection rate)
* **Identification** (closed set) -- who is it? (top-k accuracy, macro F1)
* **Verification** (open set) -- is this the same object? (ROC-AUC, EER,
  TAR@FAR). This is the one that actually matters for a gallery system,
  because it measures behaviour on identities never enrolled.
* **Transcription** -- what does it say? (exact match, character accuracy)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, roc_curve

from ..matching.text import levenshtein

__all__ = [
    "DetectionMetrics",
    "IdentificationMetrics",
    "OpenSetMetrics",
    "VerificationMetrics",
    "TranscriptionMetrics",
    "evaluate_detection",
    "evaluate_identification",
    "evaluate_open_set",
    "evaluate_verification",
    "evaluate_transcription",
    "character_accuracy",
]


def _round(value: float, places: int = 4) -> float:
    return float(round(float(value), places))


# --------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class DetectionMetrics:
    num_images: int
    detection_rate: float
    mean_iou: float
    mean_iou_on_hits: float
    iou_threshold: float

    def as_dict(self) -> dict:
        return {
            "num_images": self.num_images,
            "detection_rate": _round(self.detection_rate),
            "mean_iou": _round(self.mean_iou),
            "mean_iou_on_hits": _round(self.mean_iou_on_hits),
            "iou_threshold": self.iou_threshold,
        }


def evaluate_detection(ious: list[float], iou_threshold: float = 0.5) -> DetectionMetrics:
    """Summarise per-image best-IoU values."""
    array = np.asarray(ious, dtype=np.float32) if ious else np.zeros(0, dtype=np.float32)
    hits = array >= iou_threshold
    return DetectionMetrics(
        num_images=len(array),
        detection_rate=float(hits.mean()) if len(array) else 0.0,
        mean_iou=float(array.mean()) if len(array) else 0.0,
        mean_iou_on_hits=float(array[hits].mean()) if hits.any() else 0.0,
        iou_threshold=iou_threshold,
    )


# --------------------------------------------------------------------------
# Identification (closed set)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class IdentificationMetrics:
    num_probes: int
    top1_accuracy: float
    top5_accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float

    def as_dict(self) -> dict:
        return {
            "num_probes": self.num_probes,
            "top1_accuracy": _round(self.top1_accuracy),
            "top5_accuracy": _round(self.top5_accuracy),
            "macro_precision": _round(self.macro_precision),
            "macro_recall": _round(self.macro_recall),
            "macro_f1": _round(self.macro_f1),
        }


def evaluate_identification(
    truths: list[str],
    ranked_predictions: list[list[str]],
) -> IdentificationMetrics:
    """Score ranked candidate lists against ground-truth labels."""
    if len(truths) != len(ranked_predictions):
        raise ValueError("truths and ranked_predictions must be the same length")
    if not truths:
        return IdentificationMetrics(0, 0.0, 0.0, 0.0, 0.0, 0.0)

    top1 = [ranked[0] if ranked else "" for ranked in ranked_predictions]
    top1_hits = [truth == prediction for truth, prediction in zip(truths, top1)]
    top5_hits = [
        truth in ranked[:5] for truth, ranked in zip(truths, ranked_predictions)
    ]

    precision, recall, f1, _ = precision_recall_fscore_support(
        truths, top1, average="macro", zero_division=0
    )
    return IdentificationMetrics(
        num_probes=len(truths),
        top1_accuracy=float(np.mean(top1_hits)),
        top5_accuracy=float(np.mean(top5_hits)),
        macro_precision=float(precision),
        macro_recall=float(recall),
        macro_f1=float(f1),
    )


# --------------------------------------------------------------------------
# Verification (open set)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class VerificationMetrics:
    num_pairs: int
    roc_auc: float
    equal_error_rate: float
    best_threshold: float
    accuracy_at_best: float
    tar_at_far_1pct: float

    def as_dict(self) -> dict:
        return {
            "num_pairs": self.num_pairs,
            "roc_auc": _round(self.roc_auc),
            "equal_error_rate": _round(self.equal_error_rate),
            "best_threshold": _round(self.best_threshold),
            "accuracy_at_best": _round(self.accuracy_at_best),
            "tar_at_far_1pct": _round(self.tar_at_far_1pct),
        }


def evaluate_verification(
    similarities: np.ndarray,
    is_same: np.ndarray,
) -> VerificationMetrics:
    """Threshold-free quality of the embedding space.

    Also returns the empirically optimal acceptance threshold, which is how
    ``MatchConfig.accept_threshold`` should be chosen rather than guessed.
    """
    scores = np.asarray(similarities, dtype=np.float64).ravel()
    labels = np.asarray(is_same).astype(int).ravel()
    if len(scores) == 0 or len(np.unique(labels)) < 2:
        return VerificationMetrics(len(scores), 0.5, 0.5, 0.0, 0.0, 0.0)

    auc = float(roc_auc_score(labels, scores))
    false_positive_rate, true_positive_rate, thresholds = roc_curve(labels, scores)

    false_negative_rate = 1.0 - true_positive_rate
    eer_index = int(np.nanargmin(np.abs(false_positive_rate - false_negative_rate)))
    eer = float(
        (false_positive_rate[eer_index] + false_negative_rate[eer_index]) / 2.0
    )

    # Youden's J picks the threshold maximising (sensitivity + specificity - 1).
    # sklearn prepends a sentinel threshold of +inf ("classify nothing as
    # positive"); it is not a usable operating point, so drop it first.
    finite = np.isfinite(thresholds)
    youden = (true_positive_rate - false_positive_rate)[finite]
    finite_thresholds = thresholds[finite]
    if len(finite_thresholds) == 0:
        best_threshold = 0.0
    else:
        best_threshold = float(finite_thresholds[int(np.argmax(youden))])

    predictions = (scores >= best_threshold).astype(int)
    accuracy = float((predictions == labels).mean())

    within_budget = false_positive_rate <= 0.01
    tar_at_far = float(true_positive_rate[within_budget].max()) if within_budget.any() else 0.0

    return VerificationMetrics(
        num_pairs=len(scores),
        roc_auc=auc,
        equal_error_rate=eer,
        best_threshold=best_threshold,
        accuracy_at_best=accuracy,
        tar_at_far_1pct=tar_at_far,
    )


def build_pairs(
    embeddings: np.ndarray,
    labels: list[str],
    max_pairs: int = 20000,
    seed: int = 1337,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample balanced same/different pairs and return their similarities."""
    rng = np.random.default_rng(seed)
    count = len(labels)
    if count < 2:
        return np.zeros(0), np.zeros(0)

    total_possible = count * (count - 1) // 2
    label_array = np.asarray(labels)

    if total_possible <= max_pairs:
        left, right = np.triu_indices(count, k=1)
    else:
        left = rng.integers(0, count, size=max_pairs)
        right = rng.integers(0, count, size=max_pairs)
        keep = left != right
        left, right = left[keep], right[keep]

    similarities = np.einsum("ij,ij->i", embeddings[left], embeddings[right])
    is_same = label_array[left] == label_array[right]
    return similarities, is_same


# --------------------------------------------------------------------------
# Open-set identification (1:N)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class OpenSetMetrics:
    """Performance when a probe may belong to nobody in the gallery."""

    threshold: float
    num_genuine: int
    num_strangers: int
    gallery_size: int
    correct_identification_rate: float
    false_alarm_rate: float
    target_false_alarm: float

    def as_dict(self) -> dict:
        return {
            "threshold": _round(self.threshold),
            "num_genuine": self.num_genuine,
            "num_strangers": self.num_strangers,
            "gallery_size": self.gallery_size,
            "correct_identification_rate": _round(self.correct_identification_rate),
            "false_alarm_rate": _round(self.false_alarm_rate),
            "target_false_alarm": self.target_false_alarm,
        }


def evaluate_open_set(
    gallery_vectors: np.ndarray,
    gallery_labels: list[str],
    probe_vectors: np.ndarray,
    probe_labels: list[str],
    stranger_vectors: np.ndarray,
    target_false_alarm: float = 0.01,
) -> OpenSetMetrics:
    """Calibrate and score the 1:N open-set protocol.

    Why this exists when :func:`evaluate_verification` already reports AUC:
    verification asks "are these two the same?" and calibrates on *pairwise*
    similarity. Identification asks "who is this, if anyone?" and compares a
    probe against the **entire gallery**, keeping the maximum. Taking a max
    over N draws repeatedly samples the right-hand tail of the impostor
    distribution, so the usable threshold is far higher than the pairwise
    one. Measured on this system: mean impostor similarity 0.284, but the
    99th percentile is 0.933.

    Calibrating 1:1 and then deploying 1:N is precisely how a recogniser ends
    up accepting every stranger it meets -- which is the bug this function
    was written to kill. The threshold is set from the *max-over-gallery*
    score of known strangers.
    """
    gallery = np.asarray(gallery_vectors, dtype=np.float32)
    labels = np.asarray(gallery_labels)
    strangers = np.asarray(stranger_vectors, dtype=np.float32)

    if len(gallery) == 0 or len(strangers) == 0:
        return OpenSetMetrics(0.0, 0, 0, len(gallery), 0.0, 0.0, target_false_alarm)

    stranger_best = (strangers @ gallery.T).max(axis=1)
    # The score that only ``target_false_alarm`` of strangers may exceed.
    threshold = float(np.quantile(stranger_best, 1.0 - target_false_alarm))
    false_alarm_rate = float((stranger_best >= threshold).mean())

    probes = np.asarray(probe_vectors, dtype=np.float32)
    if len(probes) == 0:
        return OpenSetMetrics(
            threshold, 0, len(stranger_best), len(gallery),
            0.0, false_alarm_rate, target_false_alarm,
        )

    similarity = probes @ gallery.T
    best_index = similarity.argmax(axis=1)
    best_score = similarity[np.arange(len(probes)), best_index]

    # Correct only if the score clears the threshold AND names the right one.
    correct = (best_score >= threshold) & (labels[best_index] == np.asarray(probe_labels))

    return OpenSetMetrics(
        threshold=threshold,
        num_genuine=len(probes),
        num_strangers=len(stranger_best),
        gallery_size=len(gallery),
        correct_identification_rate=float(correct.mean()),
        false_alarm_rate=false_alarm_rate,
        target_false_alarm=target_false_alarm,
    )


# --------------------------------------------------------------------------
# Transcription
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TranscriptionMetrics:
    num_samples: int
    exact_match: float
    character_accuracy: float
    mean_edit_distance: float
    examples: list[tuple[str, str]] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "num_samples": self.num_samples,
            "exact_match": _round(self.exact_match),
            "character_accuracy": _round(self.character_accuracy),
            "mean_edit_distance": _round(self.mean_edit_distance),
            "examples": self.examples[:10],
        }


def character_accuracy(truth: str, prediction: str) -> float:
    """1 - CER, floored at zero.

    Character Error Rate is normalised by the *reference* length, so a
    hallucinated 30-character read against a 7-character plate correctly
    scores zero rather than a misleading partial credit.
    """
    if not truth:
        return 1.0 if not prediction else 0.0
    return max(0.0, 1.0 - levenshtein(truth, prediction) / len(truth))


def evaluate_transcription(
    truths: list[str],
    predictions: list[str],
) -> TranscriptionMetrics:
    if len(truths) != len(predictions):
        raise ValueError("truths and predictions must be the same length")
    if not truths:
        return TranscriptionMetrics(0, 0.0, 0.0, 0.0, [])

    exact = [truth == prediction for truth, prediction in zip(truths, predictions)]
    accuracies = [
        character_accuracy(truth, prediction)
        for truth, prediction in zip(truths, predictions)
    ]
    distances = [
        levenshtein(truth, prediction) for truth, prediction in zip(truths, predictions)
    ]
    mistakes = [
        (truth, prediction)
        for truth, prediction in zip(truths, predictions)
        if truth != prediction
    ]

    return TranscriptionMetrics(
        num_samples=len(truths),
        exact_match=float(np.mean(exact)),
        character_accuracy=float(np.mean(accuracies)),
        mean_edit_distance=float(np.mean(distances)),
        examples=mistakes,
    )

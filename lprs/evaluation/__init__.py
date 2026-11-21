"""Evaluation metrics and reporting."""

from __future__ import annotations

from .metrics import (
    DetectionMetrics,
    IdentificationMetrics,
    TranscriptionMetrics,
    VerificationMetrics,
    build_pairs,
    character_accuracy,
    evaluate_detection,
    evaluate_identification,
    evaluate_transcription,
    evaluate_verification,
)

__all__ = [
    "DetectionMetrics",
    "IdentificationMetrics",
    "TranscriptionMetrics",
    "VerificationMetrics",
    "build_pairs",
    "character_accuracy",
    "evaluate_detection",
    "evaluate_identification",
    "evaluate_transcription",
    "evaluate_verification",
]

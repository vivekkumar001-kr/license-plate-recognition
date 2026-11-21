"""Train both models end to end and record the training history.

Run via ``lprs train-all`` (or directly) -- it is a script rather than a
notebook so that every reported number is reproducible from a single seed.
"""

from __future__ import annotations

import json
from pathlib import Path

from lprs.config import ARTIFACT_DIR
from lprs.training import train_embedder, train_ocr


def main() -> None:
    history_path = ARTIFACT_DIR / "training_history.json"

    print("=" * 70)
    print("STAGE 1/2  FaceNet-style embedder (batch-hard triplet loss)")
    print("=" * 70)
    embedder_result = train_embedder(
        num_identities=180,
        samples_per_identity=10,
        epochs=16,
        severity=0.6,
        identities_per_batch=12,
        samples_per_batch_identity=4,
    )

    print()
    print("=" * 70)
    print("STAGE 2/2  CRNN + BiLSTM + CTC transcriber")
    print("=" * 70)
    ocr_result = train_ocr(
        num_plates=2400,
        views_per_plate=2,
        epochs=16,
        severity=0.55,
    )

    payload = {
        "embedder": {
            "checkpoint": str(embedder_result.checkpoint),
            "history": embedder_result.history,
            "final_metrics": embedder_result.final_metrics,
        },
        "ocr": {
            "checkpoint": str(ocr_result.checkpoint),
            "history": ocr_result.history,
            "final_metrics": ocr_result.final_metrics,
        },
    }
    Path(history_path).write_text(json.dumps(payload, indent=2))
    print(f"\ntraining history -> {history_path}")


if __name__ == "__main__":
    main()

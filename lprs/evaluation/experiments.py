"""Experiments: the numbers that go in the report.

Each function answers one question and returns plain dictionaries, so the
reporting layer never has to know how the measurement was made.

Note the recurring design choice: **gallery identities are disjoint from
anything the model trained on**, and probes are disjoint from the enrolled
samples. Anything less and you are measuring memorisation.
"""

from __future__ import annotations

import random
import time
from pathlib import Path

import numpy as np

from ..config import ARTIFACT_DIR, MatchConfig, PlateGeometry, PreprocessConfig
from ..data.synth_plates import (
    augment_plate,
    compose_scene,
    random_plate_text,
    render_plate,
)
from ..detectors import Detection, build_detector
from ..embeddings import build_embedder
from ..matching import Gallery
from ..ocr import PlateTranscriber
from ..preprocess import Preprocessor
from .metrics import (
    build_pairs,
    evaluate_detection,
    evaluate_identification,
    evaluate_transcription,
    evaluate_verification,
)

__all__ = [
    "benchmark_detection",
    "benchmark_embedders",
    "robustness_sweep",
    "benchmark_ocr",
    "run_full_benchmark",
]

#: Severity levels used throughout, from "studio photo" to "CCTV at 2am".
SEVERITY_LEVELS = (0.0, 0.3, 0.6, 0.9)


def _make_gallery_and_probes(
    num_identities: int,
    enrol_per_identity: int,
    probes_per_identity: int,
    severity: float,
    seed: int,
) -> tuple[list[np.ndarray], list[str], list[np.ndarray], list[str]]:
    """Build disjoint enrolment and probe sets for the same identities."""
    rng = random.Random(seed)
    enrol_images: list[np.ndarray] = []
    enrol_labels: list[str] = []
    probe_images: list[np.ndarray] = []
    probe_labels: list[str] = []

    seen: set[str] = set()
    for _ in range(num_identities):
        text = random_plate_text(rng)
        while text in seen:
            text = random_plate_text(rng)
        seen.add(text)

        clean = render_plate(text)
        for _ in range(enrol_per_identity):
            enrol_images.append(augment_plate(clean, rng, severity * 0.5))
            enrol_labels.append(text)
        for _ in range(probes_per_identity):
            probe_images.append(augment_plate(clean, rng, severity))
            probe_labels.append(text)

    return enrol_images, enrol_labels, probe_images, probe_labels


def benchmark_detection(
    num_scenes: int = 120,
    severity: float = 0.5,
    seed: int = 20,
) -> dict:
    """How often does the OpenCV ensemble actually find the plate?"""
    rng = random.Random(seed)
    detector = build_detector("plate")

    best_ious: list[float] = []
    per_source: dict[str, int] = {}
    started = time.perf_counter()

    for _ in range(num_scenes):
        plate = render_plate(random_plate_text(rng))
        scene, truth_box = compose_scene(augment_plate(plate, rng, severity), rng)
        truth = Detection(*truth_box)

        detections = detector.detect(scene)
        if not detections:
            best_ious.append(0.0)
            continue

        scored = [(d, d.intersection_over_union(truth)) for d in detections]
        best_detection, best_iou = max(scored, key=lambda pair: pair[1])
        best_ious.append(best_iou)
        per_source[best_detection.source] = per_source.get(best_detection.source, 0) + 1

    elapsed = time.perf_counter() - started
    metrics = evaluate_detection(best_ious, iou_threshold=0.5)
    return {
        **metrics.as_dict(),
        "severity": severity,
        "winning_strategy_counts": per_source,
        "milliseconds_per_image": round(1000.0 * elapsed / max(num_scenes, 1), 2),
    }


def benchmark_embedders(
    kinds: tuple[str, ...] = ("cnn", "hog", "raw"),
    num_identities: int = 60,
    enrol_per_identity: int = 4,
    probes_per_identity: int = 4,
    severity: float = 0.6,
    checkpoint: Path | None = None,
    seed: int = 21,
) -> list[dict]:
    """Head-to-head comparison of the CNN against the classical baselines."""
    geometry = PlateGeometry()
    preprocessor = Preprocessor(geometry.size, PreprocessConfig())
    checkpoint = Path(checkpoint or ARTIFACT_DIR / "embedder_plate.pt")

    enrol_images, enrol_labels, probe_images, probe_labels = _make_gallery_and_probes(
        num_identities, enrol_per_identity, probes_per_identity, severity, seed
    )
    enrol_batch = preprocessor.batch(enrol_images)
    probe_batch = preprocessor.batch(probe_images)

    rows: list[dict] = []
    for kind in kinds:
        if kind == "cnn" and not checkpoint.exists():
            rows.append({"embedder": kind, "skipped": "no checkpoint; run `lprs train-embedder`"})
            continue

        embedder = build_embedder(
            kind, geometry.size, checkpoint=checkpoint if kind == "cnn" else None
        )

        started = time.perf_counter()
        probe_vectors = embedder.embed(probe_batch)
        elapsed = time.perf_counter() - started
        enrol_vectors = embedder.embed(enrol_batch)

        gallery = Gallery(MatchConfig(accept_threshold=0.0), strategy="centroid")
        gallery.enrol_many(enrol_labels, enrol_vectors)

        ranked = [
            [candidate.label for candidate in gallery.match(vector, top_k=5)]
            for vector in probe_vectors
        ]
        identification = evaluate_identification(probe_labels, ranked)

        similarities, is_same = build_pairs(probe_vectors, probe_labels, seed=seed)
        verification = evaluate_verification(similarities, is_same)

        rows.append(
            {
                "embedder": kind,
                "embedding_dim": embedder.embedding_dim,
                **identification.as_dict(),
                **verification.as_dict(),
                "milliseconds_per_probe": round(
                    1000.0 * elapsed / max(len(probe_batch), 1), 3
                ),
            }
        )
    return rows


def robustness_sweep(
    kinds: tuple[str, ...] = ("cnn", "hog"),
    severities: tuple[float, ...] = SEVERITY_LEVELS,
    num_identities: int = 40,
    checkpoint: Path | None = None,
    seed: int = 22,
) -> list[dict]:
    """Degrade the images progressively and watch accuracy fall.

    A single accuracy number hides the interesting behaviour. The *slope* of
    this curve is what tells you whether a model will survive a rainy night.
    """
    rows: list[dict] = []
    for severity in severities:
        for row in benchmark_embedders(
            kinds=kinds,
            num_identities=num_identities,
            enrol_per_identity=3,
            probes_per_identity=3,
            severity=severity,
            checkpoint=checkpoint,
            seed=seed + int(severity * 100),
        ):
            rows.append({"severity": severity, **row})
    return rows


def benchmark_ocr(
    num_plates: int = 200,
    severities: tuple[float, ...] = SEVERITY_LEVELS,
    checkpoint: Path | None = None,
    seed: int = 23,
) -> list[dict]:
    """Character accuracy and exact-match rate of the CRNN across severities."""
    checkpoint = Path(checkpoint or ARTIFACT_DIR / "ocr_crnn.pt")
    if not checkpoint.exists():
        return [{"skipped": "no checkpoint; run `lprs train-ocr`"}]

    geometry = PlateGeometry()
    preprocessor = Preprocessor(geometry.size, PreprocessConfig())
    transcriber = PlateTranscriber.from_checkpoint(checkpoint)

    rows: list[dict] = []
    for severity in severities:
        rng = random.Random(seed + int(severity * 100))
        images: list[np.ndarray] = []
        truths: list[str] = []
        for _ in range(num_plates):
            text = random_plate_text(rng)
            images.append(augment_plate(render_plate(text), rng, severity))
            truths.append(text)

        batch = preprocessor.batch(images)
        started = time.perf_counter()
        predictions = transcriber.transcribe(batch)
        elapsed = time.perf_counter() - started

        metrics = evaluate_transcription(truths, [text for text, _ in predictions])
        rows.append(
            {
                "severity": severity,
                **metrics.as_dict(),
                "mean_confidence": round(
                    float(np.mean([conf for _, conf in predictions])), 4
                ),
                "milliseconds_per_plate": round(
                    1000.0 * elapsed / max(num_plates, 1), 3
                ),
            }
        )
    return rows


def run_full_benchmark(
    quick: bool = False,
    checkpoint: Path | None = None,
    ocr_checkpoint: Path | None = None,
) -> dict:
    """Run every experiment and return one JSON-serialisable blob."""
    scale = 0.3 if quick else 1.0
    started = time.perf_counter()

    results = {
        "detection": benchmark_detection(num_scenes=int(120 * scale) or 20),
        "embedder_comparison": benchmark_embedders(
            num_identities=int(60 * scale) or 20, checkpoint=checkpoint
        ),
        "robustness": robustness_sweep(
            num_identities=int(40 * scale) or 15, checkpoint=checkpoint
        ),
        "ocr": benchmark_ocr(
            num_plates=int(200 * scale) or 60, checkpoint=ocr_checkpoint
        ),
    }
    results["meta"] = {
        "quick": quick,
        "total_seconds": round(time.perf_counter() - started, 1),
        "severity_levels": list(SEVERITY_LEVELS),
    }
    return results

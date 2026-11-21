"""Metrics, embedders and the end-to-end pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from lprs.config import MatchConfig, PlateGeometry
from lprs.embeddings import build_embedder
from lprs.evaluation import (
    character_accuracy,
    evaluate_detection,
    evaluate_identification,
    evaluate_transcription,
    evaluate_verification,
)
from lprs.pipeline import build_pipeline


# -- embedders -------------------------------------------------------------


@pytest.mark.parametrize("kind", ["hog", "raw"])
def test_embedders_produce_unit_norm_vectors(kind, preprocessor, small_dataset, geometry):
    images, _ = small_dataset
    embedder = build_embedder(kind, geometry.size)
    vectors = embedder.embed(preprocessor.batch(images[:6]))

    assert vectors.shape == (6, embedder.embedding_dim)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)


@pytest.mark.parametrize("kind", ["hog", "raw"])
def test_embedders_handle_empty_batches(kind, geometry):
    embedder = build_embedder(kind, geometry.size)
    assert embedder.embed(np.zeros((0, 48, 160), dtype=np.float32)).shape[0] == 0


def test_embedder_is_deterministic(preprocessor, small_dataset, geometry):
    images, _ = small_dataset
    batch = preprocessor.batch(images[:4])
    embedder = build_embedder("hog", geometry.size)
    assert np.allclose(embedder.embed(batch), embedder.embed(batch))


def test_build_embedder_rejects_unknown_kind(geometry):
    with pytest.raises(ValueError, match="Unknown embedder"):
        build_embedder("telepathy", geometry.size)


def test_cnn_embedder_requires_a_checkpoint(geometry):
    with pytest.raises(ValueError, match="requires a trained checkpoint"):
        build_embedder("cnn", geometry.size, checkpoint=None)


# -- metrics ---------------------------------------------------------------


def test_detection_metrics_count_hits_at_threshold():
    metrics = evaluate_detection([0.9, 0.6, 0.2, 0.0], iou_threshold=0.5)
    assert metrics.num_images == 4
    assert metrics.detection_rate == pytest.approx(0.5)
    assert metrics.mean_iou == pytest.approx(0.425)
    assert metrics.mean_iou_on_hits == pytest.approx(0.75)


def test_detection_metrics_on_empty_input():
    assert evaluate_detection([]).detection_rate == 0.0


def test_identification_top1_and_top5():
    metrics = evaluate_identification(
        ["A", "B", "C"],
        [["A", "B"], ["X", "B"], ["Y", "Z", "C"]],
    )
    assert metrics.top1_accuracy == pytest.approx(1 / 3)
    assert metrics.top5_accuracy == pytest.approx(1.0)


def test_identification_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        evaluate_identification(["A"], [["A"], ["B"]])


def test_verification_on_perfectly_separable_scores():
    scores = np.array([0.95, 0.92, 0.10, 0.05])
    same = np.array([1, 1, 0, 0])
    metrics = evaluate_verification(scores, same)
    assert metrics.roc_auc == pytest.approx(1.0)
    assert metrics.equal_error_rate == pytest.approx(0.0)
    assert metrics.accuracy_at_best == pytest.approx(1.0)


def test_verification_threshold_is_always_finite():
    """Regression: sklearn prepends an infinite sentinel threshold."""
    rng = np.random.default_rng(0)
    scores = rng.random(60)
    labels = rng.integers(0, 2, 60)
    metrics = evaluate_verification(scores, labels)
    assert np.isfinite(metrics.best_threshold)


def test_verification_degrades_gracefully_with_one_class():
    metrics = evaluate_verification(np.array([0.5, 0.6]), np.array([1, 1]))
    assert metrics.roc_auc == 0.5


def test_character_accuracy_is_floored_at_zero():
    assert character_accuracy("ABC", "ABC") == 1.0
    assert character_accuracy("ABC", "ABD") == pytest.approx(2 / 3)
    assert character_accuracy("ABC", "ZZZZZZZZZZZZ") == 0.0


def test_transcription_metrics():
    metrics = evaluate_transcription(["AB1", "CD2"], ["AB1", "CD3"])
    assert metrics.exact_match == pytest.approx(0.5)
    assert metrics.mean_edit_distance == pytest.approx(0.5)
    assert metrics.examples == [("CD2", "CD3")]


# -- pipeline --------------------------------------------------------------


@pytest.fixture(scope="module")
def hog_pipeline():
    return build_pipeline(
        domain="plate",
        embedder_kind="hog",
        match_config=MatchConfig(accept_threshold=0.5),
    )


def test_pipeline_rejects_unknown_domain():
    with pytest.raises(ValueError, match="Unknown domain"):
        build_pipeline(domain="bicycle", embedder_kind="hog")


def test_pipeline_recognises_an_enrolled_plate(hog_pipeline, small_dataset):
    images, labels = small_dataset
    assert hog_pipeline.enrol_batch(labels, images, detect_first=False) == len(images)

    results = hog_pipeline.recognise(images[0], max_regions=1)
    assert results, "pipeline detected nothing in a plate crop"
    assert results[0].label == labels[0]
    assert results[0].best.accepted


def test_recognition_result_serialises(hog_pipeline, small_dataset):
    images, _ = small_dataset
    payload = hog_pipeline.recognise(images[1], max_regions=1)[0].as_dict()
    assert {"box", "label", "candidates", "detection_score"} <= payload.keys()
    assert isinstance(payload["candidates"], list)


def test_pipeline_with_empty_gallery_returns_no_candidates(small_dataset):
    pipeline = build_pipeline(domain="plate", embedder_kind="hog")
    images, _ = small_dataset
    results = pipeline.recognise(images[0], max_regions=1)
    assert results and results[0].candidates == []
    assert results[0].label == "UNKNOWN"


def test_enrol_rejects_images_with_nothing_detected():
    pipeline = build_pipeline(domain="face", embedder_kind="hog")
    blank = np.zeros((20, 20, 3), dtype=np.uint8)
    assert pipeline.enrol("nobody", blank, detect_first=True) is False
    assert len(pipeline.gallery) == 0


def test_embed_image_matches_embedder_output(hog_pipeline, small_dataset, geometry):
    images, _ = small_dataset
    vector = hog_pipeline.embed_image(images[0])
    assert vector.shape == (hog_pipeline.embedder.embedding_dim,)
    assert np.linalg.norm(vector) == pytest.approx(1.0, abs=1e-5)

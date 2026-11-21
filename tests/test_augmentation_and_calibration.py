"""Crop-jitter augmentation and threshold calibration.

Both of these exist because of bugs found by end-to-end testing, so they get
regression coverage.
"""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from lprs.config import EmbedderConfig, MatchConfig
from lprs.data import generate_identity_dataset, jitter_crop, render_plate
from lprs.embeddings.cnn import EmbeddingNet, TorchEmbedder
from lprs.pipeline import build_pipeline


# -- jitter ----------------------------------------------------------------


def test_jitter_crop_changes_framing(clean_plate):
    jittered = jitter_crop(clean_plate, random.Random(1))
    assert jittered.ndim == 3
    assert jittered.shape != clean_plate.shape, "jitter produced an identical crop"


def test_jitter_crop_is_reproducible(clean_plate):
    first = jitter_crop(clean_plate, random.Random(42))
    second = jitter_crop(clean_plate, random.Random(42))
    assert np.array_equal(first, second)


def test_jitter_crop_produces_varied_framings(clean_plate):
    rng = random.Random(3)
    shapes = {jitter_crop(clean_plate, rng).shape for _ in range(12)}
    assert len(shapes) > 1, "jitter is not actually varying the crop"


def test_jitter_crop_never_returns_an_empty_image():
    tiny = render_plate("AB12CD")[:20, :40]
    result = jitter_crop(tiny, random.Random(0), max_shift=0.9, max_scale=0.9)
    assert result.size > 0


def test_jitter_respects_the_requested_magnitude(clean_plate):
    """Gentle settings must stay closer to the original than aggressive ones."""
    height, width = clean_plate.shape[:2]

    def mean_deviation(shift, scale):
        rng = random.Random(9)
        crops = [jitter_crop(clean_plate, rng, shift, scale) for _ in range(20)]
        return np.mean([abs(crop.shape[1] - width) + abs(crop.shape[0] - height) for crop in crops])

    assert mean_deviation(0.02, 0.04) < mean_deviation(0.15, 0.30)


def test_dataset_jitter_can_be_disabled():
    plain, _ = generate_identity_dataset(3, 2, seed=1, crop_jitter=False)
    assert len({image.shape for image in plain}) == 1

    jittered, _ = generate_identity_dataset(3, 2, seed=1, crop_jitter=True)
    assert len({image.shape for image in jittered}) > 1


# -- threshold calibration -------------------------------------------------


def test_checkpoint_round_trips_the_calibrated_threshold(tmp_path):
    model = EmbeddingNet(in_channels=1, config=EmbedderConfig(embedding_dim=16))
    path = tmp_path / "embedder.pt"
    model.save(path, accept_threshold=0.8125)

    restored = EmbeddingNet.load(path)
    assert restored.recommended_threshold == pytest.approx(0.8125)
    assert TorchEmbedder(restored).recommended_threshold == pytest.approx(0.8125)


def test_checkpoint_without_a_threshold_is_none(tmp_path):
    path = tmp_path / "embedder.pt"
    EmbeddingNet(config=EmbedderConfig(embedding_dim=16)).save(path)
    assert EmbeddingNet.load(path).recommended_threshold is None


def test_pipeline_adopts_the_calibrated_threshold(tmp_path):
    path = tmp_path / "embedder.pt"
    EmbeddingNet(in_channels=1, config=EmbedderConfig(embedding_dim=16)).save(
        path, accept_threshold=0.77
    )
    pipeline = build_pipeline(
        domain="plate", embedder_kind="cnn", embedder_checkpoint=path
    )
    assert pipeline.gallery.config.accept_threshold == pytest.approx(0.77)


def test_explicit_match_config_overrides_the_checkpoint(tmp_path):
    """An operator asking for a specific threshold must always win."""
    path = tmp_path / "embedder.pt"
    EmbeddingNet(in_channels=1, config=EmbedderConfig(embedding_dim=16)).save(
        path, accept_threshold=0.77
    )
    pipeline = build_pipeline(
        domain="plate",
        embedder_kind="cnn",
        embedder_checkpoint=path,
        match_config=MatchConfig(accept_threshold=0.5),
    )
    assert pipeline.gallery.config.accept_threshold == pytest.approx(0.5)


def test_embedding_net_output_is_unit_norm():
    model = EmbeddingNet(in_channels=1, config=EmbedderConfig(embedding_dim=32))
    model.eval()
    with torch.no_grad():
        vectors = model(torch.randn(5, 1, 48, 160))
    assert torch.allclose(vectors.norm(dim=1), torch.ones(5), atol=1e-5)


# -- preprocessing cache ---------------------------------------------------


def test_cached_and_uncached_datasets_agree(preprocessor, small_dataset):
    from lprs.data.datasets import ImageLabelDataset

    images, labels = small_dataset
    cached = ImageLabelDataset(images[:5], labels[:5], preprocessor, precompute=True)
    lazy = ImageLabelDataset(images[:5], labels[:5], preprocessor, precompute=False)

    for index in range(5):
        assert torch.allclose(cached[index][0], lazy[index][0])
        assert cached[index][1] == lazy[index][1]

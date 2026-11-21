"""Crop-jitter augmentation and threshold calibration.

Both of these exist because of bugs found by end-to-end testing, so they get
regression coverage.
"""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from lprs.config import CropJitter, EmbedderConfig, MatchConfig
from lprs.data import generate_identity_dataset, jitter_crop, render_plate
from lprs.embeddings.cnn import EmbeddingNet, TorchEmbedder
from lprs.pipeline import build_pipeline
from lprs.training.train_embedder import embedding_spread


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
    violent = CropJitter(
        shift_x=0.9, shift_y=0.9, scale_width=(-0.9, 0.9), scale_height=(-0.9, 0.9)
    )
    result = jitter_crop(tiny, random.Random(0), violent)
    assert result.size > 0


def test_jitter_respects_the_requested_magnitude(clean_plate):
    """Gentle settings must stay closer to the original than aggressive ones."""
    height, width = clean_plate.shape[:2]

    def mean_deviation(jitter: CropJitter) -> float:
        rng = random.Random(9)
        crops = [jitter_crop(clean_plate, rng, jitter) for _ in range(20)]
        return float(
            np.mean(
                [abs(crop.shape[1] - width) + abs(crop.shape[0] - height) for crop in crops]
            )
        )

    tiny = CropJitter(
        shift_x=0.02, shift_y=0.02, scale_width=(-0.03, 0.03), scale_height=(-0.03, 0.03)
    )
    assert mean_deviation(tiny) < mean_deviation(CropJitter())


def test_gentle_jitter_is_milder_than_the_default():
    """CTC training must not have characters cropped off."""
    gentle, default = CropJitter.gentle(), CropJitter()
    assert gentle.shift_x < default.shift_x
    assert gentle.shift_y < default.shift_y
    # The aggressive shrink end is what clips glyphs.
    assert gentle.scale_height[0] > default.scale_height[0]


def test_default_jitter_is_anisotropic():
    """Measured detector error is far worse vertically; see CropJitter docs."""
    jitter = CropJitter()
    assert jitter.shift_y > jitter.shift_x
    assert abs(jitter.scale_height[0]) > abs(jitter.scale_width[0])


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


def test_a_saved_gallery_does_not_override_calibration(tmp_path):
    """Regression: loading a gallery used to silently restore a stale default."""
    from lprs.matching import Gallery

    gallery_path = tmp_path / "gallery.npz"
    stale = Gallery(MatchConfig(accept_threshold=0.62))
    stale.enrol("A", np.ones(16, dtype=np.float32))
    stale.save(gallery_path)

    checkpoint = tmp_path / "embedder.pt"
    EmbeddingNet(in_channels=1, config=EmbedderConfig(embedding_dim=16)).save(
        checkpoint, accept_threshold=0.91
    )

    pipeline = build_pipeline(
        domain="plate",
        embedder_kind="cnn",
        embedder_checkpoint=checkpoint,
        gallery_path=gallery_path,
    )
    assert pipeline.gallery.config.accept_threshold == pytest.approx(0.91)


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


# -- collapse detection ----------------------------------------------------


def test_spread_is_zero_for_a_collapsed_space():
    collapsed = np.tile(np.array([1.0, 0.0, 0.0], dtype=np.float32), (10, 1))
    assert embedding_spread(collapsed) == pytest.approx(0.0, abs=1e-6)


def test_spread_is_large_for_a_well_separated_space():
    spread_out = np.eye(10, 16, dtype=np.float32)
    assert embedding_spread(spread_out) > 0.7


def test_spread_orders_two_spaces_correctly():
    rng = np.random.default_rng(0)
    base = rng.normal(size=(40, 8)).astype(np.float32)
    tight = base * 0.01
    assert embedding_spread(tight) < embedding_spread(base)


def test_spread_of_nothing_is_zero():
    assert embedding_spread(np.zeros((0, 8), dtype=np.float32)) == 0.0


def test_soft_margin_is_the_default():
    """Hard-margin batch-hard collapsed the space; see EmbedderConfig docs."""
    assert EmbedderConfig().soft_margin is True


def test_soft_margin_keeps_rewarding_extra_separation():
    """A hard hinge goes flat past the margin; softplus never does."""
    from lprs.training.triplet import BatchHardTripletLoss

    labels = torch.tensor([0, 0, 1, 1])
    near = torch.tensor(
        [[1.0, 0.0], [0.94, 0.34], [-1.0, 0.0], [-0.94, 0.34]], dtype=torch.float32
    )
    far = torch.tensor(
        [[1.0, 0.0], [0.999, 0.044], [-1.0, 0.0], [-0.999, 0.044]], dtype=torch.float32
    )

    soft = BatchHardTripletLoss(soft=True)
    assert float(soft(far, labels)[0]) < float(soft(near, labels)[0])

    hard = BatchHardTripletLoss(margin=0.3, soft=False)
    assert float(hard(far, labels)[0]) == pytest.approx(0.0, abs=1e-6)
    assert float(hard(near, labels)[0]) == pytest.approx(0.0, abs=1e-6)


# -- preprocessing cache ---------------------------------------------------


def test_cached_and_uncached_datasets_agree(preprocessor, small_dataset):
    from lprs.data.datasets import ImageLabelDataset

    images, labels = small_dataset
    cached = ImageLabelDataset(images[:5], labels[:5], preprocessor, precompute=True)
    lazy = ImageLabelDataset(images[:5], labels[:5], preprocessor, precompute=False)

    for index in range(5):
        assert torch.allclose(cached[index][0], lazy[index][0])
        assert cached[index][1] == lazy[index][1]

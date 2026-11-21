"""Preprocessing behaviour."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from lprs.config import PreprocessConfig
from lprs.preprocess import (
    Preprocessor,
    apply_clahe,
    deskew,
    estimate_skew_angle,
    resize_keep_aspect,
    to_grayscale,
)


def test_to_grayscale_collapses_channels(clean_plate):
    gray = to_grayscale(clean_plate)
    assert gray.ndim == 2
    # Already-grey input must pass through untouched.
    assert to_grayscale(gray) is gray


def test_resize_keep_aspect_letterboxes_without_distortion():
    tall = np.full((200, 50, 3), 10, dtype=np.uint8)
    result = resize_keep_aspect(tall, (160, 48))
    assert result.shape == (48, 160, 3)
    # The content must be padded horizontally, not stretched.
    assert (result[:, 0] == 127).all()


def test_resize_rejects_empty_image():
    with pytest.raises(ValueError):
        resize_keep_aspect(np.zeros((0, 0), dtype=np.uint8), (16, 16))


def test_clahe_increases_local_contrast():
    flat = np.random.default_rng(0).integers(100, 140, (64, 64), dtype=np.uint8)
    enhanced = apply_clahe(flat, clip_limit=3.0, tile_grid=(8, 8))
    assert enhanced.std() > flat.std()


def test_clahe_preserves_colour_shape(clean_plate):
    enhanced = apply_clahe(clean_plate, 2.0, (8, 8))
    assert enhanced.shape == clean_plate.shape


def test_skew_angle_is_normalised_into_plus_minus_45():
    """Regression: OpenCV returns -90 for axis-aligned boxes in some builds."""
    canvas = np.full((100, 300), 255, dtype=np.uint8)
    cv2.rectangle(canvas, (40, 40), (260, 60), 0, -1)
    angle = estimate_skew_angle(canvas, max_degrees=20.0)
    assert -45.0 <= angle <= 45.0
    assert abs(angle) < 1.0  # it is not actually skewed


def test_deskew_corrects_a_known_rotation(clean_plate):
    height, width = clean_plate.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), 7.0, 1.0)
    rotated = cv2.warpAffine(clean_plate, matrix, (width, height), borderValue=(235, 235, 235))

    before = abs(estimate_skew_angle(to_grayscale(rotated), 20.0))
    after = abs(estimate_skew_angle(to_grayscale(deskew(rotated, 20.0)), 20.0))
    assert after <= before


def test_preprocessor_output_contract(preprocessor, clean_plate, geometry):
    result = preprocessor.run(clean_plate)
    assert result.shape == (geometry.height, geometry.width)
    assert result.dtype == np.float32
    assert 0.0 <= result.min() and result.max() <= 1.0


def test_preprocessor_face_config_keeps_colour(clean_plate):
    face_pre = Preprocessor((112, 112), PreprocessConfig.for_faces())
    result = face_pre.run(clean_plate)
    assert result.shape == (112, 112, 3)


def test_preprocessor_rejects_empty(preprocessor):
    with pytest.raises(ValueError):
        preprocessor.run(np.zeros((0, 0, 3), dtype=np.uint8))


def test_batch_matches_individual_runs(preprocessor, small_dataset):
    images, _ = small_dataset
    batch = preprocessor.batch(images[:3])
    assert batch.shape[0] == 3
    assert np.allclose(batch[1], preprocessor.run(images[1]))


def test_batch_of_nothing_is_empty(preprocessor):
    assert preprocessor.batch([]).shape[0] == 0

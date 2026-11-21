"""CTC codec, CRNN shapes, and the triplet loss."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from lprs.config import ALPHABET, NUM_CLASSES, PlateGeometry
from lprs.ocr import CRNN, PlateTranscriber
from lprs.ocr.codec import (
    BLANK_INDEX,
    collapse_repeats,
    decode_indices,
    encode_text,
    greedy_decode,
)
from lprs.training.triplet import BatchHardTripletLoss, pairwise_distances


# -- codec -----------------------------------------------------------------


def test_blank_is_index_zero():
    assert BLANK_INDEX == 0
    assert len(ALPHABET) == NUM_CLASSES == 37  # blank + 10 digits + 26 letters


def test_encode_decode_round_trip():
    text = "MH12AB1234"
    assert decode_indices(encode_text(text)) == text


def test_encode_ignores_unknown_glyphs():
    assert decode_indices(encode_text("AB-12 ?")) == "AB12"


def test_collapse_repeats_merges_runs():
    assert collapse_repeats([1, 1, 1, 2, 2, 3]) == [1, 2, 3]


def test_blank_separates_genuine_double_letters():
    """The whole reason CTC needs a blank symbol."""
    letter = encode_text("A")[0]
    assert collapse_repeats([letter, letter]) == [letter]
    assert collapse_repeats([letter, BLANK_INDEX, letter]) == [letter, letter]


def test_greedy_decode_recovers_a_planted_sequence():
    target = encode_text("AB1")
    log_probs = np.full((9, NUM_CLASSES), -20.0, dtype=np.float32)
    frames = [BLANK_INDEX, target[0], target[0], BLANK_INDEX,
              target[1], BLANK_INDEX, target[2], target[2], BLANK_INDEX]
    for step, index in enumerate(frames):
        log_probs[step, index] = 0.0

    text, confidence = greedy_decode(log_probs)
    assert text == "AB1"
    assert 0.0 <= confidence <= 1.0


def test_greedy_decode_rejects_wrong_rank():
    with pytest.raises(ValueError, match=r"\(T, C\) matrix"):
        greedy_decode(np.zeros((2, 3, 4)))


# -- CRNN ------------------------------------------------------------------


def test_crnn_emits_one_frame_per_four_input_columns():
    geometry = PlateGeometry()
    model = CRNN(in_channels=1, geometry=geometry)
    output = model(torch.randn(3, 1, geometry.height, geometry.width))
    assert output.shape == (model.time_steps, 3, NUM_CLASSES)
    assert model.time_steps == geometry.width // 4


def test_crnn_has_enough_timesteps_for_ctc():
    """CTC requires T >= 2*L + 1 to represent a label of length L."""
    model = CRNN()
    longest = model.config.max_label_length
    assert model.time_steps >= 2 * longest + 1


def test_crnn_output_is_log_probabilities():
    output = CRNN()(torch.randn(2, 1, 48, 160))
    assert torch.allclose(output.exp().sum(dim=2), torch.ones(output.shape[:2]), atol=1e-4)


def test_crnn_checkpoint_round_trip(tmp_path):
    model = CRNN()
    model.eval()
    path = tmp_path / "crnn.pt"
    model.save(path)

    restored = CRNN.load(path)
    sample = torch.randn(1, 1, 48, 160)
    with torch.no_grad():
        assert torch.allclose(model(sample), restored(sample), atol=1e-6)


def test_transcriber_returns_one_result_per_crop():
    transcriber = PlateTranscriber(CRNN())
    results = transcriber.transcribe(np.random.rand(3, 48, 160).astype(np.float32))
    assert len(results) == 3
    assert all(isinstance(text, str) for text, _ in results)


def test_transcriber_handles_empty_batch():
    assert PlateTranscriber(CRNN()).transcribe(np.zeros((0, 48, 160), dtype=np.float32)) == []


# -- triplet loss ----------------------------------------------------------


def test_pairwise_distances_has_zero_diagonal():
    distances = pairwise_distances(torch.randn(6, 8))
    assert torch.allclose(torch.diagonal(distances), torch.zeros(6), atol=1e-5)


def test_pairwise_distances_are_symmetric():
    distances = pairwise_distances(torch.randn(5, 4))
    assert torch.allclose(distances, distances.t(), atol=1e-5)


def test_triplet_loss_is_zero_when_margin_already_satisfied():
    # Two tight, far-apart clusters.
    embeddings = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [-1.0, 0.0]], dtype=torch.float32
    )
    labels = torch.tensor([0, 0, 1, 1])
    loss, stats = BatchHardTripletLoss(margin=0.3)(embeddings, labels)
    assert float(loss) == pytest.approx(0.0, abs=1e-5)
    assert stats["active"] == 0.0


def test_triplet_loss_is_positive_when_classes_overlap():
    embeddings = torch.tensor(
        [[1.0, 0.0], [-1.0, 0.0], [0.99, 0.01], [-0.99, 0.01]], dtype=torch.float32
    )
    labels = torch.tensor([0, 0, 1, 1])  # deliberately scrambled identities
    loss, _ = BatchHardTripletLoss(margin=0.5)(embeddings, labels)
    assert float(loss) > 0.0


def test_triplet_loss_survives_a_batch_with_no_positives():
    loss, stats = BatchHardTripletLoss()(torch.randn(3, 4), torch.tensor([0, 1, 2]))
    assert float(loss) == 0.0
    assert stats["active"] == 0.0

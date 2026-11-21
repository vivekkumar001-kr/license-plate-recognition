"""Gallery similarity matching and OCR-aware text distance."""

from __future__ import annotations

import numpy as np
import pytest

from lprs.config import MatchConfig
from lprs.matching import UNKNOWN_LABEL, Gallery, levenshtein, text_similarity
from lprs.matching.text import confusion_aware_distance


# -- text ------------------------------------------------------------------


def test_levenshtein_basics():
    assert levenshtein("ABC", "ABC") == 0
    assert levenshtein("ABC", "ABD") == 1
    assert levenshtein("", "ABC") == 3
    assert levenshtein("ABC", "") == 3


def test_confusion_pairs_cost_less_than_a_real_error():
    confusable = confusion_aware_distance("MH120B1234", "MH12OB1234")  # 0 vs O
    unrelated = confusion_aware_distance("MH120B1234", "MH12XB1234")
    assert confusable < unrelated
    assert 0 < confusable < 1


def test_text_similarity_is_bounded_and_ordered():
    assert text_similarity("ABC123", "ABC123") == 1.0
    assert text_similarity("", "") == 1.0
    near = text_similarity("MH12AB1234", "MH12A81234")
    far = text_similarity("MH12AB1234", "XY99ZZ0000")
    assert 0.0 <= far < near <= 1.0


# -- gallery ---------------------------------------------------------------


def _orthogonal_vectors(count: int, dim: int = 16) -> np.ndarray:
    vectors = np.zeros((count, dim), dtype=np.float32)
    for index in range(count):
        vectors[index, index % dim] = 1.0
    return vectors


def test_enrol_normalises_embeddings():
    gallery = Gallery()
    gallery.enrol("A", np.array([3.0, 4.0]))
    matrix, _ = gallery._scoring_matrix()
    assert np.linalg.norm(matrix[0]) == pytest.approx(1.0)


def test_identify_returns_the_enrolled_label():
    gallery = Gallery(MatchConfig(accept_threshold=0.5))
    vectors = _orthogonal_vectors(4)
    gallery.enrol_many(["A", "B", "C", "D"], vectors)
    assert gallery.identify(vectors[2]).label == "C"


def test_identify_refuses_an_unknown_probe():
    """Open-set behaviour: a stranger must not be forced onto a neighbour."""
    gallery = Gallery(MatchConfig(accept_threshold=0.9))
    gallery.enrol_many(["A", "B"], _orthogonal_vectors(2))

    stranger = np.zeros(16, dtype=np.float32)
    stranger[9] = 1.0
    verdict = gallery.identify(stranger)
    assert verdict.label == UNKNOWN_LABEL
    assert not verdict.accepted


def test_match_returns_unique_labels_ranked():
    gallery = Gallery(strategy="nearest")
    vectors = _orthogonal_vectors(3)
    gallery.enrol_many(["A", "A", "B"], vectors[[0, 0, 1]])

    results = gallery.match(vectors[0], top_k=5)
    labels = [result.label for result in results]
    assert labels == list(dict.fromkeys(labels)), "duplicate labels in ranking"
    assert results[0].rank == 1


def test_centroid_strategy_averages_samples():
    gallery = Gallery(strategy="centroid")
    gallery.enrol_many(["A", "A"], np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    matrix, labels = gallery._scoring_matrix()
    assert labels == ["A"]
    assert matrix.shape[0] == 1
    assert matrix[0] == pytest.approx([0.7071, 0.7071], abs=1e-3)


def test_dimension_mismatch_is_rejected():
    gallery = Gallery()
    gallery.enrol("A", np.ones(8, dtype=np.float32))
    with pytest.raises(ValueError, match="does not match gallery"):
        gallery.enrol("B", np.ones(16, dtype=np.float32))


def test_label_count_mismatch_is_rejected():
    with pytest.raises(ValueError, match="labels for"):
        Gallery().enrol_many(["A", "B"], np.ones((3, 4), dtype=np.float32))


def test_scoring_an_empty_gallery_raises():
    with pytest.raises(ValueError, match="empty gallery"):
        Gallery().match(np.ones(4, dtype=np.float32))


def test_samples_per_label_counts_correctly():
    gallery = Gallery()
    gallery.enrol_many(["A", "A", "B"], _orthogonal_vectors(3))
    assert gallery.samples_per_label() == {"A": 2, "B": 1}


def test_save_and_load_round_trip(tmp_path):
    gallery = Gallery(MatchConfig(accept_threshold=0.44), strategy="nearest")
    vectors = _orthogonal_vectors(5)
    gallery.enrol_many(list("ABCDE"), vectors, [{"note": str(i)} for i in range(5)])

    path = tmp_path / "gallery.npz"
    gallery.save(path)
    restored = Gallery.load(path)

    assert len(restored) == len(gallery)
    assert restored.labels == gallery.labels
    assert restored.strategy == "nearest"
    assert restored.config.accept_threshold == pytest.approx(0.44)
    assert restored.identify(vectors[3]).label == gallery.identify(vectors[3]).label

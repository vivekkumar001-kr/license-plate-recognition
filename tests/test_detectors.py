"""Detector geometry, NMS and the plate/face ensembles."""

from __future__ import annotations

import numpy as np
import pytest

from lprs.detectors import Detection, build_detector, non_max_suppression
from lprs.detectors.base import clamp_box
from lprs.detectors.plate import ContourPlateDetector, HaarPlateDetector


def test_detection_geometry():
    detection = Detection(10, 20, 100, 50, score=0.9)
    assert detection.area == 5000
    assert detection.aspect_ratio == 2.0
    assert detection.xyxy == (10, 20, 110, 70)


def test_iou_identical_boxes_is_one():
    box = Detection(0, 0, 10, 10)
    assert box.intersection_over_union(box) == pytest.approx(1.0)


def test_iou_disjoint_boxes_is_zero():
    assert Detection(0, 0, 10, 10).intersection_over_union(Detection(50, 50, 10, 10)) == 0.0


def test_iou_half_overlap():
    left = Detection(0, 0, 10, 10)
    right = Detection(5, 0, 10, 10)
    # intersection 50, union 150
    assert left.intersection_over_union(right) == pytest.approx(1 / 3)


def test_clamp_box_stays_inside_image():
    x, y, width, height = clamp_box((-20, -20, 500, 500), (100, 200, 3))
    assert (x, y) == (0, 0)
    assert x + width <= 200 and y + height <= 100


def test_crop_with_margin_grows_but_stays_in_bounds():
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    detection = Detection(0, 0, 50, 25)
    crop = detection.crop(image, margin=0.5)
    assert crop.shape[0] >= 25 and crop.shape[1] >= 50
    assert crop.shape[0] <= 100 and crop.shape[1] <= 200


def test_nms_keeps_highest_scoring_of_an_overlapping_cluster():
    kept = non_max_suppression(
        [
            Detection(0, 0, 100, 50, score=0.5),
            Detection(5, 5, 100, 50, score=0.9),
            Detection(500, 500, 100, 50, score=0.3),
        ],
        iou_threshold=0.3,
    )
    assert len(kept) == 2
    assert kept[0].score == 0.9


def test_nms_on_empty_input():
    assert non_max_suppression([]) == []


def test_build_detector_rejects_unknown_domain():
    with pytest.raises(ValueError, match="Unknown detection domain"):
        build_detector("submarine")


def test_cascades_are_actually_available():
    """OpenCV 5 removed the bundled cascades; this project requires 4.x."""
    assert HaarPlateDetector().available, (
        "haarcascade_russian_plate_number.xml is missing -- install opencv-python<5"
    )


def test_plate_detector_finds_a_composed_scene(scene_with_box):
    scene, truth_box = scene_with_box
    detections = build_detector("plate").detect(scene)
    assert detections, "detector found nothing in a clean synthetic scene"

    truth = Detection(*truth_box)
    best_iou = max(d.intersection_over_union(truth) for d in detections)
    assert best_iou > 0.4, f"best IoU was only {best_iou:.3f}"


def test_detector_handles_empty_image():
    assert build_detector("plate").detect(np.zeros((0, 0, 3), dtype=np.uint8)) == []


def test_contour_detector_filters_implausible_shapes():
    """A tall thin blob is never a plate."""
    image = np.full((400, 400, 3), 200, dtype=np.uint8)
    image[50:350, 180:200] = 0
    for detection in ContourPlateDetector().detect(image):
        assert detection.aspect_ratio >= 1.8


def test_whole_image_fallback_for_pre_cropped_plates(clean_plate):
    """A tight crop should still be recognisable, not rejected."""
    detector = build_detector("plate")
    detections = detector.detect(clean_plate)
    assert detections
    assert detections[0].source in {"whole-image", "haar-plate", "contour-plate"}


def test_detections_are_sorted_by_score(scene_with_box):
    scores = [d.score for d in build_detector("plate").detect(scene_with_box[0])]
    assert scores == sorted(scores, reverse=True)


def test_face_detector_constructs_and_returns_list():
    detector = build_detector("face")
    result = detector.detect(np.full((200, 200, 3), 128, dtype=np.uint8))
    assert isinstance(result, list)

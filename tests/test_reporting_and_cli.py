"""Report generation and the CLI argument parser."""

from __future__ import annotations

import json

import pytest

from lprs.cli import build_parser, draw_results
from lprs.reporting import build_report, derive_insights

SAMPLE_BENCHMARK = {
    "detection": {
        "num_images": 10,
        "detection_rate": 0.9,
        "mean_iou": 0.7,
        "mean_iou_on_hits": 0.75,
        "milliseconds_per_image": 12.3,
        "winning_strategy_counts": {"haar-plate": 7, "contour-plate": 2},
    },
    "embedder_comparison": [
        {
            "embedder": "cnn", "embedding_dim": 128, "top1_accuracy": 0.90,
            "top5_accuracy": 1.0, "macro_f1": 0.9, "roc_auc": 0.95,
            "equal_error_rate": 0.05, "best_threshold": 0.8,
            "tar_at_far_1pct": 0.88, "milliseconds_per_probe": 0.5,
        },
        {
            "embedder": "hog", "embedding_dim": 3420, "top1_accuracy": 0.70,
            "top5_accuracy": 0.9, "macro_f1": 0.7, "roc_auc": 0.80,
            "equal_error_rate": 0.2, "best_threshold": 0.9,
            "tar_at_far_1pct": 0.5, "milliseconds_per_probe": 1.2,
        },
    ],
    "robustness": [],
    "ocr": [
        {"severity": 0.0, "num_samples": 50, "character_accuracy": 0.95,
         "exact_match": 0.8, "mean_edit_distance": 0.4, "mean_confidence": 0.9},
        {"severity": 0.9, "num_samples": 50, "character_accuracy": 0.60,
         "exact_match": 0.1, "mean_edit_distance": 3.1, "mean_confidence": 0.5},
    ],
}


# -- reporting -------------------------------------------------------------


def test_report_renders_valid_standalone_html(tmp_path):
    benchmark = tmp_path / "benchmark.json"
    benchmark.write_text(json.dumps(SAMPLE_BENCHMARK))
    output = build_report(benchmark, tmp_path / "missing.json", tmp_path / "out.html")

    html = output.read_text()
    assert html.startswith("<!DOCTYPE html>")
    assert html.rstrip().endswith("</html>")
    assert 'lang="en"' in html


def test_report_survives_missing_input_files(tmp_path):
    """A fresh clone has no artifacts; the report must still render."""
    output = build_report(
        tmp_path / "nope.json", tmp_path / "also-nope.json", tmp_path / "out.html"
    )
    assert "No benchmark data found" in output.read_text()


def test_report_embeds_chart_data(tmp_path):
    benchmark = tmp_path / "benchmark.json"
    benchmark.write_text(json.dumps(SAMPLE_BENCHMARK))
    html = build_report(benchmark, tmp_path / "x.json", tmp_path / "out.html").read_text()

    assert "comparisonChart" in html
    assert "ocrChart" in html
    # Chart.js ignores canvas height unless the wrapper is fixed-height.
    assert "h-80" in html or "h-72" in html


def test_insights_compare_the_cnn_against_the_baseline():
    insights = " ".join(derive_insights(SAMPLE_BENCHMARK, {}))
    assert "cnn" in insights
    assert "HOG baseline" in insights
    assert "+20.0" in insights  # 0.90 - 0.70, reported honestly


def test_insights_report_a_losing_cnn_honestly():
    flipped = json.loads(json.dumps(SAMPLE_BENCHMARK))
    flipped["embedder_comparison"][0]["top1_accuracy"] = 0.50
    insights = " ".join(derive_insights(flipped, {}))
    assert "loses" in insights


def test_insights_cover_detection_and_ocr():
    insights = " ".join(derive_insights(SAMPLE_BENCHMARK, {}))
    assert "90.0%" in insights          # detection rate
    assert "CRNN+BiLSTM+CTC" in insights


# -- CLI -------------------------------------------------------------------


def test_parser_exposes_every_documented_command():
    parser = build_parser()
    expected = {
        "synth", "train-embedder", "train-ocr", "enrol",
        "enrol-synthetic", "recognise", "benchmark", "report", "serve",
    }
    subparsers = next(
        action for action in parser._actions if hasattr(action, "choices") and action.choices
    )
    assert expected <= set(subparsers.choices)


def test_every_command_binds_a_handler():
    parser = build_parser()
    subparsers = next(
        action for action in parser._actions if hasattr(action, "choices") and action.choices
    )
    for name in subparsers.choices:
        assert callable(subparsers.choices[name].get_default("handler")), name


def test_threshold_defaults_to_none_so_calibration_wins():
    """Regression: an argparse default of 0.62 silently beat calibration."""
    args = build_parser().parse_args(["recognise", "img.png"])
    assert args.threshold is None


def test_parser_rejects_an_unknown_command():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["teleport"])


def test_draw_results_does_not_mutate_the_input(scene_with_box):
    from lprs.pipeline import build_pipeline

    scene = scene_with_box[0]
    original = scene.copy()
    pipeline = build_pipeline(domain="plate", embedder_kind="hog")
    annotated = draw_results(scene, pipeline.recognise(scene, max_regions=2))

    assert annotated.shape == scene.shape
    assert (scene == original).all(), "draw_results drew on the caller's image"

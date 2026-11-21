"""Command-line interface.

Thin by design: every subcommand parses arguments and immediately delegates
to the library. No business logic lives here, which is what keeps the web
app and the CLI from drifting apart.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from .config import ARTIFACT_DIR, DATA_DIR, MatchConfig, REPORT_DIR, SAMPLE_DIR
from .data import (
    augment_plate,
    compose_scene,
    generate_identity_dataset,
    load_image,
    load_image_folder,
    random_plate_text,
    render_plate,
)
from .pipeline import build_pipeline

__all__ = ["main", "build_parser"]


def _print_json(payload) -> None:
    print(json.dumps(payload, indent=2, default=str))


def _default_gallery_path(domain: str) -> str:
    """Avoid mixing plate and face embeddings in the same gallery file."""
    suffix = "face" if domain == "face" else "plate"
    return str(ARTIFACT_DIR / f"gallery_{suffix}.npz")


def _match_config(args: argparse.Namespace) -> MatchConfig | None:
    """Build a MatchConfig only when the user actually asked for a threshold.

    Returning None lets the pipeline fall back to the value calibrated at
    training time and stored in the checkpoint, which beats any constant we
    could bake into an argparse default.
    """
    threshold = getattr(args, "threshold", None)
    return None if threshold is None else MatchConfig(accept_threshold=threshold)


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------


def command_synth(args: argparse.Namespace) -> int:
    """Generate a synthetic dataset on disk."""
    import random

    output = Path(args.output)
    images, labels = generate_identity_dataset(
        num_identities=args.identities,
        samples_per_identity=args.samples,
        severity=args.severity,
        seed=args.seed,
        output_dir=output / "crops",
    )
    print(f"wrote {len(images)} crops across {len(set(labels))} identities -> {output/'crops'}")

    if args.scenes > 0:
        rng = random.Random(args.seed)
        scene_dir = output / "scenes"
        scene_dir.mkdir(parents=True, exist_ok=True)
        annotations = []
        for index in range(args.scenes):
            text = random_plate_text(rng)
            plate = augment_plate(render_plate(text), rng, args.severity)
            scene, box = compose_scene(plate, rng)
            name = f"{index:04d}_{text}.png"
            cv2.imwrite(str(scene_dir / name), scene)
            annotations.append({"file": name, "text": text, "box": list(box)})
        (scene_dir / "annotations.json").write_text(json.dumps(annotations, indent=2))
        print(f"wrote {args.scenes} scenes + annotations.json -> {scene_dir}")
    return 0


def command_train_embedder(args: argparse.Namespace) -> int:
    from .training import train_embedder

    result = train_embedder(
        num_identities=args.identities,
        samples_per_identity=args.samples,
        epochs=args.epochs,
        severity=args.severity,
        device=args.device,
    )
    _print_json(result.final_metrics)
    return 0


def command_train_face_embedder(args: argparse.Namespace) -> int:
    """Train a face embedder from a labelled image folder."""
    from .training import train_face_embedder

    result = train_face_embedder(
        source=args.source,
        epochs=args.epochs,
        device=args.device,
        detect_first=not args.no_detect,
        min_samples_per_label=args.min_samples,
    )
    _print_json(result.final_metrics)
    return 0


def command_train_ocr(args: argparse.Namespace) -> int:
    from .training import train_ocr

    result = train_ocr(
        num_plates=args.plates,
        views_per_plate=args.views,
        epochs=args.epochs,
        severity=args.severity,
        device=args.device,
    )
    _print_json(result.final_metrics)
    return 0


def command_enrol(args: argparse.Namespace) -> int:
    """Enrol an image folder (``root/<label>/*.jpg``) into a gallery."""
    gallery_path = args.gallery or _default_gallery_path(args.domain)
    pipeline = build_pipeline(
        domain=args.domain,
        embedder_kind=args.embedder,
        gallery_path=gallery_path if args.append else None,
        match_config=_match_config(args),
    )
    images, labels = load_image_folder(args.source)
    accepted = pipeline.enrol_batch(labels, images, detect_first=args.detect)
    pipeline.gallery.save(gallery_path)

    print(f"enrolled {accepted}/{len(images)} images")
    print(
        f"gallery now holds {len(pipeline.gallery)} samples "
        f"across {len(pipeline.gallery.labels)} identities -> {gallery_path}"
    )
    return 0 if accepted else 1


def command_enrol_synthetic(args: argparse.Namespace) -> int:
    """Build a demo gallery from freshly synthesised plates."""
    pipeline = build_pipeline(
        domain="plate",
        embedder_kind=args.embedder,
        match_config=_match_config(args),
    )
    images, labels = generate_identity_dataset(
        num_identities=args.identities,
        samples_per_identity=args.samples,
        severity=0.3,
        seed=args.seed,
    )
    accepted = pipeline.enrol_batch(labels, images, detect_first=False)
    pipeline.gallery.save(args.gallery)
    print(f"enrolled {accepted} synthetic samples across "
          f"{len(pipeline.gallery.labels)} identities -> {args.gallery}")
    return 0


def command_recognise(args: argparse.Namespace) -> int:
    gallery_path = args.gallery or _default_gallery_path(args.domain)
    pipeline = build_pipeline(
        domain=args.domain,
        embedder_kind=args.embedder,
        gallery_path=gallery_path,
        match_config=_match_config(args),
    )
    image = load_image(args.image)
    results = pipeline.recognise(image, max_regions=args.max_regions, top_k=args.top_k)

    if not results:
        print("no regions detected")
        return 1

    _print_json([result.as_dict() for result in results])

    if args.annotate:
        annotated = draw_results(image, results)
        Path(args.annotate).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(args.annotate, annotated)
        print(f"annotated image -> {args.annotate}")
    return 0


def command_benchmark(args: argparse.Namespace) -> int:
    from .evaluation.experiments import run_full_benchmark

    results = run_full_benchmark(quick=args.quick)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2))
    print(f"benchmark results -> {output}")
    _print_json(results["detection"])
    for row in results["embedder_comparison"]:
        print(row)
    return 0


def command_report(args: argparse.Namespace) -> int:
    from .reporting import build_report

    path = build_report(
        benchmark_path=Path(args.benchmark),
        history_path=Path(args.history),
        output_path=Path(args.output),
    )
    print(f"report -> {path}")
    if args.open:
        _open_in_browser(path)
    return 0


def command_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run("webapp.main:app", host=args.host, port=args.port, reload=args.reload)
    return 0


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def draw_results(image: np.ndarray, results) -> np.ndarray:
    """Draw boxes and labels onto a copy of the image."""
    annotated = image.copy()
    for result in results:
        x, y, width, height = result.detection.box
        accepted = result.best is not None and result.best.accepted
        colour = (60, 200, 60) if accepted else (60, 60, 220)
        cv2.rectangle(annotated, (x, y), (x + width, y + height), colour, 2)

        caption = result.label
        if result.text:
            caption = f"{caption} [{result.text}]"
        if result.best:
            caption = f"{caption} {result.best.similarity:.2f}"

        (text_width, text_height), _ = cv2.getTextSize(
            caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        top = max(0, y - text_height - 8)
        cv2.rectangle(
            annotated, (x, top), (x + text_width + 8, top + text_height + 8), colour, -1
        )
        cv2.putText(
            annotated,
            caption,
            (x + 4, top + text_height + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return annotated


def _open_in_browser(path: Path) -> None:
    import subprocess
    import webbrowser

    url = path.resolve().as_uri()
    if sys.platform == "darwin":
        subprocess.run(["open", str(path)], check=False)
    elif sys.platform.startswith("win"):
        webbrowser.open(url)
    else:
        subprocess.run(["xdg-open", str(path)], check=False)


# --------------------------------------------------------------------------
# Parser
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lprs",
        description="License Plate Recognition System: OpenCV detection, CNN "
        "embeddings, similarity matching, CRNN+BiLSTM+CTC transcription.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    synth = subparsers.add_parser("synth", help="generate a synthetic dataset")
    synth.add_argument("--identities", type=int, default=60)
    synth.add_argument("--samples", type=int, default=6)
    synth.add_argument("--scenes", type=int, default=30)
    synth.add_argument("--severity", type=float, default=0.6)
    synth.add_argument("--seed", type=int, default=1337)
    synth.add_argument("--output", default=str(DATA_DIR / "synthetic"))
    synth.set_defaults(handler=command_synth)

    train_emb = subparsers.add_parser("train-embedder", help="train the CNN embedder")
    train_emb.add_argument("--identities", type=int, default=180)
    train_emb.add_argument("--samples", type=int, default=10)
    train_emb.add_argument("--epochs", type=int, default=16)
    train_emb.add_argument("--severity", type=float, default=0.6)
    train_emb.add_argument("--device", default="cpu")
    train_emb.set_defaults(handler=command_train_embedder)

    train_face_emb = subparsers.add_parser(
        "train-face-embedder",
        help="train the face CNN embedder from a labelled image folder",
    )
    train_face_emb.add_argument("source", help="root/<person_name>/*.jpg")
    train_face_emb.add_argument("--epochs", type=int, default=16)
    train_face_emb.add_argument("--device", default="cpu")
    train_face_emb.add_argument(
        "--no-detect",
        action="store_true",
        help="treat images as already-cropped faces (skip OpenCV face detection)",
    )
    train_face_emb.add_argument(
        "--min-samples",
        type=int,
        default=2,
        help="skip identities with fewer than this many images",
    )
    train_face_emb.set_defaults(handler=command_train_face_embedder)

    train_ocr_parser = subparsers.add_parser("train-ocr", help="train the CRNN/CTC head")
    train_ocr_parser.add_argument("--plates", type=int, default=2400)
    train_ocr_parser.add_argument("--views", type=int, default=2)
    train_ocr_parser.add_argument("--epochs", type=int, default=16)
    train_ocr_parser.add_argument("--severity", type=float, default=0.55)
    train_ocr_parser.add_argument("--device", default="cpu")
    train_ocr_parser.set_defaults(handler=command_train_ocr)

    enrol = subparsers.add_parser("enrol", help="enrol an image folder into a gallery")
    enrol.add_argument("source", help="root/<label>/*.jpg")
    enrol.add_argument("--domain", choices=("plate", "face"), default="plate")
    enrol.add_argument("--embedder", choices=("cnn", "hog", "raw"), default="cnn")
    enrol.add_argument(
        "--gallery",
        default=None,
        help="gallery file (default: artifacts/gallery_plate.npz or gallery_face.npz)",
    )
    enrol.add_argument("--threshold", type=float, default=None,
                       help="override the threshold calibrated during training")
    enrol.add_argument("--detect", action="store_true", help="detect before enrolling")
    enrol.add_argument("--append", action="store_true", help="add to an existing gallery")
    enrol.set_defaults(handler=command_enrol)

    demo = subparsers.add_parser(
        "enrol-synthetic", help="build a demo gallery from synthetic plates"
    )
    demo.add_argument("--identities", type=int, default=40)
    demo.add_argument("--samples", type=int, default=5)
    demo.add_argument("--embedder", choices=("cnn", "hog", "raw"), default="cnn")
    demo.add_argument("--gallery", default=str(ARTIFACT_DIR / "gallery_plate.npz"))
    demo.add_argument("--threshold", type=float, default=None,
                      help="override the threshold calibrated during training")
    demo.add_argument("--seed", type=int, default=99)
    demo.set_defaults(handler=command_enrol_synthetic)

    recognise = subparsers.add_parser("recognise", help="recognise one image")
    recognise.add_argument("image")
    recognise.add_argument("--domain", choices=("plate", "face"), default="plate")
    recognise.add_argument("--embedder", choices=("cnn", "hog", "raw"), default="cnn")
    recognise.add_argument(
        "--gallery",
        default=None,
        help="gallery file (default: artifacts/gallery_plate.npz or gallery_face.npz)",
    )
    recognise.add_argument("--threshold", type=float, default=None,
                           help="override the threshold calibrated during training")
    recognise.add_argument("--top-k", type=int, default=5)
    recognise.add_argument("--max-regions", type=int, default=5)
    recognise.add_argument("--annotate", help="write an annotated copy here")
    recognise.set_defaults(handler=command_recognise)

    benchmark = subparsers.add_parser("benchmark", help="run every experiment")
    benchmark.add_argument("--quick", action="store_true")
    benchmark.add_argument("--output", default=str(ARTIFACT_DIR / "benchmark.json"))
    benchmark.set_defaults(handler=command_benchmark)

    report = subparsers.add_parser("report", help="render the HTML results report")
    report.add_argument("--benchmark", default=str(ARTIFACT_DIR / "benchmark.json"))
    report.add_argument("--history", default=str(ARTIFACT_DIR / "training_history.json"))
    report.add_argument("--output", default=str(REPORT_DIR / "index.html"))
    report.add_argument("--open", action="store_true")
    report.set_defaults(handler=command_report)

    serve = subparsers.add_parser("serve", help="launch the web demo")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8321)
    serve.add_argument("--reload", action="store_true")
    serve.set_defaults(handler=command_serve)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    return int(args.handler(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())

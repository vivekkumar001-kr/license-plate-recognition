"""FastAPI + HTMX demo for the License Plate Recognition System.

The web layer owns zero recognition logic -- it loads a
:class:`~lprs.pipeline.RecognitionPipeline` and renders whatever that
returns. Same engine as the CLI, so the demo cannot silently disagree with
the benchmark.
"""

from __future__ import annotations

import base64
import io
import random
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from lprs.config import ARTIFACT_DIR
from lprs.data import augment_plate, compose_scene, random_plate_text, render_plate
from lprs.pipeline import RecognitionPipeline, build_pipeline

BASE_DIR = Path(__file__).resolve().parent
GALLERY_PATH = ARTIFACT_DIR / "gallery_plate.npz"

app = FastAPI(title="License Plate Recognition System")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

#: One pipeline per domain, built lazily so startup stays fast.
_pipelines: dict[str, RecognitionPipeline] = {}


def get_pipeline(domain: str = "plate") -> RecognitionPipeline:
    if domain not in _pipelines:
        _pipelines[domain] = build_pipeline(
            domain=domain,
            embedder_kind="cnn",
            gallery_path=GALLERY_PATH if domain == "plate" else None,
            # No explicit MatchConfig: use the threshold calibrated at
            # training time and carried inside the checkpoint.
            match_config=None,
        )
    return _pipelines[domain]


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def decode_upload(payload: bytes) -> np.ndarray | None:
    """Decode uploaded bytes into a BGR image, or None if unreadable."""
    buffer = np.frombuffer(payload, dtype=np.uint8)
    return cv2.imdecode(buffer, cv2.IMREAD_COLOR)


def to_data_uri(image: np.ndarray) -> str:
    """Encode a BGR image as an inline PNG data URI."""
    success, encoded = cv2.imencode(".png", image)
    if not success:
        return ""
    return "data:image/png;base64," + base64.b64encode(encoded.tobytes()).decode()


def annotate(image: np.ndarray, results) -> np.ndarray:
    from lprs.cli import draw_results

    return draw_results(image, results)


def gallery_summary(pipeline: RecognitionPipeline) -> dict:
    gallery = pipeline.gallery
    return {
        "samples": len(gallery),
        "identities": len(gallery.labels),
        "strategy": gallery.strategy,
        "threshold": gallery.config.accept_threshold,
        "embedder": pipeline.embedder.name,
        "has_ocr": pipeline.transcriber is not None,
        "labels": gallery.labels[:40],
    }


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    pipeline = get_pipeline("plate")
    return templates.TemplateResponse(
        request,
        "index.html",
        {"summary": gallery_summary(pipeline)},
    )


@app.post("/recognise", response_class=HTMLResponse)
async def recognise(
    request: Request,
    file: UploadFile = File(...),
    domain: str = Form("plate"),
) -> HTMLResponse:
    image = decode_upload(await file.read())
    if image is None:
        return templates.TemplateResponse(
            request, "partials/error.html", {"message": "Could not decode that file as an image."}
        )

    pipeline = get_pipeline(domain)
    # Use max_regions=1 so only the single best/highest-confidence plate detection
    # is returned, filtering out false-positive detections on grilles/emblems.
    results = pipeline.recognise(image, max_regions=1)
    return templates.TemplateResponse(
        request,
        "partials/results.html",
        {
            "results": [result.as_dict() for result in results],
            "image": to_data_uri(annotate(image, results)),
            "domain": domain,
        },
    )


@app.post("/demo", response_class=HTMLResponse)
def demo(request: Request, severity: float = Form(0.6)) -> HTMLResponse:
    """Synthesise a random scene and run it through the pipeline.

    The detector is imperfect (by design), so we retry a few times to avoid a
    confusing "demo" that shows *nothing detected*.
    """
    pipeline = get_pipeline("plate")

    rng = random.Random(1337)
    scene = None
    text = None
    results = []

    for _ in range(6):
        text = random_plate_text(rng)
        plate = augment_plate(render_plate(text), rng, severity)
        scene, _ = compose_scene(plate, rng)
        results = pipeline.recognise(scene, max_regions=5)
        if results:
            break

    assert scene is not None and text is not None

    return templates.TemplateResponse(
        request,
        "partials/results.html",
        {
            "results": [result.as_dict() for result in results],
            "image": to_data_uri(annotate(scene, results)),
            "ground_truth": text,
            "domain": "plate",
        },
    )


@app.post("/enrol", response_class=HTMLResponse)
async def enrol(
    request: Request,
    label: str = Form(...),
    file: UploadFile = File(...),
    domain: str = Form("plate"),
) -> HTMLResponse:
    image = decode_upload(await file.read())
    if image is None:
        return templates.TemplateResponse(
            request, "partials/error.html", {"message": "Could not decode that file as an image."}
        )

    pipeline = get_pipeline(domain)
    accepted = pipeline.enrol(label.strip().upper(), image, detect_first=True)
    if accepted and domain == "plate":
        pipeline.gallery.save(GALLERY_PATH)

    return templates.TemplateResponse(
        request,
        "partials/gallery.html",
        {
            "summary": gallery_summary(pipeline),
            "message": (
                f"Enrolled '{label.strip().upper()}'."
                if accepted
                else "Nothing detected in that image, so it was not enrolled."
            ),
            "success": accepted,
        },
    )


@app.get("/gallery", response_class=HTMLResponse)
def gallery(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "partials/gallery.html",
        {"summary": gallery_summary(get_pipeline("plate")), "message": "", "success": True},
    )


@app.get("/health")
def health() -> dict:
    pipeline = get_pipeline("plate")
    return {"status": "ok", **gallery_summary(pipeline)}

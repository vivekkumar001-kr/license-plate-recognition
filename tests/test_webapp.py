"""Web application routes.

These tests exercise the HTTP surface only -- the recognition logic itself is
covered by the pipeline tests. The point here is that the routes wire up
correctly and degrade sensibly on bad input.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

pytest.importorskip("httpx", reason="httpx is required by fastapi.testclient")

from fastapi.testclient import TestClient  # noqa: E402

from lprs.data import render_plate  # noqa: E402
from webapp.main import app  # noqa: E402


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


def _png_bytes(image: np.ndarray) -> bytes:
    success, encoded = cv2.imencode(".png", image)
    assert success
    return encoded.tobytes()


def test_health_reports_pipeline_state(client):
    payload = client.get("/health").json()
    assert payload["status"] == "ok"
    assert {"samples", "identities", "embedder"} <= payload.keys()


def test_index_renders(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "License Plate Recognition System" in response.text


def test_index_is_accessible_markup(client):
    """Spot-check the WCAG basics we committed to."""
    html = client.get("/").text
    assert 'lang="en"' in html
    assert "Skip to main content" in html
    assert 'aria-live="polite"' in html
    assert 'for="recognise-file"' in html  # every input has a real label


def test_gallery_fragment_renders(client):
    response = client.get("/gallery")
    assert response.status_code == 200
    assert "Identities" in response.text


def test_recognise_accepts_an_image(client):
    files = {"file": ("plate.png", _png_bytes(render_plate("KA05CD9876")), "image/png")}
    response = client.post("/recognise", files=files, data={"domain": "plate"})
    assert response.status_code == 200
    # Either a result card or an explicit "nothing detected" message.
    assert "UNKNOWN" in response.text or "detected" in response.text


def test_recognise_rejects_a_non_image(client):
    files = {"file": ("notes.txt", b"definitely not a picture", "text/plain")}
    response = client.post("/recognise", files=files, data={"domain": "plate"})
    assert response.status_code == 200
    assert "Could not decode" in response.text


def test_demo_generates_and_recognises(client):
    response = client.post("/demo", data={"severity": 0.3})
    assert response.status_code == 200
    assert "Synthetic ground truth" in response.text
    assert "data:image/png;base64," in response.text


def test_enrol_without_a_detectable_region_is_reported(client):
    blank = np.zeros((12, 12, 3), dtype=np.uint8)
    files = {"file": ("blank.png", _png_bytes(blank), "image/png")}
    response = client.post(
        "/enrol", files=files, data={"label": "nothing", "domain": "face"}
    )
    assert response.status_code == 200
    assert "not enrolled" in response.text

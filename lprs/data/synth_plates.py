"""Synthetic licence-plate generator.

Public plate datasets are either tiny, licence-encumbered, or full of real
vehicle identifiers -- a privacy problem you do not want in a coursework
repository. Generating plates instead gives us unlimited, perfectly-labelled,
PII-free data, and crucially lets us dial the *difficulty* (blur, skew,
glare, noise) to measure exactly where the model breaks.

The augmentation chain deliberately mimics real failure modes:
perspective from off-axis cameras, motion blur from moving vehicles,
specular glare from headlights, and sensor noise from night capture.
"""

from __future__ import annotations

import random
import string
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

__all__ = [
    "PlateStyle",
    "random_plate_text",
    "render_plate",
    "augment_plate",
    "jitter_crop",
    "compose_scene",
    "generate_identity_dataset",
]

#: Format tokens: ``A`` = letter, ``0`` = digit, anything else is a literal.
DEFAULT_FORMATS = ("AA00AA0000", "AA00AAA000", "AAA0000", "00AA0000")

_FONT_CANDIDATES = (
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Courier New Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
)


@dataclass(frozen=True)
class PlateStyle:
    """Colour scheme and geometry of a rendered plate."""

    background: tuple[int, int, int] = (235, 235, 235)
    foreground: tuple[int, int, int] = (20, 20, 20)
    border: tuple[int, int, int] = (40, 40, 40)
    width: int = 320
    height: int = 96

    @classmethod
    def palette(cls) -> tuple["PlateStyle", ...]:
        """A handful of real-world plate colourways."""
        return (
            cls(),  # white on black text (EU/India private)
            cls(background=(45, 190, 235), foreground=(15, 15, 15)),  # yellow
            cls(background=(40, 120, 40), foreground=(245, 245, 245),
                border=(230, 230, 230)),  # green commercial
            cls(background=(30, 30, 30), foreground=(240, 240, 240),
                border=(200, 200, 200)),  # black
        )


def _load_font(size: int) -> ImageFont.ImageFont | None:
    for candidate in _FONT_CANDIDATES:
        if Path(candidate).exists():
            try:
                return ImageFont.truetype(candidate, size)
            except OSError:
                continue
    return None


def random_plate_text(rng: random.Random, plate_format: str | None = None) -> str:
    """Build a plate string from a format template."""
    template = plate_format or rng.choice(DEFAULT_FORMATS)
    characters = []
    for token in template:
        if token == "A":
            characters.append(rng.choice(string.ascii_uppercase))
        elif token == "0":
            characters.append(rng.choice(string.digits))
        else:
            characters.append(token)
    return "".join(characters)


def _draw_with_cv2(canvas: np.ndarray, text: str, style: PlateStyle) -> np.ndarray:
    """Fallback renderer when no TrueType font is installed."""
    font = cv2.FONT_HERSHEY_DUPLEX
    scale = 1.6
    thickness = 3
    (text_width, text_height), _ = cv2.getTextSize(text, font, scale, thickness)
    scale *= min(1.0, (style.width - 24) / max(text_width, 1))
    (text_width, text_height), _ = cv2.getTextSize(text, font, scale, thickness)
    origin = ((style.width - text_width) // 2, (style.height + text_height) // 2)
    cv2.putText(canvas, text, origin, font, scale, style.foreground, thickness, cv2.LINE_AA)
    return canvas


def render_plate(text: str, style: PlateStyle | None = None) -> np.ndarray:
    """Render a clean, axis-aligned plate image in BGR."""
    style = style or PlateStyle()
    canvas = np.full((style.height, style.width, 3), style.background, dtype=np.uint8)
    cv2.rectangle(
        canvas, (3, 3), (style.width - 4, style.height - 4), style.border, thickness=3
    )

    font = _load_font(int(style.height * 0.62))
    if font is None:
        return _draw_with_cv2(canvas, text, style)

    # PIL works in RGB; convert at the boundary and convert straight back.
    pil_image = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    position = (
        (style.width - (right - left)) // 2 - left,
        (style.height - (bottom - top)) // 2 - top,
    )
    draw.text(position, text, font=font, fill=style.foreground[::-1])
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


# --------------------------------------------------------------------------
# Augmentation -- each function models one real-world degradation
# --------------------------------------------------------------------------


def _perspective(image: np.ndarray, rng: random.Random, severity: float) -> np.ndarray:
    height, width = image.shape[:2]
    jitter = severity * 0.16
    source = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    destination = np.float32(
        [
            [rng.uniform(0, width * jitter), rng.uniform(0, height * jitter)],
            [width - rng.uniform(0, width * jitter), rng.uniform(0, height * jitter)],
            [width - rng.uniform(0, width * jitter), height - rng.uniform(0, height * jitter)],
            [rng.uniform(0, width * jitter), height - rng.uniform(0, height * jitter)],
        ]
    )
    matrix = cv2.getPerspectiveTransform(source, destination)
    return cv2.warpPerspective(image, matrix, (width, height), borderMode=cv2.BORDER_REPLICATE)


def _motion_blur(image: np.ndarray, rng: random.Random, severity: float) -> np.ndarray:
    length = int(1 + severity * rng.uniform(1, 7))
    if length < 2:
        return image
    kernel = np.zeros((length, length), dtype=np.float32)
    kernel[length // 2, :] = 1.0 / length
    if rng.random() < 0.35:  # occasional vertical shake
        kernel = kernel.T
    return cv2.filter2D(image, -1, kernel)


def _glare(image: np.ndarray, rng: random.Random, severity: float) -> np.ndarray:
    height, width = image.shape[:2]
    overlay = np.zeros((height, width), dtype=np.float32)
    centre = (rng.randint(0, width), rng.randint(0, height))
    radius = int(max(width, height) * rng.uniform(0.2, 0.55))
    cv2.circle(overlay, centre, radius, 1.0, thickness=-1)
    overlay = cv2.GaussianBlur(overlay, (0, 0), sigmaX=radius / 2.5)
    strength = severity * rng.uniform(20, 70)
    return np.clip(image + (overlay[:, :, None] * strength), 0, 255).astype(np.uint8)


def _sensor_noise(image: np.ndarray, rng: random.Random, severity: float) -> np.ndarray:
    sigma = severity * rng.uniform(2, 14)
    noise = np.random.default_rng(rng.randrange(2**32)).normal(0, sigma, image.shape)
    return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def _exposure(image: np.ndarray, rng: random.Random, severity: float) -> np.ndarray:
    gain = 1.0 + severity * rng.uniform(-0.35, 0.35)
    bias = severity * rng.uniform(-30, 30)
    return np.clip(image.astype(np.float32) * gain + bias, 0, 255).astype(np.uint8)


def augment_plate(
    image: np.ndarray,
    rng: random.Random,
    severity: float = 0.6,
) -> np.ndarray:
    """Apply the full degradation chain at the requested severity in [0, 1]."""
    severity = float(np.clip(severity, 0.0, 1.0))
    if severity == 0.0:
        return image

    pipeline = (_perspective, _exposure, _glare, _motion_blur, _sensor_noise)
    result = image
    for step in pipeline:
        if rng.random() < 0.75:
            result = step(result, rng, severity)
    return result


def jitter_crop(
    image: np.ndarray,
    rng: random.Random,
    max_shift: float = 0.10,
    max_scale: float = 0.22,
) -> np.ndarray:
    """Re-crop the plate with a random offset and scale, over background.

    This is the single most important augmentation in the whole project. A
    detector never returns the perfect box -- it returns something 10-25% too
    large, slightly offset, with asphalt in the corners. An embedder trained
    only on pristine tight crops is therefore being asked at inference time
    for a kind of image it has literally never seen, and its accuracy
    collapses (measured: genuine cosine similarity 0.99 -> 0.66).

    Simulating that box imprecision during training makes the embedding
    invariant to it, which is exactly what face-recognition and person-re-id
    pipelines do with random-resized-crop.
    """
    height, width = image.shape[:2]

    # Surround the plate with plausible background so the jittered box has
    # something realistic to include at the edges.
    pad_x = int(width * (max_scale + max_shift) + 2)
    pad_y = int(height * (max_scale + max_shift) + 2)
    generator = np.random.default_rng(rng.randrange(2**32))
    canvas = generator.integers(
        55, 165, size=(height + 2 * pad_y, width + 2 * pad_x, 3), dtype=np.int16
    ).astype(np.uint8)
    canvas = cv2.GaussianBlur(canvas, (7, 7), 0)
    canvas[pad_y : pad_y + height, pad_x : pad_x + width] = image

    grow_x = rng.uniform(-max_scale * 0.35, max_scale)
    grow_y = rng.uniform(-max_scale * 0.35, max_scale)
    shift_x = rng.uniform(-max_shift, max_shift)
    shift_y = rng.uniform(-max_shift, max_shift)

    left = int(pad_x + width * (shift_x - grow_x / 2))
    top = int(pad_y + height * (shift_y - grow_y / 2))
    box_width = int(width * (1 + grow_x))
    box_height = int(height * (1 + grow_y))

    left = max(0, min(left, canvas.shape[1] - 2))
    top = max(0, min(top, canvas.shape[0] - 2))
    box_width = max(8, min(box_width, canvas.shape[1] - left))
    box_height = max(8, min(box_height, canvas.shape[0] - top))
    return canvas[top : top + box_height, left : left + box_width]


def compose_scene(
    plate: np.ndarray,
    rng: random.Random,
    scene_size: tuple[int, int] = (640, 400),
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Paste a plate into a textured background; return the scene and its box.

    Used to evaluate the *detector* rather than the recogniser, since a plate
    crop alone cannot tell you whether localisation works.
    """
    width, height = scene_size
    generator = np.random.default_rng(rng.randrange(2**32))
    base = generator.integers(60, 150, size=(height, width, 3), dtype=np.int16)
    gradient = np.linspace(-35, 35, height, dtype=np.int16)[:, None, None]
    scene = np.clip(base + gradient, 0, 255).astype(np.uint8)
    scene = cv2.GaussianBlur(scene, (9, 9), 0)

    scale = rng.uniform(0.35, 0.62)
    plate_width = max(48, int(width * scale))
    plate_height = max(16, int(plate_width * plate.shape[0] / plate.shape[1]))
    resized = cv2.resize(plate, (plate_width, plate_height), interpolation=cv2.INTER_AREA)

    x = rng.randint(0, max(0, width - plate_width))
    y = rng.randint(0, max(0, height - plate_height))
    scene[y : y + plate_height, x : x + plate_width] = resized
    return scene, (x, y, plate_width, plate_height)


def generate_identity_dataset(
    num_identities: int,
    samples_per_identity: int,
    severity: float = 0.6,
    seed: int = 1337,
    output_dir: Path | None = None,
    crop_jitter: bool = True,
    jitter_shift: float = 0.10,
    jitter_scale: float = 0.22,
) -> tuple[list[np.ndarray], list[str]]:
    """Produce ``num_identities`` plates, each seen ``samples_per_identity`` times.

    This is the structure a metric-learning model needs: multiple degraded
    views of the same identity so triplet loss has positives to pull together.

    ``crop_jitter`` simulates imperfect detector boxes and should stay on for
    anything that will later be fed real detections -- see :func:`jitter_crop`.
    Transcription training wants gentler values than matching, because a crop
    that clips a character turns into a mislabelled training example.
    """
    rng = random.Random(seed)
    styles = PlateStyle.palette()

    images: list[np.ndarray] = []
    labels: list[str] = []
    seen: set[str] = set()

    for _ in range(num_identities):
        text = random_plate_text(rng)
        while text in seen:
            text = random_plate_text(rng)
        seen.add(text)

        style = rng.choice(styles)
        clean = render_plate(text, style)
        for _ in range(samples_per_identity):
            sample = augment_plate(clean, rng, severity)
            if crop_jitter:
                sample = jitter_crop(sample, rng, jitter_shift, jitter_scale)
            images.append(sample)
            labels.append(text)

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for index, (image, label) in enumerate(zip(images, labels)):
            cv2.imwrite(str(output_dir / f"{index:05d}_{label}.png"), image)

    return images, labels

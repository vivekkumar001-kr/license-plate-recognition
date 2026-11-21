"""Central configuration for the License Plate Recognition System.

Every tunable lives here so experiments are reproducible and nothing is
hard-coded three files deep. One source of truth, as the Zen intends.
"""

from __future__ import annotations

import string
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
SYNTHETIC_DIR = DATA_DIR / "synthetic"
ENROLLED_DIR = DATA_DIR / "enrolled"
ARTIFACT_DIR = ROOT / "artifacts"
REPORT_DIR = ROOT / "reports"
SAMPLE_DIR = DATA_DIR / "samples"

for _directory in (DATA_DIR, ARTIFACT_DIR, REPORT_DIR):
    _directory.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# Character vocabulary (shared by the synthesiser, the CTC head and decoding)
# --------------------------------------------------------------------------

#: Index 0 is reserved for the CTC blank label.
BLANK = "_"
ALPHABET = BLANK + string.digits + string.ascii_uppercase
CHAR_TO_INDEX = {char: index for index, char in enumerate(ALPHABET)}
INDEX_TO_CHAR = {index: char for char, index in CHAR_TO_INDEX.items()}
NUM_CLASSES = len(ALPHABET)


# --------------------------------------------------------------------------
# Image geometry
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PlateGeometry:
    """Canonical size a cropped plate is warped to before any model sees it."""

    width: int = 160
    height: int = 48
    min_aspect_ratio: float = 1.8
    max_aspect_ratio: float = 7.0
    min_area: int = 900

    @property
    def size(self) -> tuple[int, int]:
        return (self.width, self.height)


@dataclass(frozen=True)
class FaceGeometry:
    """Canonical size a cropped face is warped to (FaceNet convention)."""

    width: int = 112
    height: int = 112
    min_aspect_ratio: float = 0.65
    max_aspect_ratio: float = 1.55
    min_area: int = 1600

    @property
    def size(self) -> tuple[int, int]:
        return (self.width, self.height)


# --------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PreprocessConfig:
    """Switches for the image-preprocessing pipeline.

    Defaults are tuned for plates; the face pipeline disables deskewing and
    binarisation because facial texture must survive intact.
    """

    to_grayscale: bool = True
    clahe_clip_limit: float = 2.5
    clahe_tile_grid: tuple[int, int] = (8, 8)
    bilateral_diameter: int = 7
    bilateral_sigma_colour: float = 45.0
    bilateral_sigma_space: float = 45.0
    deskew: bool = True
    max_deskew_degrees: float = 20.0
    sharpen: bool = True
    normalise: bool = True

    @classmethod
    def for_faces(cls) -> "PreprocessConfig":
        return cls(
            to_grayscale=False,
            clahe_clip_limit=2.0,
            bilateral_diameter=5,
            deskew=False,
            sharpen=False,
        )


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CropJitter:
    """Framing error to simulate during training.

    These numbers are **measured, not guessed**. Running
    ``scripts/measure_detector_error.py`` over 150 synthetic scenes gives the
    relative error between detector boxes and ground truth::

        shift_x   p5 -0.110   p50 -0.040   p95 +0.116
        shift_y   p5 -0.276   p50 -0.168   p95 +0.269
        scale_w   p5 -0.200   p50 +0.103   p95 +0.269
        scale_h   p5 -0.567   p50 +0.234   p95 +0.420

    The error is strongly **anisotropic**: Haar plate boxes are far sloppier
    vertically than horizontally, and they tend to overshoot. An isotropic
    jitter would under-cover the vertical case, which is exactly the failure
    mode that tanked end-to-end accuracy.
    """

    shift_x: float = 0.13
    shift_y: float = 0.24
    scale_width: tuple[float, float] = (-0.20, 0.27)
    # The measured p5 of -0.567 comes from the contour detector's occasional
    # thin-strip boxes, which chop characters in half. Simulating those
    # creates positives that are genuinely unidentifiable and that actively
    # destabilise training, so the shrink end is clamped well short of it.
    scale_height: tuple[float, float] = (-0.25, 0.45)

    @classmethod
    def gentle(cls) -> "CropJitter":
        """Milder jitter for CTC training.

        A crop that clips a character hands the transcriber a label it cannot
        possibly justify, so the aggressive shrink end is pulled back.
        """
        return cls(
            shift_x=0.07,
            shift_y=0.14,
            scale_width=(-0.08, 0.25),
            scale_height=(-0.12, 0.40),
        )


@dataclass(frozen=True)
class EmbedderConfig:
    """FaceNet-style embedding network hyper-parameters."""

    embedding_dim: int = 128
    channels: tuple[int, ...] = (32, 64, 128, 256)
    dropout: float = 0.2
    triplet_margin: float = 0.3
    #: Use softplus(d+ - d-) instead of a hard hinge.
    #:
    #: With a hard margin on a difficult task the cheapest way to reduce the
    #: loss is to shrink the whole embedding space, and batch-hard mining
    #: happily drives it there: measured spread collapsed to 0.168 with every
    #: cosine similarity above 0.96, which makes any absolute threshold
    #: useless even though the *ranking* stayed good. The soft margin has no
    #: unreachable target and keeps rewarding extra separation, so the space
    #: stays spread out (Hermans et al., "In Defense of the Triplet Loss").
    soft_margin: bool = True
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    epochs: int = 12


@dataclass(frozen=True)
class OCRConfig:
    """CRNN (CNN + BiLSTM + CTC) hyper-parameters for plate transcription."""

    hidden_size: int = 160
    lstm_layers: int = 2
    dropout: float = 0.15
    learning_rate: float = 1e-3
    batch_size: int = 48
    epochs: int = 18
    max_label_length: int = 10


# --------------------------------------------------------------------------
# Matching
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class MatchConfig:
    """Similarity-matching thresholds.

    ``accept_threshold`` is cosine similarity in [-1, 1]; anything below it is
    reported as ``UNKNOWN`` rather than forced onto the nearest neighbour.
    """

    accept_threshold: float = 0.62
    top_k: int = 5
    use_text_fallback: bool = True
    text_distance_tolerance: int = 1


@dataclass(frozen=True)
class Settings:
    """Aggregate settings object passed around the pipeline."""

    plate_geometry: PlateGeometry = field(default_factory=PlateGeometry)
    face_geometry: FaceGeometry = field(default_factory=FaceGeometry)
    embedder: EmbedderConfig = field(default_factory=EmbedderConfig)
    ocr: OCRConfig = field(default_factory=OCRConfig)
    matching: MatchConfig = field(default_factory=MatchConfig)
    random_seed: int = 1337


SETTINGS = Settings()

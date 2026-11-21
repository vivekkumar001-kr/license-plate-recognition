"""Image preprocessing.

A small, composable set of OpenCV operations that turn a raw crop into a
clean, canonically-sized tensor-ready array. Every step is individually
testable and individually switchable via :class:`PreprocessConfig`.
"""

from __future__ import annotations

import cv2
import numpy as np

from .config import PreprocessConfig

__all__ = [
    "to_grayscale",
    "apply_clahe",
    "denoise",
    "estimate_skew_angle",
    "deskew",
    "unsharp_mask",
    "resize_keep_aspect",
    "Preprocessor",
]


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Collapse to a single luminance channel if the image has colour."""
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def apply_clahe(
    image: np.ndarray,
    clip_limit: float,
    tile_grid: tuple[int, int],
) -> np.ndarray:
    """Contrast Limited Adaptive Histogram Equalisation.

    Plates photographed in shade or glare have crushed local contrast; CLAHE
    rescues it without the global blow-out that plain ``equalizeHist`` causes.
    Colour images are equalised on the L channel of LAB so hue is preserved.
    """
    if clip_limit <= 0:
        return image

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    if image.ndim == 2:
        return clahe.apply(image)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def denoise(
    image: np.ndarray,
    diameter: int,
    sigma_colour: float,
    sigma_space: float,
) -> np.ndarray:
    """Edge-preserving bilateral blur.

    Chosen over Gaussian precisely because character strokes must stay crisp
    while sensor noise and asphalt texture are smoothed away.
    """
    if diameter <= 0:
        return image
    return cv2.bilateralFilter(image, diameter, sigma_colour, sigma_space)


def estimate_skew_angle(gray: np.ndarray, max_degrees: float) -> float:
    """Estimate plate rotation from the minimum-area rectangle of ink pixels."""
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = cv2.findNonZero(binary)
    if coords is None or len(coords) < 10:
        return 0.0

    angle = cv2.minAreaRect(coords)[-1]
    # OpenCV's reported angle has drifted between releases: 4.5+ documents
    # (0, 90] but still returns -90 for axis-aligned boxes. Folding modulo 90
    # into [-45, 45] is correct under every convention.
    angle = angle % 90.0
    if angle > 45.0:
        angle -= 90.0
    if abs(angle) > max_degrees:
        return 0.0
    return float(angle)


def deskew(image: np.ndarray, max_degrees: float) -> np.ndarray:
    """Rotate the crop so character baselines run horizontally."""
    gray = to_grayscale(image)
    angle = estimate_skew_angle(gray, max_degrees)
    if abs(angle) < 0.25:
        return image

    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, 1.0)
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def unsharp_mask(image: np.ndarray, amount: float = 0.8) -> np.ndarray:
    """Classic unsharp mask to recover stroke definition after denoising."""
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=1.6)
    return cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)


def resize_keep_aspect(
    image: np.ndarray,
    size: tuple[int, int],
    pad_value: int = 127,
) -> np.ndarray:
    """Resize into ``size`` (w, h) letterboxing rather than stretching.

    Stretching a 2:1 plate into a 3.3:1 box distorts glyph shapes and the CNN
    pays for it later, so we pad instead.
    """
    target_width, target_height = size
    height, width = image.shape[:2]
    if height == 0 or width == 0:
        raise ValueError("Cannot resize an empty image")

    scale = min(target_width / width, target_height / height)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    resized = cv2.resize(image, (new_width, new_height), interpolation=interpolation)

    canvas_shape: tuple[int, ...]
    if image.ndim == 2:
        canvas_shape = (target_height, target_width)
    else:
        canvas_shape = (target_height, target_width, image.shape[2])
    canvas = np.full(canvas_shape, pad_value, dtype=image.dtype)

    top = (target_height - new_height) // 2
    left = (target_width - new_width) // 2
    canvas[top : top + new_height, left : left + new_width] = resized
    return canvas


class Preprocessor:
    """Applies the configured preprocessing chain to a crop.

    One instance is built per domain (plates vs faces) and then reused, which
    keeps the expensive CLAHE object allocation out of the hot loop.
    """

    def __init__(self, size: tuple[int, int], config: PreprocessConfig) -> None:
        self.size = size
        self.config = config

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return self.run(image)

    def run(self, image: np.ndarray) -> np.ndarray:
        """Return a float32 array in [0, 1] shaped (H, W) or (H, W, 3)."""
        if image is None or image.size == 0:
            raise ValueError("Preprocessor received an empty image")

        working = image
        if working.dtype != np.uint8:
            working = np.clip(working, 0, 255).astype(np.uint8)

        config = self.config
        if config.to_grayscale:
            working = to_grayscale(working)

        working = apply_clahe(working, config.clahe_clip_limit, config.clahe_tile_grid)
        working = denoise(
            working,
            config.bilateral_diameter,
            config.bilateral_sigma_colour,
            config.bilateral_sigma_space,
        )

        if config.deskew:
            working = deskew(working, config.max_deskew_degrees)
        if config.sharpen:
            working = unsharp_mask(working)

        working = resize_keep_aspect(working, self.size)

        if not config.normalise:
            return working
        return working.astype(np.float32) / 255.0

    def batch(self, images: list[np.ndarray]) -> np.ndarray:
        """Preprocess a list of crops into a single stacked array."""
        if not images:
            return np.empty((0, self.size[1], self.size[0]), dtype=np.float32)
        return np.stack([self.run(image) for image in images])

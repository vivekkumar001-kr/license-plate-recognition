"""Loading labelled images from disk.

Expected layout (the universal "image folder" convention)::

    root/
        MH12AB1234/
            front.jpg
            night.jpg
        KA05CD9876/
            ...

The directory name *is* the identity label, which keeps enrolment data
self-describing and avoids a separate manifest that can drift out of sync.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

__all__ = ["IMAGE_SUFFIXES", "load_image", "load_image_folder"]

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"})


def load_image(path: str | Path) -> np.ndarray:
    """Read an image as BGR, raising a useful error if OpenCV returns None."""
    path = Path(path)
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read an image from {path}")
    return image


def load_image_folder(
    root: str | Path,
    min_samples_per_label: int = 1,
) -> tuple[list[np.ndarray], list[str]]:
    """Load every image under ``root``, labelled by its parent directory."""
    root = Path(root)
    if not root.is_dir():
        raise NotADirectoryError(f"{root} is not a directory")

    images: list[np.ndarray] = []
    labels: list[str] = []
    for label_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        files = sorted(
            path for path in label_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
        )
        if len(files) < min_samples_per_label:
            continue
        for file_path in files:
            images.append(load_image(file_path))
            labels.append(label_dir.name)

    if not images:
        raise ValueError(
            f"No images found under {root}. Expected subdirectories named after "
            f"each identity, containing files with suffixes {sorted(IMAGE_SUFFIXES)}."
        )
    return images, labels

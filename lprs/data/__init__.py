"""Synthetic data generation, disk loading, and torch dataset plumbing."""

from __future__ import annotations

from .datasets import CTCPlateDataset, ImageLabelDataset, PKSampler, collate_ctc
from .loaders import load_image, load_image_folder
from .synth_plates import (
    PlateStyle,
    augment_plate,
    compose_scene,
    generate_identity_dataset,
    random_plate_text,
    render_plate,
)

__all__ = [
    "CTCPlateDataset",
    "ImageLabelDataset",
    "PKSampler",
    "PlateStyle",
    "augment_plate",
    "collate_ctc",
    "compose_scene",
    "generate_identity_dataset",
    "load_image",
    "load_image_folder",
    "random_plate_text",
    "render_plate",
]

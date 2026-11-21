"""Shared fixtures."""

from __future__ import annotations

import random

import numpy as np
import pytest

from lprs.config import PlateGeometry, PreprocessConfig
from lprs.data import compose_scene, generate_identity_dataset, render_plate
from lprs.preprocess import Preprocessor


@pytest.fixture(scope="session")
def geometry() -> PlateGeometry:
    return PlateGeometry()


@pytest.fixture(scope="session")
def preprocessor(geometry: PlateGeometry) -> Preprocessor:
    return Preprocessor(geometry.size, PreprocessConfig())


@pytest.fixture(scope="session")
def clean_plate() -> np.ndarray:
    return render_plate("MH12AB1234")


@pytest.fixture(scope="session")
def scene_with_box(clean_plate: np.ndarray):
    return compose_scene(clean_plate, random.Random(5))


@pytest.fixture(scope="session")
def small_dataset() -> tuple[list[np.ndarray], list[str]]:
    return generate_identity_dataset(
        num_identities=8, samples_per_identity=4, severity=0.4, seed=11
    )

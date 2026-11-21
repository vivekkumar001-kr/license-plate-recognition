"""Diagnose embedding-space collapse and compare against the HOG baseline.

Symptom: calibrated threshold 0.9907 but 100% false accepts. That pattern
means the embeddings all sit on top of each other -- the *ranking* is fine
(AUC 0.9866) but every cosine similarity is ~0.99, so no single global
threshold can separate genuine from impostor.
"""

import random

import numpy as np

from lprs.config import PlateGeometry, PreprocessConfig
from lprs.data import augment_plate, jitter_crop, random_plate_text, render_plate
from lprs.embeddings import build_embedder
from lprs.preprocess import Preprocessor

geometry = PlateGeometry()
preprocessor = Preprocessor(geometry.size, PreprocessConfig())
rng = random.Random(777)

labels, images = [], []
for _ in range(60):
    text = random_plate_text(rng)
    for _ in range(4):
        images.append(jitter_crop(augment_plate(render_plate(text), rng, 0.6), rng))
        labels.append(text)

batch = preprocessor.batch(images)
label_array = np.asarray(labels)

print(f"{len(images)} crops / {len(set(labels))} identities\n")
print(f"{'model':6s} {'genuine':>9s} {'impostor':>9s} {'margin':>8s} "
      f"{'spread':>8s} {'top1':>7s}")
print("-" * 54)

for kind, checkpoint in (("cnn", "artifacts/embedder_plate.pt"), ("hog", None), ("raw", None)):
    embedder = build_embedder(kind, geometry.size, checkpoint=checkpoint)
    vectors = embedder.embed(batch)

    similarity = vectors @ vectors.T
    same = label_array[:, None] == label_array[None, :]
    off_diagonal = ~np.eye(len(labels), dtype=bool)

    genuine = similarity[same & off_diagonal].mean()
    impostor = similarity[~same].mean()

    # Spread: mean pairwise distance from the centroid. Near zero == collapse.
    centroid = vectors.mean(axis=0)
    spread = float(np.linalg.norm(vectors - centroid, axis=1).mean())

    ranked = similarity - np.eye(len(labels)) * 2
    nearest = ranked.argmax(axis=1)
    top1 = float((label_array[nearest] == label_array).mean())

    print(f"{kind:6s} {genuine:9.4f} {impostor:9.4f} {genuine - impostor:+8.4f} "
          f"{spread:8.4f} {top1:7.1%}")

print("\nspread near 0 means the embedding space has collapsed to a point;")
print("a healthy unit-hypersphere embedding should spread around 0.7-1.0")

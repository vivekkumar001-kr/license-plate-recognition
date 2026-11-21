"""Measure the embedding space using REAL detector crops.

Everything so far was measured on `jitter_crop` simulations. If the
simulation and the detector still disagree, impostors will cluster and the
threshold will keep failing. This compares three enrolment strategies using
identical probe images, all passed through the actual pipeline.
"""

import random

import numpy as np

from lprs.data import augment_plate, compose_scene, jitter_crop, random_plate_text, render_plate
from lprs.pipeline import build_pipeline

pipeline = build_pipeline(domain="plate", embedder_kind="cnn")
rng = random.Random(31337)

NUM_IDENTITIES = 40
texts = []
seen = set()
while len(texts) < NUM_IDENTITIES:
    text = random_plate_text(rng)
    if text not in seen:
        seen.add(text)
        texts.append(text)


def detector_crop_embedding(label: str, severity: float):
    """Full real path: scene -> detect -> align -> preprocess -> embed."""
    scene, _ = compose_scene(augment_plate(render_plate(label), rng, severity), rng)
    pairs = pipeline.crops_from(scene, max_regions=1)
    if not pairs:
        return None
    return pipeline.embedder.embed_one(pairs[0][1])


def jitter_embedding(label: str, severity: float):
    """Simulated path, as used for enrolment today."""
    crop = jitter_crop(augment_plate(render_plate(label), rng, severity), rng)
    return pipeline.embed_image(crop)


def report(name, gallery_vectors, gallery_labels, probe_vectors, probe_labels):
    gallery = np.vstack(gallery_vectors)
    probes = np.vstack(probe_vectors)
    similarity = probes @ gallery.T

    same = np.asarray(probe_labels)[:, None] == np.asarray(gallery_labels)[None, :]
    genuine = similarity[same]
    impostor = similarity[~same]

    predictions = np.asarray(gallery_labels)[similarity.argmax(axis=1)]
    top1 = float((predictions == np.asarray(probe_labels)).mean())

    # Best achievable threshold separation.
    overlap = float((impostor > genuine.mean()).mean())
    print(
        f"{name:34s} top1={top1:6.1%}  genuine={genuine.mean():.3f}  "
        f"impostor={impostor.mean():.3f}  p99_impostor={np.percentile(impostor, 99):.3f}"
        f"  overlap={overlap:.3f}"
    )


# Probes: always the real detector path.
probe_vectors, probe_labels = [], []
for text in texts:
    vector = detector_crop_embedding(text, 0.7)
    if vector is not None:
        probe_vectors.append(vector)
        probe_labels.append(text)

print(f"{len(probe_vectors)} probes via the real detector\n")

# Strategy A: enrol from simulated jitter crops (what we do today).
jitter_vectors = [jitter_embedding(text, 0.3) for text in texts]
report("A enrol=jitter  probe=detector", jitter_vectors, texts, probe_vectors, probe_labels)

# Strategy B: enrol from real detector crops (distribution-matched).
detector_vectors, detector_labels = [], []
for text in texts:
    vector = detector_crop_embedding(text, 0.3)
    if vector is not None:
        detector_vectors.append(vector)
        detector_labels.append(text)
report("B enrol=detector probe=detector", detector_vectors, detector_labels,
       probe_vectors, probe_labels)

# Strategy C: enrol from tight crops (the naive original).
tight_vectors = [pipeline.embed_image(render_plate(text)) for text in texts]
report("C enrol=tight   probe=detector", tight_vectors, texts, probe_vectors, probe_labels)

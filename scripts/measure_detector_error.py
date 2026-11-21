"""Measure the real framing error of the detector.

jitter_crop's magnitude should not be a guess. This samples the actual
distribution of (shift, scale) error between detector boxes and ground
truth, so the augmentation can be set to cover what really happens.
"""

import random

import numpy as np

from lprs.data import augment_plate, compose_scene, random_plate_text, render_plate
from lprs.detectors import Detection, build_detector

detector = build_detector("plate")
rng = random.Random(1234)

shift_x, shift_y, scale_w, scale_h, ious = [], [], [], [], []
misses = 0

for _ in range(150):
    plate = render_plate(random_plate_text(rng))
    scene, truth_box = compose_scene(augment_plate(plate, rng, 0.6), rng)
    truth = Detection(*truth_box)

    detections = detector.detect(scene)
    if not detections:
        misses += 1
        continue

    best = max(detections, key=lambda d: d.intersection_over_union(truth))
    ious.append(best.intersection_over_union(truth))

    # Relative error of the detected box versus ground truth.
    shift_x.append((best.x - truth.x) / truth.width)
    shift_y.append((best.y - truth.y) / truth.height)
    scale_w.append(best.width / truth.width - 1.0)
    scale_h.append(best.height / truth.height - 1.0)


def summarise(name, values):
    array = np.asarray(values)
    print(
        f"  {name:10s} mean={array.mean():+.3f}  p5={np.percentile(array, 5):+.3f}  "
        f"p50={np.percentile(array, 50):+.3f}  p95={np.percentile(array, 95):+.3f}"
    )


print(f"detections: {len(ious)}/150 (misses: {misses})")
print(f"mean IoU: {np.mean(ious):.3f}\n")
print("relative framing error of detector boxes vs ground truth:")
summarise("shift_x", shift_x)
summarise("shift_y", shift_y)
summarise("scale_w", scale_w)
summarise("scale_h", scale_h)

print("\ncurrent jitter_crop covers shift +/-0.10 and scale -0.08..+0.22")
print("-> required to cover p5..p95:")
print(f"   shift: +/-{max(abs(np.percentile(shift_x, 5)), abs(np.percentile(shift_x, 95)), abs(np.percentile(shift_y, 5)), abs(np.percentile(shift_y, 95))):.2f}")
print(f"   scale: {min(np.percentile(scale_w, 5), np.percentile(scale_h, 5)):+.2f} .. {max(np.percentile(scale_w, 95), np.percentile(scale_h, 95)):+.2f}")

"""Train a FaceNet-style embedder for *faces*.

Why this exists separately from the plate embedder trainer:

- Plates can be generated synthetically (no PII) so training data ships.
- Faces cannot (ethically) ship with a dataset in this repo.

So face training is **bring-your-own-data** using the same folder convention
as enrolment::

    root/
      alice/
        001.jpg
        002.jpg
      bob/
        001.jpg
        ...

This trainer optionally runs OpenCV face detection + eye-line alignment on
those photos so you can point it at "raw" images, not just pre-cropped faces.

It uses the exact same CNN + batch-hard triplet loss recipe as
:func:`lprs.training.train_embedder.train_embedder` and stores a calibrated
acceptance threshold inside the checkpoint.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..align import EyeAligner
from ..config import ARTIFACT_DIR, EmbedderConfig, FaceGeometry, PreprocessConfig
from ..data.datasets import ImageLabelDataset, PKSampler
from ..data.loaders import load_image_folder
from ..detectors import build_detector
from ..embeddings.cnn import EmbeddingNet
from ..evaluation.metrics import build_pairs, evaluate_verification
from ..preprocess import Preprocessor
from .train_embedder import EmbedderTrainingResult, _split_by_identity, _embed_all, embedding_spread
from .triplet import BatchHardTripletLoss

__all__ = ["train_face_embedder"]


def _extract_face_crops(
    images: list[np.ndarray],
    labels: list[str],
    crop_margin: float = 0.06,
    seed: int = 1337,
) -> tuple[list[np.ndarray], list[str]]:
    """Run OpenCV face detection and keep the best crop per image.

    This keeps training aligned with inference: the embedder learns on the
    same kind of crops the pipeline will feed it.
    """

    detector = build_detector("face")
    aligner = EyeAligner()
    rng = random.Random(seed)

    crops: list[np.ndarray] = []
    crop_labels: list[str] = []

    for image, label in zip(images, labels):
        # Deterministic-ish: ties break randomly so we don't always pick the
        # exact same face when multiple are present.
        detections = detector.detect(image)
        if not detections:
            continue
        rng.shuffle(detections)
        best = max(detections, key=lambda det: det.score)
        crop = best.crop(image, margin=crop_margin)
        if crop is None or crop.size == 0:
            continue
        crops.append(aligner.align(crop))
        crop_labels.append(label)

    return crops, crop_labels


def train_face_embedder(
    source: str | Path,
    epochs: int | None = None,
    identities_per_batch: int = 8,
    samples_per_batch_identity: int = 4,
    validation_fraction: float = 0.25,
    device: str = "cpu",
    seed: int = 1337,
    config: EmbedderConfig | None = None,
    output_path: Path | None = None,
    detect_first: bool = True,
    min_samples_per_label: int = 2,
    verbose: bool = True,
) -> EmbedderTrainingResult:
    """Train the face embedder on an on-disk dataset."""

    torch.manual_seed(seed)
    np.random.seed(seed)

    config = config or EmbedderConfig()
    epochs = epochs or config.epochs
    geometry = FaceGeometry()
    output_path = Path(output_path or ARTIFACT_DIR / "embedder_face.pt")
    torch_device = torch.device(device)

    raw_images, raw_labels = load_image_folder(source, min_samples_per_label=min_samples_per_label)

    if detect_first:
        images, labels = _extract_face_crops(raw_images, raw_labels, seed=seed)
    else:
        images, labels = raw_images, raw_labels

    if len(set(labels)) < 2:
        raise ValueError(
            "Need at least 2 identities after preprocessing (face detection may have dropped too many images)."
        )

    (train_images, train_labels), (val_images, val_labels) = _split_by_identity(
        images, labels, validation_fraction, seed
    )

    preprocess_config = PreprocessConfig.for_faces()
    preprocessor = Preprocessor(geometry.size, preprocess_config)
    train_set = ImageLabelDataset(train_images, train_labels, preprocessor)
    val_set = ImageLabelDataset(val_images, val_labels, preprocessor)

    if verbose:
        kept = len(images)
        dropped = len(raw_images) - kept
        print(
            f"loaded {len(raw_images)} images across {len(set(raw_labels))} identities from {Path(source).resolve()}"
        )
        if detect_first:
            print(f"face-detection kept {kept}, dropped {dropped}")
        print(
            f"train: {len(train_set)} samples / {train_set.num_classes} identities | "
            f"val: {len(val_set)} samples / {val_set.num_classes} identities (disjoint)"
        )

    sampler = PKSampler(
        train_set.targets,
        identities_per_batch=identities_per_batch,
        samples_per_identity=samples_per_batch_identity,
        seed=seed,
    )
    loader = DataLoader(train_set, batch_size=sampler.batch_size, sampler=sampler)

    model = EmbeddingNet(in_channels=train_set.channels, config=config).to(torch_device)
    criterion = BatchHardTripletLoss(margin=config.triplet_margin, soft=config.soft_margin)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)

    result = EmbedderTrainingResult(checkpoint=output_path)
    best_auc = -1.0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_stats: list[dict] = []
        for batch_images, batch_targets in loader:
            batch_images = batch_images.to(torch_device)
            batch_targets = batch_targets.to(torch_device)

            optimiser.zero_grad(set_to_none=True)
            loss, stats = criterion(model(batch_images), batch_targets)
            if loss.requires_grad:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimiser.step()
            epoch_stats.append(stats)
        scheduler.step()

        validation_embeddings = _embed_all(model, val_set, torch_device)
        similarities, is_same = build_pairs(validation_embeddings, val_labels, seed=seed)
        verification = evaluate_verification(similarities, is_same)
        spread = embedding_spread(validation_embeddings)

        row = {
            "epoch": epoch,
            "train_loss": float(np.mean([s["loss"] for s in epoch_stats])),
            "active_triplets": float(np.mean([s["active"] for s in epoch_stats])),
            "mean_positive_distance": float(np.mean([s["pos"] for s in epoch_stats])),
            "mean_negative_distance": float(np.mean([s["neg"] for s in epoch_stats])),
            "embedding_spread": spread,
            "val_roc_auc": verification.roc_auc,
            "val_eer": verification.equal_error_rate,
            "val_best_threshold": verification.best_threshold,
            "learning_rate": scheduler.get_last_lr()[0],
        }
        result.history.append(row)

        if verbose:
            print(
                f"epoch {epoch:3d}/{epochs} | loss {row['train_loss']:.4f} "
                f"| active {row['active_triplets']:.2f} "
                f"| spread {spread:.3f} "
                f"| val AUC {row['val_roc_auc']:.4f} | EER {row['val_eer']:.4f}"
            )

        if verification.roc_auc > best_auc:
            best_auc = verification.roc_auc
            model.save(output_path, accept_threshold=verification.best_threshold)

    result.final_metrics = {
        "best_val_roc_auc": best_auc,
        "recommended_accept_threshold": result.best_epoch().get("val_best_threshold", 0.6),
        "final_embedding_spread": result.history[-1]["embedding_spread"] if result.history else 0.0,
        "epochs": epochs,
        "train_identities": train_set.num_classes,
        "val_identities": val_set.num_classes,
        "detect_first": bool(detect_first),
        "source": str(Path(source)),
    }
    if verbose:
        print(f"saved best checkpoint (AUC={best_auc:.4f}) -> {output_path}")

    return result

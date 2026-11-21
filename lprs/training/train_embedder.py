"""Train the FaceNet-style embedding network with batch-hard triplet loss.

Validation uses **identity-disjoint** data: the validation plates are never
seen during training. That is deliberate. Metric learning is only worth the
trouble if the embedding space generalises to identities the model has never
met -- measuring on seen identities would flatter it enormously and tell you
nothing about deployment.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..config import ARTIFACT_DIR, EmbedderConfig, PlateGeometry, PreprocessConfig
from ..data.datasets import ImageLabelDataset, PKSampler
from ..data.synth_plates import generate_identity_dataset
from ..embeddings.cnn import EmbeddingNet
from ..evaluation.metrics import build_pairs, evaluate_verification
from ..preprocess import Preprocessor
from .triplet import BatchHardTripletLoss

__all__ = ["EmbedderTrainingResult", "train_embedder", "embedding_spread"]


def embedding_spread(embeddings: np.ndarray) -> float:
    """Mean distance of embeddings from their centroid.

    A collapse detector. Triplet loss can minimise itself by squashing the
    whole space into a point, which leaves ranking intact but makes every
    cosine similarity ~1.0 and any absolute threshold meaningless. Healthy
    unit-norm embeddings spread around 0.7-1.0; below ~0.3 is collapse.
    """
    if len(embeddings) == 0:
        return 0.0
    centroid = np.asarray(embeddings, dtype=np.float32).mean(axis=0)
    return float(np.linalg.norm(embeddings - centroid, axis=1).mean())


@dataclass
class EmbedderTrainingResult:
    """Everything the caller needs to report on or plot a training run."""

    checkpoint: Path
    history: list[dict] = field(default_factory=list)
    final_metrics: dict = field(default_factory=dict)

    def best_epoch(self) -> dict:
        if not self.history:
            return {}
        return max(self.history, key=lambda row: row.get("val_roc_auc", 0.0))


def _split_by_identity(
    images: list[np.ndarray],
    labels: list[str],
    validation_fraction: float,
    seed: int,
) -> tuple[tuple[list, list], tuple[list, list]]:
    """Partition on *identity*, never on individual samples."""
    unique = sorted(set(labels))
    rng = random.Random(seed)
    rng.shuffle(unique)

    cut = max(2, int(len(unique) * validation_fraction))
    validation_ids = set(unique[:cut])

    train: tuple[list, list] = ([], [])
    validation: tuple[list, list] = ([], [])
    for image, label in zip(images, labels):
        bucket = validation if label in validation_ids else train
        bucket[0].append(image)
        bucket[1].append(label)
    return train, validation


@torch.no_grad()
def _embed_all(
    model: EmbeddingNet,
    dataset: ImageLabelDataset,
    device: torch.device,
    batch_size: int = 128,
) -> np.ndarray:
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    chunks = [model(images.to(device)).cpu().numpy() for images, _ in loader]
    return np.vstack(chunks) if chunks else np.zeros((0, model.config.embedding_dim))


def train_embedder(
    num_identities: int = 140,
    samples_per_identity: int = 10,
    epochs: int | None = None,
    severity: float = 0.6,
    identities_per_batch: int = 12,
    samples_per_batch_identity: int = 4,
    validation_fraction: float = 0.25,
    device: str = "cpu",
    seed: int = 1337,
    config: EmbedderConfig | None = None,
    output_path: Path | None = None,
    verbose: bool = True,
) -> EmbedderTrainingResult:
    """Generate data, train the embedder, and checkpoint the best model."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    config = config or EmbedderConfig()
    epochs = epochs or config.epochs
    geometry = PlateGeometry()
    output_path = Path(output_path or ARTIFACT_DIR / "embedder_plate.pt")
    torch_device = torch.device(device)

    images, labels = generate_identity_dataset(
        num_identities=num_identities,
        samples_per_identity=samples_per_identity,
        severity=severity,
        seed=seed,
    )
    (train_images, train_labels), (val_images, val_labels) = _split_by_identity(
        images, labels, validation_fraction, seed
    )

    preprocessor = Preprocessor(geometry.size, PreprocessConfig())
    train_set = ImageLabelDataset(train_images, train_labels, preprocessor)
    val_set = ImageLabelDataset(val_images, val_labels, preprocessor)

    if verbose:
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
    criterion = BatchHardTripletLoss(
        margin=config.triplet_margin, soft=config.soft_margin
    )
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
                f"| d+ {row['mean_positive_distance']:.3f} "
                f"| d- {row['mean_negative_distance']:.3f} "
                f"| spread {spread:.3f} "
                f"| val AUC {row['val_roc_auc']:.4f} | EER {row['val_eer']:.4f}"
            )

        if verification.roc_auc > best_auc:
            best_auc = verification.roc_auc
            # Persist the operating point alongside the weights it was
            # measured for, so the two can never drift apart.
            model.save(output_path, accept_threshold=verification.best_threshold)

    result.final_metrics = {
        "best_val_roc_auc": best_auc,
        "recommended_accept_threshold": result.best_epoch().get("val_best_threshold", 0.6),
        "final_embedding_spread": result.history[-1]["embedding_spread"] if result.history else 0.0,
        "epochs": epochs,
        "train_identities": train_set.num_classes,
        "val_identities": val_set.num_classes,
    }
    if verbose:
        print(f"saved best checkpoint (AUC={best_auc:.4f}) -> {output_path}")
    return result

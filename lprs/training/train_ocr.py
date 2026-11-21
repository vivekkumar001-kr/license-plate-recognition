"""Train the CRNN + BiLSTM + CTC plate transcriber.

Unlike the embedder, the OCR head is a *closed-vocabulary-free* sequence
model: it learns glyph shapes, not identities, so train and validation may
share no plate strings and still both be in-distribution. Validation
therefore reports character accuracy and exact-match rate on unseen strings.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..config import ARTIFACT_DIR, NUM_CLASSES, OCRConfig, PlateGeometry, PreprocessConfig
from ..data.datasets import CTCPlateDataset, collate_ctc
from ..data.synth_plates import generate_identity_dataset
from ..evaluation.metrics import evaluate_transcription
from ..ocr.codec import BLANK_INDEX, greedy_decode
from ..ocr.crnn import CRNN
from ..preprocess import Preprocessor

__all__ = ["OCRTrainingResult", "train_ocr"]


@dataclass
class OCRTrainingResult:
    checkpoint: Path
    history: list[dict] = field(default_factory=list)
    final_metrics: dict = field(default_factory=dict)

    def best_epoch(self) -> dict:
        if not self.history:
            return {}
        return max(self.history, key=lambda row: row.get("val_character_accuracy", 0.0))


@torch.no_grad()
def _validate(
    model: CRNN,
    loader: DataLoader,
    texts: list[str],
    device: torch.device,
) -> tuple[list[str], list[str]]:
    model.eval()
    predictions: list[str] = []
    for images, _, _ in loader:
        log_probs = model(images.to(device)).cpu().numpy()
        for index in range(log_probs.shape[1]):
            text, _ = greedy_decode(log_probs[:, index, :])
            predictions.append(text)
    return texts[: len(predictions)], predictions


def train_ocr(
    num_plates: int = 900,
    views_per_plate: int = 2,
    epochs: int | None = None,
    severity: float = 0.55,
    validation_fraction: float = 0.15,
    device: str = "cpu",
    seed: int = 1337,
    config: OCRConfig | None = None,
    output_path: Path | None = None,
    verbose: bool = True,
) -> OCRTrainingResult:
    """Generate plates, train the CRNN, checkpoint the best character accuracy."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    config = config or OCRConfig()
    epochs = epochs or config.epochs
    geometry = PlateGeometry()
    output_path = Path(output_path or ARTIFACT_DIR / "ocr_crnn.pt")
    torch_device = torch.device(device)

    images, texts = generate_identity_dataset(
        num_identities=num_plates,
        samples_per_identity=views_per_plate,
        severity=severity,
        seed=seed,
    )

    order = list(range(len(images)))
    random.Random(seed).shuffle(order)
    cut = int(len(order) * (1.0 - validation_fraction))
    train_index, val_index = order[:cut], order[cut:]

    preprocessor = Preprocessor(geometry.size, PreprocessConfig())
    train_set = CTCPlateDataset(
        [images[i] for i in train_index], [texts[i] for i in train_index], preprocessor
    )
    val_texts = [texts[i] for i in val_index]
    val_set = CTCPlateDataset(
        [images[i] for i in val_index], val_texts, preprocessor
    )

    if verbose:
        print(f"train: {len(train_set)} crops | val: {len(val_set)} crops")

    train_loader = DataLoader(
        train_set, batch_size=config.batch_size, shuffle=True, collate_fn=collate_ctc
    )
    val_loader = DataLoader(
        val_set, batch_size=config.batch_size, shuffle=False, collate_fn=collate_ctc
    )

    model = CRNN(in_channels=1, num_classes=NUM_CLASSES, config=config, geometry=geometry)
    model = model.to(torch_device)
    criterion = torch.nn.CTCLoss(blank=BLANK_INDEX, zero_infinity=True)
    optimiser = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)

    result = OCRTrainingResult(checkpoint=output_path)
    best_accuracy = -1.0

    for epoch in range(1, epochs + 1):
        model.train()
        losses: list[float] = []
        for batch_images, targets, target_lengths in train_loader:
            batch_images = batch_images.to(torch_device)
            log_probs = model(batch_images)

            # CTC needs the input length of every sequence in the batch; our
            # trunk emits a fixed T, so they are all identical.
            input_lengths = torch.full(
                size=(batch_images.size(0),),
                fill_value=log_probs.size(0),
                dtype=torch.long,
            )

            optimiser.zero_grad(set_to_none=True)
            loss = criterion(
                log_probs, targets.to(torch_device), input_lengths, target_lengths
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimiser.step()
            losses.append(float(loss.detach()))
        scheduler.step()

        truths, predictions = _validate(model, val_loader, val_texts, torch_device)
        metrics = evaluate_transcription(truths, predictions)

        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)) if losses else 0.0,
            "val_character_accuracy": metrics.character_accuracy,
            "val_exact_match": metrics.exact_match,
            "val_edit_distance": metrics.mean_edit_distance,
            "learning_rate": scheduler.get_last_lr()[0],
        }
        result.history.append(row)

        if verbose:
            print(
                f"epoch {epoch:3d}/{epochs} | CTC loss {row['train_loss']:.4f} "
                f"| char acc {row['val_character_accuracy']:.4f} "
                f"| exact {row['val_exact_match']:.4f} "
                f"| edit {row['val_edit_distance']:.2f}"
            )

        if metrics.character_accuracy > best_accuracy:
            best_accuracy = metrics.character_accuracy
            model.save(output_path)

    result.final_metrics = {
        "best_val_character_accuracy": best_accuracy,
        "best_val_exact_match": result.best_epoch().get("val_exact_match", 0.0),
        "epochs": epochs,
        "train_samples": len(train_set),
        "val_samples": len(val_set),
    }
    if verbose:
        print(f"saved best checkpoint (char acc={best_accuracy:.4f}) -> {output_path}")
    return result

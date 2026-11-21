"""CRNN: convolutional feature extractor + BiLSTM sequence model + CTC head.

Why a sequence model at all? A plate is not a single class -- it is a
variable-length *ordered* string, and the characters are not independently
segmentable once the image is blurred or skewed. The convolutional trunk
collapses the image to a horizontal strip of feature columns, the
bidirectional LSTM gives every column context from both directions (so the
network can use 'this looks like a digit block' reasoning), and CTC aligns
the resulting frame predictions to the label without needing per-character
bounding boxes.

Height is reduced to exactly 1 while width is preserved, producing T feature
columns -- the sequence that the recurrent layer consumes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from ..config import NUM_CLASSES, OCRConfig, PlateGeometry
from .codec import greedy_decode

__all__ = ["CRNN", "PlateTranscriber"]


def _conv_bn(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class CRNN(nn.Module):
    """Convolutional Recurrent Neural Network with a CTC output layer."""

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = NUM_CLASSES,
        config: OCRConfig | None = None,
        geometry: PlateGeometry | None = None,
    ) -> None:
        super().__init__()
        self.config = config or OCRConfig()
        self.geometry = geometry or PlateGeometry()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # Pooling is asymmetric on purpose: squash height to 1, keep width.
        self.trunk = nn.Sequential(
            _conv_bn(in_channels, 32),
            nn.MaxPool2d(2, 2),           # H/2,  W/2
            _conv_bn(32, 64),
            nn.MaxPool2d(2, 2),           # H/4,  W/4
            _conv_bn(64, 128),
            _conv_bn(128, 128),
            nn.MaxPool2d((2, 1), (2, 1)),  # H/8,  W/4
            _conv_bn(128, 256),
            nn.MaxPool2d((2, 1), (2, 1)),  # H/16, W/4
            nn.Dropout2d(self.config.dropout),
        )

        self.column_pool = nn.AdaptiveAvgPool2d((1, None))
        self.rnn = nn.LSTM(
            input_size=256,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.lstm_layers,
            bidirectional=True,
            batch_first=False,
            dropout=self.config.dropout if self.config.lstm_layers > 1 else 0.0,
        )
        self.classifier = nn.Linear(self.config.hidden_size * 2, num_classes)

    @property
    def time_steps(self) -> int:
        """Number of CTC frames produced for a canonical-width input."""
        return self.geometry.width // 4

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return ``(T, N, num_classes)`` log-probabilities for CTC."""
        features = self.column_pool(self.trunk(images))  # (N, C, 1, W')
        features = features.squeeze(2).permute(2, 0, 1)  # (W', N, C)
        sequence, _ = self.rnn(features)
        return nn.functional.log_softmax(self.classifier(sequence), dim=2)

    # -- persistence -------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "in_channels": self.in_channels,
                "num_classes": self.num_classes,
                "config": vars(self.config),
                "geometry": vars(self.geometry),
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, map_location: str = "cpu") -> "CRNN":
        payload = torch.load(path, map_location=map_location, weights_only=False)
        model = cls(
            in_channels=payload["in_channels"],
            num_classes=payload["num_classes"],
            config=OCRConfig(**payload["config"]),
            geometry=PlateGeometry(**payload["geometry"]),
        )
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return model


class PlateTranscriber:
    """Inference adapter: preprocessed crops in, plate strings out."""

    name = "crnn-ctc"

    def __init__(self, model: CRNN, device: str = "cpu", batch_size: int = 32) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.batch_size = batch_size

    @classmethod
    def from_checkpoint(cls, path: str | Path, device: str = "cpu") -> "PlateTranscriber":
        return cls(CRNN.load(path, map_location=device), device=device)

    @torch.no_grad()
    def transcribe(self, batch: np.ndarray) -> list[tuple[str, float]]:
        """Return ``(text, confidence)`` for each crop in the batch."""
        array = np.asarray(batch, dtype=np.float32)
        if len(array) == 0:
            return []
        if array.ndim == 3:
            array = array[:, :, :, None]

        tensor = torch.from_numpy(
            np.ascontiguousarray(array.transpose(0, 3, 1, 2))
        ).to(self.device)

        results: list[tuple[str, float]] = []
        for start in range(0, len(tensor), self.batch_size):
            log_probs = self.model(tensor[start : start + self.batch_size])
            # (T, N, C) -> iterate over the batch dimension.
            for index in range(log_probs.shape[1]):
                results.append(greedy_decode(log_probs[:, index, :].cpu().numpy()))
        return results

    def transcribe_one(self, image: np.ndarray) -> tuple[str, float]:
        return self.transcribe(image[None, ...])[0]

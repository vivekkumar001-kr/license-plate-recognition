"""FaceNet-style CNN embedding network and its inference wrapper.

The architecture follows the FaceNet recipe rather than its exact topology:
a convolutional trunk, global pooling, a linear bottleneck, and an explicit
L2 normalisation so that every embedding lands on the unit hypersphere.
Training uses a triplet objective (see ``lprs.training.train_embedder``),
which optimises *relative* distances and therefore generalises to identities
never seen during training -- the whole point of an open-set recogniser.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from ..config import EmbedderConfig
from .base import Embedder, l2_normalise

__all__ = ["EmbeddingNet", "TorchEmbedder", "to_nchw"]


def to_nchw(batch: np.ndarray) -> torch.Tensor:
    """Convert an ``(N, H, W[, C])`` float array to an NCHW float tensor."""
    array = np.asarray(batch, dtype=np.float32)
    if array.ndim == 3:
        array = array[:, :, :, None]
    if array.ndim != 4:
        raise ValueError(f"Expected a 3D or 4D batch, received shape {array.shape}")
    return torch.from_numpy(np.ascontiguousarray(array.transpose(0, 3, 1, 2)))


class _ConvBlock(nn.Module):
    """Conv -> BatchNorm -> ReLU -> Conv -> BatchNorm -> ReLU -> MaxPool."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.body(inputs)


class EmbeddingNet(nn.Module):
    """Convolutional trunk producing unit-norm embeddings."""

    def __init__(self, in_channels: int = 1, config: EmbedderConfig | None = None) -> None:
        super().__init__()
        self.config = config or EmbedderConfig()
        self.in_channels = in_channels

        blocks: list[nn.Module] = []
        previous = in_channels
        for channels in self.config.channels:
            blocks.append(_ConvBlock(previous, channels))
            previous = channels
        self.trunk = nn.Sequential(*blocks)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(self.config.dropout),
            nn.Linear(previous, self.config.embedding_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.head(self.pool(self.trunk(inputs)))
        # Unit-norm output is part of the model contract, not an afterthought.
        return nn.functional.normalize(features, p=2.0, dim=1)

    # -- persistence -------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "in_channels": self.in_channels,
                "config": vars(self.config),
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, map_location: str = "cpu") -> "EmbeddingNet":
        payload = torch.load(path, map_location=map_location, weights_only=False)
        config = EmbedderConfig(**payload["config"])
        model = cls(in_channels=payload["in_channels"], config=config)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return model


class TorchEmbedder(Embedder):
    """Inference-time adapter around :class:`EmbeddingNet`."""

    name = "cnn"

    def __init__(
        self,
        model: EmbeddingNet,
        device: str = "cpu",
        batch_size: int = 64,
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.batch_size = batch_size

    @classmethod
    def from_checkpoint(cls, path: str | Path, device: str = "cpu") -> "TorchEmbedder":
        return cls(EmbeddingNet.load(path, map_location=device), device=device)

    @property
    def embedding_dim(self) -> int:
        return self.model.config.embedding_dim

    @torch.no_grad()
    def embed(self, batch: np.ndarray) -> np.ndarray:
        if len(batch) == 0:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        tensor = to_nchw(batch).to(self.device)
        chunks = [
            self.model(tensor[start : start + self.batch_size])
            for start in range(0, len(tensor), self.batch_size)
        ]
        vectors = torch.cat(chunks).cpu().numpy()
        # Belt and braces: re-normalise after the float round-trip.
        return l2_normalise(vectors)

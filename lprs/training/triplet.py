"""Triplet loss with online batch-hard mining.

Sampling triplets offline is hopeless: the overwhelming majority are already
separated by the margin and contribute exactly zero gradient, so training
stalls. Batch-hard mining instead picks, *for every anchor in the batch*, the
furthest positive and the nearest negative. Those are the only triplets that
teach the model anything.

Because embeddings are unit-norm, squared Euclidean distance and cosine
similarity are equivalent up to a constant, so the distance matrix below is
computed with a single matrix multiply.
"""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["pairwise_distances", "BatchHardTripletLoss"]


def pairwise_distances(embeddings: torch.Tensor, epsilon: float = 1e-12) -> torch.Tensor:
    """Euclidean distance matrix for a batch of row vectors."""
    squared_norms = (embeddings * embeddings).sum(dim=1, keepdim=True)
    squared = squared_norms + squared_norms.t() - 2.0 * (embeddings @ embeddings.t())
    squared = squared.clamp(min=0.0)
    # sqrt has infinite gradient at zero; nudge the diagonal off it.
    distances = torch.sqrt(squared + epsilon)
    return distances - torch.diag_embed(torch.diagonal(distances))


class BatchHardTripletLoss(nn.Module):
    """Hardest-positive / hardest-negative triplet loss with a soft margin."""

    def __init__(self, margin: float = 0.3, soft: bool = False) -> None:
        super().__init__()
        self.margin = margin
        self.soft = soft

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        distances = pairwise_distances(embeddings)
        same_identity = labels[:, None] == labels[None, :]
        identity_matrix = torch.eye(
            len(labels), dtype=torch.bool, device=labels.device
        )
        positive_mask = same_identity & ~identity_matrix
        negative_mask = ~same_identity

        # Anchors with no positive (or no negative) in the batch are skipped.
        valid = positive_mask.any(dim=1) & negative_mask.any(dim=1)
        if not valid.any():
            zero = embeddings.sum() * 0.0
            return zero, {"loss": 0.0, "active": 0.0, "pos": 0.0, "neg": 0.0}

        hardest_positive = (distances * positive_mask).max(dim=1).values
        # Mask negatives with +inf so the min ignores same-identity pairs.
        masked_negatives = distances.masked_fill(~negative_mask, float("inf"))
        hardest_negative = masked_negatives.min(dim=1).values

        hardest_positive = hardest_positive[valid]
        hardest_negative = hardest_negative[valid]

        if self.soft:
            losses = nn.functional.softplus(hardest_positive - hardest_negative)
        else:
            losses = nn.functional.relu(hardest_positive - hardest_negative + self.margin)

        statistics = {
            "loss": float(losses.mean().detach()),
            "active": float((losses > 1e-6).float().mean().detach()),
            "pos": float(hardest_positive.mean().detach()),
            "neg": float(hardest_negative.mean().detach()),
        }
        return losses.mean(), statistics

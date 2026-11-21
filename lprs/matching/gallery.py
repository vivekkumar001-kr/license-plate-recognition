"""Similarity matching against an enrolled gallery.

This is the recognition step proper. Enrolment stores one or more unit-norm
embeddings per identity; a probe is identified by cosine similarity. Because
every embedding is L2-normalised, cosine similarity is a plain dot product,
so the whole gallery is scored with a single matrix multiply.

Two scoring strategies are supported:

``nearest``
    Similarity to the single closest enrolled sample. Sensitive to outliers
    but handles multi-modal identities (day plate vs night plate).
``centroid``
    Similarity to the per-identity mean vector (a "prototype"). Smoother and
    faster, and usually the better default once you have 5+ samples each.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal

import numpy as np

from ..config import MatchConfig
from ..embeddings.base import l2_normalise

__all__ = ["MatchResult", "Gallery", "UNKNOWN_LABEL"]

UNKNOWN_LABEL = "UNKNOWN"

ScoringStrategy = Literal["nearest", "centroid"]


@dataclass(frozen=True)
class MatchResult:
    """One candidate identity returned by the matcher."""

    label: str
    similarity: float
    rank: int
    accepted: bool
    strategy: str = "cosine"
    metadata: dict = field(default_factory=dict)

    @property
    def distance(self) -> float:
        """Cosine distance, for readers who prefer smaller-is-better."""
        return 1.0 - self.similarity

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "similarity": round(self.similarity, 4),
            "distance": round(self.distance, 4),
            "rank": self.rank,
            "accepted": self.accepted,
            "strategy": self.strategy,
            "metadata": self.metadata,
        }


class Gallery:
    """An enrolment database of unit-norm embeddings keyed by label."""

    def __init__(
        self,
        config: MatchConfig | None = None,
        strategy: ScoringStrategy = "centroid",
    ) -> None:
        self.config = config or MatchConfig()
        self.strategy: ScoringStrategy = strategy
        self._vectors: list[np.ndarray] = []
        self._labels: list[str] = []
        self._metadata: list[dict] = []
        self._cache: tuple[np.ndarray, list[str]] | None = None

    # -- introspection -----------------------------------------------------

    def __len__(self) -> int:
        return len(self._labels)

    @property
    def labels(self) -> list[str]:
        """Unique identities, in insertion order."""
        return list(dict.fromkeys(self._labels))

    @property
    def is_empty(self) -> bool:
        return not self._labels

    def samples_per_label(self) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        for label in self._labels:
            counts[label] += 1
        return dict(counts)

    # -- enrolment ---------------------------------------------------------

    def enrol(self, label: str, embedding: np.ndarray, metadata: dict | None = None) -> None:
        """Add a single embedding for ``label``."""
        vector = l2_normalise(np.asarray(embedding, dtype=np.float32))[0]
        if self._vectors and vector.shape != self._vectors[0].shape:
            raise ValueError(
                f"Embedding dimension {vector.shape} does not match gallery "
                f"dimension {self._vectors[0].shape}"
            )
        self._vectors.append(vector)
        self._labels.append(label)
        self._metadata.append(metadata or {})
        self._cache = None

    def enrol_many(
        self,
        labels: Iterable[str],
        embeddings: np.ndarray,
        metadata: Iterable[dict] | None = None,
    ) -> None:
        """Bulk enrolment; far cheaper than looping over :meth:`enrol`."""
        labels = list(labels)
        matrix = l2_normalise(np.asarray(embeddings, dtype=np.float32))
        if len(labels) != len(matrix):
            raise ValueError(
                f"Got {len(labels)} labels for {len(matrix)} embeddings"
            )
        metadata_list = list(metadata) if metadata is not None else [{}] * len(labels)
        self._vectors.extend(matrix)
        self._labels.extend(labels)
        self._metadata.extend(metadata_list)
        self._cache = None

    # -- scoring -----------------------------------------------------------

    def _scoring_matrix(self) -> tuple[np.ndarray, list[str]]:
        """Return the (M, D) matrix actually compared against, plus its labels."""
        if self._cache is not None:
            return self._cache

        if self.is_empty:
            raise ValueError("Cannot score against an empty gallery")

        stacked = np.vstack(self._vectors)
        if self.strategy == "nearest":
            result = (stacked, list(self._labels))
        else:
            unique_labels = self.labels
            centroids = np.vstack(
                [
                    stacked[[i for i, lab in enumerate(self._labels) if lab == label]].mean(axis=0)
                    for label in unique_labels
                ]
            )
            result = (l2_normalise(centroids), unique_labels)

        self._cache = result
        return result

    def _metadata_for(self, label: str) -> dict:
        for stored_label, meta in zip(self._labels, self._metadata):
            if stored_label == label and meta:
                return meta
        return {}

    def match(self, query: np.ndarray, top_k: int | None = None) -> list[MatchResult]:
        """Return the ``top_k`` most similar identities, best first."""
        matrix, labels = self._scoring_matrix()
        probe = l2_normalise(np.asarray(query, dtype=np.float32))[0]
        similarities = matrix @ probe

        # Collapse duplicates so 'nearest' reports each identity only once.
        best_per_label: dict[str, float] = {}
        for label, similarity in zip(labels, similarities):
            if similarity > best_per_label.get(label, -np.inf):
                best_per_label[label] = float(similarity)

        ranked = sorted(best_per_label.items(), key=lambda item: item[1], reverse=True)
        limit = top_k or self.config.top_k
        return [
            MatchResult(
                label=label,
                similarity=similarity,
                rank=rank,
                accepted=similarity >= self.config.accept_threshold,
                strategy=f"cosine/{self.strategy}",
                metadata=self._metadata_for(label),
            )
            for rank, (label, similarity) in enumerate(ranked[:limit], start=1)
        ]

    def identify(self, query: np.ndarray) -> MatchResult:
        """Return the single best match, or an explicit ``UNKNOWN`` verdict.

        Refusing to answer is a feature: an open-set recogniser that always
        names its nearest neighbour will confidently accuse strangers.
        """
        candidates = self.match(query, top_k=1)
        if candidates and candidates[0].accepted:
            return candidates[0]

        similarity = candidates[0].similarity if candidates else 0.0
        return MatchResult(
            label=UNKNOWN_LABEL,
            similarity=similarity,
            rank=1,
            accepted=False,
            strategy=f"cosine/{self.strategy}",
            metadata={"reason": "below_accept_threshold"},
        )

    # -- persistence -------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            vectors=np.vstack(self._vectors) if self._vectors else np.empty((0, 0)),
            labels=np.asarray(self._labels, dtype=object),
            metadata=np.asarray([json.dumps(m) for m in self._metadata], dtype=object),
            strategy=self.strategy,
            accept_threshold=self.config.accept_threshold,
        )

    @classmethod
    def load(cls, path: str | Path) -> "Gallery":
        payload = np.load(path, allow_pickle=True)
        config = MatchConfig(accept_threshold=float(payload["accept_threshold"]))
        gallery = cls(config=config, strategy=str(payload["strategy"]))
        labels = [str(label) for label in payload["labels"]]
        if labels:
            gallery.enrol_many(
                labels,
                payload["vectors"],
                [json.loads(raw) for raw in payload["metadata"]],
            )
        return gallery

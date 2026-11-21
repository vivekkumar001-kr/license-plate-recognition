"""Torch dataset and sampler plumbing.

The interesting piece here is :class:`PKSampler`. Triplet loss is useless if a
batch contains no positive pairs, so batches are built as *P identities x K
samples* rather than uniformly at random. That guarantees every anchor has
K-1 positives available for online hard mining.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from ..ocr.codec import encode_text
from ..preprocess import Preprocessor

__all__ = ["ImageLabelDataset", "CTCPlateDataset", "PKSampler", "collate_ctc"]


class ImageLabelDataset(Dataset):
    """Preprocessed crops paired with integer identity indices.

    Preprocessing happens lazily in ``__getitem__`` so that augmented copies
    of a 20k-image dataset never all live in memory at once.
    """

    def __init__(
        self,
        images: list[np.ndarray],
        labels: list[str],
        preprocessor: Preprocessor,
    ) -> None:
        if len(images) != len(labels):
            raise ValueError(f"Got {len(images)} images but {len(labels)} labels")

        self.images = images
        self.labels = labels
        self.preprocessor = preprocessor
        self.classes = sorted(set(labels))
        self.class_to_index = {label: index for index, label in enumerate(self.classes)}
        self.targets = [self.class_to_index[label] for label in labels]

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        processed = self.preprocessor.run(self.images[index])
        if processed.ndim == 2:
            processed = processed[:, :, None]
        tensor = torch.from_numpy(
            np.ascontiguousarray(processed.transpose(2, 0, 1), dtype=np.float32)
        )
        return tensor, self.targets[index]

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    @property
    def channels(self) -> int:
        sample = self.preprocessor.run(self.images[0])
        return 1 if sample.ndim == 2 else sample.shape[2]


class CTCPlateDataset(Dataset):
    """Crops paired with their encoded character sequences, for CTC training."""

    def __init__(
        self,
        images: list[np.ndarray],
        texts: list[str],
        preprocessor: Preprocessor,
    ) -> None:
        if len(images) != len(texts):
            raise ValueError(f"Got {len(images)} images but {len(texts)} texts")
        self.images = images
        self.texts = texts
        self.preprocessor = preprocessor

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        processed = self.preprocessor.run(self.images[index])
        if processed.ndim == 2:
            processed = processed[:, :, None]
        image = torch.from_numpy(
            np.ascontiguousarray(processed.transpose(2, 0, 1), dtype=np.float32)
        )
        target = torch.tensor(encode_text(self.texts[index]), dtype=torch.long)
        return image, target


class PKSampler(Sampler[int]):
    """Yields batches of ``P`` identities with ``K`` samples each.

    Identities with fewer than K samples are sampled with replacement rather
    than dropped, so small galleries still train.
    """

    def __init__(
        self,
        targets: list[int],
        identities_per_batch: int,
        samples_per_identity: int,
        batches_per_epoch: int | None = None,
        seed: int = 1337,
    ) -> None:
        self.samples_per_identity = samples_per_identity
        self.index_by_target: dict[int, list[int]] = defaultdict(list)
        for index, target in enumerate(targets):
            self.index_by_target[target].append(index)

        self.targets = sorted(self.index_by_target)
        self.identities_per_batch = min(identities_per_batch, len(self.targets))
        if self.identities_per_batch < 2:
            raise ValueError("Triplet training needs at least 2 identities per batch")

        self.batch_size = self.identities_per_batch * samples_per_identity
        self.batches_per_epoch = batches_per_epoch or max(1, len(targets) // self.batch_size)
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.batches_per_epoch * self.batch_size

    def __iter__(self):
        for _ in range(self.batches_per_epoch):
            chosen = self.rng.choice(
                self.targets, size=self.identities_per_batch, replace=False
            )
            for target in chosen:
                pool = self.index_by_target[int(target)]
                replace = len(pool) < self.samples_per_identity
                picks = self.rng.choice(pool, size=self.samples_per_identity, replace=replace)
                yield from (int(pick) for pick in picks)


def collate_ctc(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate (image, encoded_label) pairs into CTC's flattened format.

    ``torch.nn.CTCLoss`` wants all targets concatenated into one 1-D tensor
    plus a vector of lengths, not a padded matrix.
    """
    images = torch.stack([item[0] for item in batch])
    lengths = torch.tensor([len(item[1]) for item in batch], dtype=torch.long)
    targets = torch.cat([item[1] for item in batch])
    return images, targets, lengths

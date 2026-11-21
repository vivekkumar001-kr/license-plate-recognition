"""End-to-end recognition pipeline.

This module contains the one class the application layer actually talks to.
It is deliberately domain-agnostic: plates and faces differ only in which
detector, geometry, aligner and embedder are injected, so there is exactly
one implementation of the detect -> preprocess -> embed -> match sequence
rather than two near-identical copies.

    raw image
        |-- detector      locate regions of interest
        |-- aligner       normalise pose
        |-- preprocessor  grayscale / CLAHE / denoise / deskew / resize
        |-- embedder      CNN -> unit-norm vector
        |-- gallery       cosine similarity -> identity or UNKNOWN
        `-- transcriber   (plates only) CRNN+BiLSTM+CTC -> text
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .align import Aligner, EyeAligner, NullAligner
from .config import (
    ARTIFACT_DIR,
    FaceGeometry,
    MatchConfig,
    PlateGeometry,
    PreprocessConfig,
)
from .detectors import Detection, Detector, build_detector
from .embeddings import Embedder, build_embedder
from .matching import Gallery, MatchResult, text_similarity
from .ocr import PlateTranscriber
from .preprocess import Preprocessor

__all__ = ["RecognitionResult", "RecognitionPipeline", "build_pipeline"]


@dataclass
class RecognitionResult:
    """Everything known about one detected region."""

    detection: Detection
    embedding: np.ndarray
    candidates: list[MatchResult] = field(default_factory=list)
    text: str | None = None
    text_confidence: float | None = None

    @property
    def best(self) -> MatchResult | None:
        return self.candidates[0] if self.candidates else None

    @property
    def label(self) -> str:
        best = self.best
        return best.label if best else "UNKNOWN"

    def as_dict(self) -> dict:
        return {
            "box": list(self.detection.box),
            "detector": self.detection.source,
            "detection_score": round(float(self.detection.score), 4),
            "label": self.label,
            "text": self.text,
            "text_confidence": (
                round(self.text_confidence, 4) if self.text_confidence is not None else None
            ),
            "candidates": [candidate.as_dict() for candidate in self.candidates],
        }


class RecognitionPipeline:
    """Composes a detector, preprocessor, embedder and gallery."""

    def __init__(
        self,
        detector: Detector,
        preprocessor: Preprocessor,
        embedder: Embedder,
        gallery: Gallery | None = None,
        aligner: Aligner | None = None,
        transcriber: PlateTranscriber | None = None,
        crop_margin: float = 0.04,
        domain: str = "plate",
    ) -> None:
        self.detector = detector
        self.preprocessor = preprocessor
        self.embedder = embedder
        self.gallery = gallery if gallery is not None else Gallery()
        self.aligner = aligner or NullAligner()
        self.transcriber = transcriber
        self.crop_margin = crop_margin
        self.domain = domain

    # -- core stages -------------------------------------------------------

    def crops_from(self, image: np.ndarray, max_regions: int = 5) -> list[tuple[Detection, np.ndarray]]:
        """Detect, align and preprocess -- everything short of embedding."""
        pairs: list[tuple[Detection, np.ndarray]] = []
        for detection in self.detector.detect(image)[:max_regions]:
            raw = detection.crop(image, margin=self.crop_margin)
            if raw.size == 0:
                continue
            pairs.append((detection, self.preprocessor.run(self.aligner.align(raw))))
        return pairs

    def embed_image(self, image: np.ndarray) -> np.ndarray:
        """Embed an image that is already a tight crop. Used for enrolment."""
        processed = self.preprocessor.run(self.aligner.align(image))
        return self.embedder.embed_one(processed)

    # -- enrolment ---------------------------------------------------------

    def enrol(
        self,
        label: str,
        image: np.ndarray,
        metadata: dict | None = None,
        detect_first: bool = True,
    ) -> bool:
        """Add an image to the gallery under ``label``.

        Returns False when ``detect_first`` is set and nothing was found,
        so the caller can report the failure instead of silently enrolling
        a photo of someone's bumper.
        """
        if detect_first:
            pairs = self.crops_from(image, max_regions=1)
            if not pairs:
                return False
            processed = pairs[0][1]
        else:
            processed = self.preprocessor.run(self.aligner.align(image))

        self.gallery.enrol(label, self.embedder.embed_one(processed), metadata)
        return True

    def enrol_batch(
        self,
        labels: list[str],
        images: list[np.ndarray],
        detect_first: bool = False,
    ) -> int:
        """Bulk enrolment. Returns how many images were accepted."""
        accepted_labels: list[str] = []
        processed_crops: list[np.ndarray] = []

        for label, image in zip(labels, images):
            if detect_first:
                pairs = self.crops_from(image, max_regions=1)
                if not pairs:
                    continue
                processed_crops.append(pairs[0][1])
            else:
                processed_crops.append(self.preprocessor.run(self.aligner.align(image)))
            accepted_labels.append(label)

        if not accepted_labels:
            return 0
        self.gallery.enrol_many(accepted_labels, self.embedder.embed(np.stack(processed_crops)))
        return len(accepted_labels)

    # -- recognition -------------------------------------------------------

    def recognise(
        self,
        image: np.ndarray,
        max_regions: int = 5,
        top_k: int | None = None,
    ) -> list[RecognitionResult]:
        """Run the full pipeline over one frame."""
        pairs = self.crops_from(image, max_regions=max_regions)
        if not pairs:
            return []

        batch = np.stack([processed for _, processed in pairs])
        embeddings = self.embedder.embed(batch)

        transcriptions: list[tuple[str, float]] = []
        if self.transcriber is not None:
            transcriptions = self.transcriber.transcribe(batch)

        results: list[RecognitionResult] = []
        for index, (detection, _) in enumerate(pairs):
            text, confidence = (
                transcriptions[index] if index < len(transcriptions) else (None, None)
            )
            candidates = (
                [] if self.gallery.is_empty else self.gallery.match(embeddings[index], top_k)
            )
            candidates = self._apply_text_fallback(candidates, text)
            results.append(
                RecognitionResult(
                    detection=detection,
                    embedding=embeddings[index],
                    candidates=candidates,
                    text=text,
                    text_confidence=confidence,
                )
            )
        return results

    def _apply_text_fallback(
        self,
        candidates: list[MatchResult],
        text: str | None,
    ) -> list[MatchResult]:
        """Re-rank candidates using OCR text when the visual match is weak.

        Two independent signals (appearance and transcription) disagreeing is
        information, not noise. When the appearance match is below threshold
        but the OCR string nearly equals an enrolled label, trust the text.
        """
        config = self.gallery.config
        if not config.use_text_fallback or not text or not candidates:
            return candidates
        if candidates[0].accepted:
            return candidates

        scored = [
            (candidate, text_similarity(text, candidate.label)) for candidate in candidates
        ]
        best_candidate, best_similarity = max(scored, key=lambda pair: pair[1])
        if best_similarity < 0.8:
            return candidates

        promoted = MatchResult(
            label=best_candidate.label,
            similarity=max(best_candidate.similarity, best_similarity),
            rank=1,
            accepted=True,
            strategy="text-fallback",
            metadata={**best_candidate.metadata, "ocr_text": text},
        )
        others = [c for c in candidates if c.label != promoted.label]
        return [promoted, *others]


# --------------------------------------------------------------------------
# Factory
# --------------------------------------------------------------------------


def build_pipeline(
    domain: str = "plate",
    embedder_kind: str = "cnn",
    embedder_checkpoint: str | Path | None = None,
    ocr_checkpoint: str | Path | None = None,
    gallery_path: str | Path | None = None,
    match_config: MatchConfig | None = None,
    gallery_strategy: str = "centroid",
    device: str = "cpu",
) -> RecognitionPipeline:
    """Assemble a pipeline for ``domain`` with sensible defaults.

    Falls back from the CNN embedder to HOG when no checkpoint exists, so a
    freshly-cloned repository is usable before anything has been trained.
    """
    if domain == "plate":
        geometry: PlateGeometry | FaceGeometry = PlateGeometry()
        preprocess_config = PreprocessConfig()
        aligner: Aligner = NullAligner()
        default_embedder = ARTIFACT_DIR / "embedder_plate.pt"
    elif domain == "face":
        geometry = FaceGeometry()
        preprocess_config = PreprocessConfig.for_faces()
        aligner = EyeAligner()
        default_embedder = ARTIFACT_DIR / "embedder_face.pt"
    else:
        raise ValueError(f"Unknown domain {domain!r}; expected 'plate' or 'face'")

    checkpoint = Path(embedder_checkpoint) if embedder_checkpoint else default_embedder
    if embedder_kind == "cnn" and not checkpoint.exists():
        print(
            f"[lprs] No embedder checkpoint at {checkpoint}; falling back to the "
            f"HOG baseline. Train one with: lprs train-embedder"
        )
        embedder_kind = "hog"

    embedder = build_embedder(
        embedder_kind,
        geometry.size,
        checkpoint=checkpoint if embedder_kind == "cnn" else None,
        device=device,
    )

    # Threshold precedence, highest first:
    #   1. an explicit MatchConfig  -- the operator chose a risk budget
    #   2. the checkpoint's calibrated value -- measured against these weights
    #   3. whatever was stored in the gallery file
    # Getting this order wrong silently reverted calibration to a stale
    # default, which is precisely the bug this comment exists to prevent.
    calibrated = getattr(embedder, "recommended_threshold", None)
    if match_config is None and calibrated is not None:
        match_config = MatchConfig(accept_threshold=float(calibrated))

    transcriber = None
    if domain == "plate":
        ocr_path = Path(ocr_checkpoint) if ocr_checkpoint else ARTIFACT_DIR / "ocr_crnn.pt"
        if ocr_path.exists():
            transcriber = PlateTranscriber.from_checkpoint(ocr_path, device=device)

    if gallery_path and Path(gallery_path).exists():
        gallery = Gallery.load(gallery_path)
        if match_config is not None:
            gallery.config = match_config
    else:
        gallery = Gallery(match_config or MatchConfig(), strategy=gallery_strategy)

    return RecognitionPipeline(
        detector=build_detector(domain),
        preprocessor=Preprocessor(geometry.size, preprocess_config),
        embedder=embedder,
        gallery=gallery,
        aligner=aligner,
        transcriber=transcriber,
        domain=domain,
    )

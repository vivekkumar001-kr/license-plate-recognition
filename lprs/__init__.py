"""License Plate Recognition System.

OpenCV detection -> image preprocessing -> CNN/FaceNet-style embeddings ->
cosine similarity matching, with an optional CRNN+BiLSTM+CTC transcription head.
"""

from __future__ import annotations

__version__ = "0.1.0"

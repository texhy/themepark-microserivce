"""ArcFace embedding generation via InsightFace's recognition model.

For the ingestion pipeline (Phase 2), embeddings are already produced by
`face_app.get()` inside ScrfdDetector.  This module exposes a standalone
embedder for the query pipeline (Phase 3) where a face is detected by YOLOv8,
aligned externally, and then needs to be embedded via the ArcFace model
independently.
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np

from app.core.model_loader import get_models

logger = logging.getLogger(__name__)


class ArcFaceEmbedder:
    """Generates 512-dim L2-normalized embeddings from aligned 112×112 face crops."""

    def embed(self, aligned_face: np.ndarray) -> np.ndarray:
        """Produce a single 512-dim embedding.

        Args:
            aligned_face: BGR uint8 array, shape (112, 112, 3).

        Returns:
            L2-normalized float32 vector of length 512.
        """
        models = get_models()
        rec_model = models.face_analysis.models.get("recognition")
        if rec_model is None:
            raise RuntimeError("ArcFace recognition model not loaded in InsightFace")

        embedding = rec_model.get_feat(aligned_face)
        embedding = np.asarray(embedding, dtype=np.float32).flatten()
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding

    def embed_batch(self, faces: List[np.ndarray]) -> np.ndarray:
        """Embed multiple aligned faces.

        Args:
            faces: List of BGR uint8 arrays, each (112, 112, 3).

        Returns:
            np.ndarray of shape (N, 512), each row L2-normalized.
        """
        if not faces:
            return np.empty((0, 512), dtype=np.float32)

        embeddings = np.stack([self.embed(f) for f in faces])
        return embeddings

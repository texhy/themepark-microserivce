"""Pipeline 1 — Photographer image ingestion.

POST /ingest/          Single image ingestion
POST /ingest/batch     Multi-image batch ingestion
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import List

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.models.schemas import (
    BatchIngestImageResult,
    BatchIngestResponse,
    IngestFaceResult,
    IngestResponse,
)
from app.services.face_detector import ScrfdDetector
from app.services.faiss_manager import get_faiss_manager
from app.services.image_utils import decode_image_bytes, validate_content_type
from app.utils.malaysia_time import malaysia_today_iso

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ingest", tags=["ingest"])

_detector = ScrfdDetector()


def _process_single_image(
    raw_bytes: bytes,
    park_id: str,
    image_id: str,
    ingest_date: str = "",
) -> tuple[int, List[IngestFaceResult]]:
    """Detect faces, generate embeddings, store in FAISS.

    Returns (faces_detected, list_of_face_results).
    """
    image = decode_image_bytes(raw_bytes)
    detected = _detector.detect(image)

    fm = get_faiss_manager()
    face_results: list[IngestFaceResult] = []

    if detected:
        face_ids = [str(uuid.uuid4()) for _ in detected]
        import numpy as np

        embeddings = np.stack([f.embedding for f in detected])
        image_ids = [image_id] * len(detected)
        fm.add_embeddings_batch(
            park_id, face_ids, embeddings,
            image_ids=image_ids, ingest_date=ingest_date,
        )

        for face, fid in zip(detected, face_ids):
            face_results.append(
                IngestFaceResult(
                    face_id=fid,
                    bbox=face.bbox,
                    confidence=round(face.confidence, 4),
                    embedding_stored=True,
                )
            )

    return len(detected), face_results


# ── Single image ingest ─────────────────────────────────────────────

@router.post("/", response_model=IngestResponse)
async def ingest_image(
    file: UploadFile = File(...),
    park_id: str = Form(...),
    image_id: str = Form(...),
):
    """Ingest a single photographer image: detect → embed → store in FAISS."""
    err = validate_content_type(file.content_type)
    if err:
        raise HTTPException(status_code=400, detail=err)

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file body")

    t0 = time.perf_counter()

    try:
        faces_detected, face_results = _process_single_image(
            raw, park_id, image_id, ingest_date=malaysia_today_iso(),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        logger.exception("Ingest failed for image_id=%s park_id=%s", image_id, park_id)
        raise HTTPException(status_code=500, detail="Internal processing error")

    fm = get_faiss_manager()
    fm.save_index(park_id)

    elapsed_ms = (time.perf_counter() - t0) * 1000

    logger.info(
        "[Ingest] image_id=%s park_id=%s faces=%d time=%.1fms",
        image_id, park_id, faces_detected, elapsed_ms,
    )

    return IngestResponse(
        image_id=image_id,
        park_id=park_id,
        faces_detected=faces_detected,
        faces=face_results,
        processing_time_ms=round(elapsed_ms, 2),
    )


# ── Batch ingest ────────────────────────────────────────────────────

@router.post("/batch", response_model=BatchIngestResponse)
async def ingest_batch(
    files: List[UploadFile] = File(...),
    park_id: str = Form(...),
    image_ids: str = Form(..., description="Comma-separated image IDs, one per file"),
):
    """Ingest multiple images in one request.

    The FAISS index is saved once at the end (not per-image) for efficiency.
    `image_ids` should be a comma-separated string with one ID per uploaded file.
    """
    id_list = [s.strip() for s in image_ids.split(",") if s.strip()]
    if len(id_list) != len(files):
        raise HTTPException(
            status_code=400,
            detail=f"Got {len(files)} files but {len(id_list)} image_ids",
        )

    t0 = time.perf_counter()
    results: list[BatchIngestImageResult] = []
    total_faces = 0
    today = malaysia_today_iso()

    for upload, img_id in zip(files, id_list):
        err = validate_content_type(upload.content_type)
        if err:
            results.append(
                BatchIngestImageResult(image_id=img_id, faces_detected=0, faces=[], error=err)
            )
            continue

        raw = await upload.read()
        if not raw:
            results.append(
                BatchIngestImageResult(
                    image_id=img_id, faces_detected=0, faces=[], error="Empty file"
                )
            )
            continue

        try:
            n, faces = _process_single_image(
                raw, park_id, img_id, ingest_date=today,
            )
            total_faces += n
            results.append(
                BatchIngestImageResult(image_id=img_id, faces_detected=n, faces=faces)
            )
        except Exception as exc:
            logger.exception("Batch ingest error image_id=%s", img_id)
            results.append(
                BatchIngestImageResult(
                    image_id=img_id, faces_detected=0, faces=[], error=str(exc)
                )
            )

    fm = get_faiss_manager()
    fm.save_index(park_id)

    elapsed_ms = (time.perf_counter() - t0) * 1000

    logger.info(
        "[Batch Ingest] park_id=%s images=%d total_faces=%d time=%.1fms",
        park_id, len(files), total_faces, elapsed_ms,
    )

    return BatchIngestResponse(
        park_id=park_id,
        images_processed=len(results),
        total_faces=total_faces,
        results=results,
        processing_time_ms=round(elapsed_ms, 2),
    )

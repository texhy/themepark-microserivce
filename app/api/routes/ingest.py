"""Pipeline 1 — Photographer image ingestion.

POST /ingest/          Single image ingestion
POST /ingest/batch     Multi-image batch ingestion

Date is always auto-set to today (Malaysia timezone) by the FAISS manager.
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

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ingest", tags=["ingest"])

_detector = ScrfdDetector()


def _process_single_image(
    raw_bytes: bytes,
    park_id: str,
    image_id: str,
) -> tuple[int, List[IngestFaceResult]]:
    """Detect faces, generate embeddings, store in FAISS.

    Returns (faces_detected, list_of_face_results).
    """
    t0 = time.perf_counter()

    image = decode_image_bytes(raw_bytes)
    t_decode = time.perf_counter()

    detected = _detector.detect(image)
    t_detect = time.perf_counter()

    fm = get_faiss_manager()
    face_results: list[IngestFaceResult] = []

    if detected:
        face_ids = [str(uuid.uuid4()) for _ in detected]
        import numpy as np

        embeddings = np.stack([f.embedding for f in detected])
        image_ids = [image_id] * len(detected)
        fm.add_embeddings_batch(
            park_id, face_ids, embeddings,
            image_ids=image_ids,
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

    t_store = time.perf_counter()

    logger.info(
        "  ├─ image=%s faces=%d  [decode=%.0fms detect=%.0fms store=%.0fms]",
        image_id.split("/")[-1] if "/" in image_id else image_id,
        len(detected),
        (t_decode - t0) * 1000,
        (t_detect - t_decode) * 1000,
        (t_store - t_detect) * 1000,
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

    logger.info(
        "[Ingest] START park=%s image=%s date=auto(today) size=%dKB",
        park_id, image_id, len(raw) // 1024,
    )

    try:
        faces_detected, face_results = _process_single_image(
            raw, park_id, image_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        logger.exception("Ingest failed for image_id=%s park_id=%s", image_id, park_id)
        raise HTTPException(status_code=500, detail="Internal processing error")

    fm = get_faiss_manager()
    fm.save_index(park_id)

    elapsed_ms = (time.perf_counter() - t0) * 1000

    pi = fm._parks.get(park_id)
    logger.info(
        "[Ingest] DONE park=%s image=%s faces=%d time=%.1fms  "
        "index_total=%d next_id=%d",
        park_id, image_id, faces_detected, elapsed_ms,
        pi.total_vectors if pi else 0,
        pi._next_id if pi else 0,
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
    Date is always auto-set to today (Malaysia timezone).
    `image_ids` should be a comma-separated string with one ID per uploaded file.
    """
    id_list = [s.strip() for s in image_ids.split(",") if s.strip()]
    if len(id_list) != len(files):
        raise HTTPException(
            status_code=400,
            detail=f"Got {len(files)} files but {len(id_list)} image_ids",
        )

    fm = get_faiss_manager()
    pi_before = fm._parks.get(park_id)
    vectors_before = pi_before.total_vectors if pi_before else 0
    next_id_before = pi_before._next_id if pi_before else 0

    logger.info(
        "═══════════════════════════════════════════════════════════════\n"
        "[Batch Ingest] START  park=%s  images=%d  date=auto(today)\n"
        "  ├─ index BEFORE: %d vectors, next_id=%d\n"
        "  ├─ day_ranges BEFORE: %s",
        park_id, len(files),
        vectors_before, next_id_before,
        sorted(pi_before.day_ranges.keys()) if pi_before else "[]",
    )

    t0 = time.perf_counter()
    results: list[BatchIngestImageResult] = []
    total_faces = 0
    errors = 0

    for i, (upload, img_id) in enumerate(zip(files, id_list)):
        err = validate_content_type(upload.content_type)
        if err:
            results.append(
                BatchIngestImageResult(image_id=img_id, faces_detected=0, faces=[], error=err)
            )
            errors += 1
            continue

        raw = await upload.read()
        if not raw:
            results.append(
                BatchIngestImageResult(
                    image_id=img_id, faces_detected=0, faces=[], error="Empty file"
                )
            )
            errors += 1
            continue

        try:
            n, faces = _process_single_image(
                raw, park_id, img_id,
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
            errors += 1

    t_process = time.perf_counter()

    fm.save_index(park_id)

    t_save = time.perf_counter()
    elapsed_ms = (t_save - t0) * 1000

    pi_after = fm._parks.get(park_id)
    vectors_after = pi_after.total_vectors if pi_after else 0
    next_id_after = pi_after._next_id if pi_after else 0

    logger.info(
        "[Batch Ingest] DONE  park=%s  images=%d  faces=%d  errors=%d  time=%.1fms\n"
        "  ├─ processing: %.1fms  saving: %.1fms\n"
        "  ├─ index AFTER: %d vectors (+%d), next_id=%d\n"
        "  ├─ day_ranges AFTER: %s\n"
        "═══════════════════════════════════════════════════════════════",
        park_id, len(files), total_faces, errors, elapsed_ms,
        (t_process - t0) * 1000, (t_save - t_process) * 1000,
        vectors_after, vectors_after - vectors_before, next_id_after,
        {d: f"[{r['start_id']},{r['end_id']})={r['end_id']-r['start_id']}v"
         for d, r in sorted((pi_after.day_ranges if pi_after else {}).items())},
    )

    return BatchIngestResponse(
        park_id=park_id,
        images_processed=len(results),
        total_faces=total_faces,
        results=results,
        processing_time_ms=round(elapsed_ms, 2),
    )

"""Pipeline 1 — Photographer image ingestion.

POST /ingest/              Single image ingestion (sync)
POST /ingest/batch         Multi-image batch ingestion (async via Redis Streams)
GET  /ingest/batch/status  Poll batch status

Date is always auto-set to today (Malaysia timezone) by the FAISS manager.
"""

from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path
from typing import List

import redis
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.config import get_settings
from app.models.schemas import (
    BatchIngestAsyncResponse,
    BatchStatusResponse,
    IngestFaceResult,
    IngestResponse,
)
from app.services.face_detector import ScrfdDetector
from app.services.faiss_manager import get_faiss_manager
from app.services.image_utils import decode_image_bytes, validate_content_type

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ingest", tags=["ingest"])

_detector = ScrfdDetector()

# ── Redis connection (lazy init) ─────────────────────────────────
_redis: redis.Redis | None = None


def _get_redis() -> redis.Redis:
    global _redis
    if _redis is None:
        settings = get_settings()
        _redis = redis.Redis.from_url(settings.redis_url, decode_responses=False)
        # Ensure consumer groups exist
        for stream, group in [("ingest:raw", "embed-workers"), ("ingest:results", "faiss-writers")]:
            try:
                _redis.xgroup_create(stream, group, id="0", mkstream=True)
            except redis.ResponseError as e:
                if "BUSYGROUP" not in str(e):
                    raise
    return _redis


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


# ── Batch ingest (ASYNC via Redis Streams) ─────────────────────────

@router.post("/batch", response_model=BatchIngestAsyncResponse, status_code=202)
async def ingest_batch(
    files: List[UploadFile] = File(...),
    park_id: str = Form(...),
    image_ids: str = Form(..., description="Comma-separated image IDs, one per file"),
):
    """Ingest multiple images asynchronously via Redis Streams.

    Files are saved to disk and one message per image is published to the
    `ingest:raw` stream. Workers detect faces, generate embeddings, and write
    to FAISS in the background. Returns 202 immediately with a batch_id.

    Poll GET /ingest/batch/status/{batch_id} for progress.
    """
    id_list = [s.strip() for s in image_ids.split(",") if s.strip()]
    if len(id_list) != len(files):
        raise HTTPException(
            status_code=400,
            detail=f"Got {len(files)} files but {len(id_list)} image_ids",
        )

    batch_id = str(uuid.uuid4())
    settings = get_settings()
    uploads_dir = settings.resolved_uploads_dir()

    t0 = time.perf_counter()

    # ── Save files to disk + publish to Redis ────────────────────
    r = _get_redis()
    saved_count = 0

    for i, (upload, img_id) in enumerate(zip(files, id_list)):
        err = validate_content_type(upload.content_type)
        if err:
            logger.warning("[Batch Async] Skipping %s: %s", img_id, err)
            continue

        raw = await upload.read()
        if not raw:
            logger.warning("[Batch Async] Skipping %s: empty file", img_id)
            continue

        # Resolve the absolute path for the image on disk.
        # image_ids come in as "uploads/ParkName/filename.jpg" from Node.js.
        # If file is sent via multipart (not already on disk), save it.
        image_path = uploads_dir / img_id.replace("uploads/", "", 1) if img_id.startswith("uploads/") else uploads_dir / img_id

        if not image_path.exists():
            image_path.parent.mkdir(parents=True, exist_ok=True)
            image_path.write_bytes(raw)

        r.xadd("ingest:raw", {
            "batch_id": batch_id,
            "image_id": img_id,
            "park_id": park_id,
            "image_path": str(image_path),
            "position": str(i + 1),
            "batch_total": str(len(id_list)),
            "enqueued_at": str(int(time.time() * 1000)),
        })
        saved_count += 1

    # ── Init batch tracking hash ─────────────────────────────────
    r.hset(f"batch:{batch_id}", mapping={
        "total": str(saved_count),
        "pending": str(saved_count),
        "done": "0",
        "failed": "0",
        "park_id": park_id,
        "status": "processing",
        "created_at": str(int(time.time() * 1000)),
    })
    r.expire(f"batch:{batch_id}", 86400)  # TTL: 24 hours

    elapsed_ms = (time.perf_counter() - t0) * 1000

    logger.info(
        "═══════════════════════════════════════════════════════════════\n"
        "[Batch Async] QUEUED  batch=%s  park=%s  images=%d  time=%.1fms\n"
        "═══════════════════════════════════════════════════════════════",
        batch_id[:8], park_id, saved_count, elapsed_ms,
    )

    return BatchIngestAsyncResponse(
        batch_id=batch_id,
        park_id=park_id,
        images_queued=saved_count,
    )


# ── Batch status polling ───────────────────────────────────────────

@router.get("/batch/status/{batch_id}", response_model=BatchStatusResponse)
async def batch_status(batch_id: str):
    """Poll the status of an async batch ingest job."""
    r = _get_redis()
    data = r.hgetall(f"batch:{batch_id}")

    if not data:
        return BatchStatusResponse(batch_id=batch_id, status="not_found")

    return BatchStatusResponse(
        batch_id=batch_id,
        status=data.get(b"status", b"processing").decode(),
        total=int(data.get(b"total", b"0")),
        done=int(data.get(b"done", b"0")),
        failed=int(data.get(b"failed", b"0")),
        pending=int(data.get(b"pending", b"0")),
    )

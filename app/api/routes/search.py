"""Pipeline 2 — Kiosk / user selfie face search.

POST /search/   Upload selfie → detect face → embed → search FAISS → return matches
"""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.config import get_settings
from app.models.schemas import SearchMatchResult, SearchResponse
from app.services.face_detector import ScrfdDetector, YoloFaceDetector
from app.services.faiss_manager import get_faiss_manager
from app.services.image_utils import decode_image_bytes, validate_content_type
from app.utils.malaysia_time import malaysia_today_iso

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/search", tags=["search"])


@router.post("/", response_model=SearchResponse)
async def search_face(
    file: UploadFile = File(...),
    park_id: str = Form(...),
    top_k: int = Form(default=None),
    threshold: float = Form(default=None),
    detector: str = Form(
        default="yolo",
        description="Detection backend: 'yolo' (fast, default) or 'scrfd' (same as ingest)",
    ),
):
    """Search for matching faces in a park's FAISS index.

    Accepts a selfie image, detects the single best face, generates an
    ArcFace embedding, and searches the park's FAISS index for matches.

    Form fields:
    - **top_k**: Number of top matches to return (default from config)
    - **threshold**: Minimum similarity score (default from config)
    - **detector**: 'yolo' (YOLOv8n-face, faster) or 'scrfd' (InsightFace, same as ingest)

    Search is restricted to **today's** Malaysia calendar date (Asia/Kuala_Lumpur); set on the server only.
    """
    search_date = malaysia_today_iso()
    settings = get_settings()
    k = top_k if top_k is not None else settings.top_k
    sim_threshold = threshold if threshold is not None else settings.similarity_threshold

    # ── Validate input ───────────────────────────────────────────
    err = validate_content_type(file.content_type)
    if err:
        raise HTTPException(status_code=400, detail=err)

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file body")

    t0 = time.perf_counter()

    # ── Decode image ─────────────────────────────────────────────
    try:
        image = decode_image_bytes(raw)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    # ── Detect single best face ──────────────────────────────────
    detector_used = detector.strip().lower()
    face = None

    try:
        if detector_used == "yolo":
            yolo = YoloFaceDetector(conf_threshold=0.4)
            face = yolo.detect_best_face(image)
        elif detector_used == "scrfd":
            scrfd = ScrfdDetector(confidence_threshold=0.4)
            faces = scrfd.detect(image)
            if faces:
                face = max(
                    faces,
                    key=lambda f: max(0.0, f.bbox[2] - f.bbox[0])
                    * max(0.0, f.bbox[3] - f.bbox[1])
                    * f.confidence,
                )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown detector '{detector_used}'. Use 'yolo' or 'scrfd'.",
            )
    except HTTPException:
        raise
    except Exception:
        logger.exception("Face detection failed in search (detector=%s)", detector_used)
        raise HTTPException(status_code=500, detail="Face detection error")

    if face is None:
        fm = get_faiss_manager()
        pi = fm._parks.get(park_id)
        faces_in_index = pi.total_vectors if pi else 0
        filtered_ff = faces_in_index
        if pi and search_date in pi.day_ranges:
            r = pi.day_ranges[search_date]
            filtered_ff = r["end_id"] - r["start_id"]
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return SearchResponse(
            status="no_face",
            park_id=park_id,
            matches=[],
            total_matches=0,
            query_face_confidence=0.0,
            search_time_ms=round(elapsed_ms, 2),
            faces_in_index=faces_in_index,
            filtered_faces=filtered_ff,
            detector_used=detector_used,
            target_date=search_date,
        )

    # ── Search FAISS ─────────────────────────────────────────────
    fm = get_faiss_manager()
    pi = fm._parks.get(park_id)
    if pi is None or pi.total_vectors == 0:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return SearchResponse(
            status="empty_index",
            park_id=park_id,
            matches=[],
            total_matches=0,
            query_face_confidence=round(face.confidence, 4),
            search_time_ms=round(elapsed_ms, 2),
            faces_in_index=0,
            filtered_faces=0,
            detector_used=detector_used,
            target_date=search_date,
        )

    matches = fm.search(
        park_id=park_id,
        query_embedding=face.embedding,
        top_k=k,
        threshold=sim_threshold,
        target_date=search_date,
    )

    match_results = [
        SearchMatchResult(
            face_id=m.face_id,
            image_id=m.image_id,
            similarity_score=round(m.score, 4),
            rank=m.rank + 1,
        )
        for m in matches
    ]

    elapsed_ms = (time.perf_counter() - t0) * 1000

    filtered_count = pi.total_vectors
    if search_date in pi.day_ranges:
        r = pi.day_ranges[search_date]
        filtered_count = r["end_id"] - r["start_id"]

    logger.info(
        "[Search] park=%s detector=%s matches=%d/%d (filtered=%d) threshold=%.2f date=%s time=%.1fms",
        park_id,
        detector_used,
        len(match_results),
        pi.total_vectors,
        filtered_count,
        sim_threshold,
        search_date,
        elapsed_ms,
    )

    return SearchResponse(
        status="found" if match_results else "no_match",
        park_id=park_id,
        matches=match_results,
        total_matches=len(match_results),
        query_face_confidence=round(face.confidence, 4),
        search_time_ms=round(elapsed_ms, 2),
        faces_in_index=pi.total_vectors,
        filtered_faces=filtered_count,
        detector_used=detector_used,
        target_date=search_date,
    )

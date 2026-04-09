"""Pipeline 2 — Kiosk / user selfie face search.

POST /search/   Upload selfie → detect face → embed → search FAISS → return matches

Date is always auto-set to today (Malaysia timezone). Only faces ingested
today are searched — matching the park's daily operational hours (12pm–9pm).
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from zoneinfo import ZoneInfo

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.config import get_settings
from app.models.schemas import SearchMatchResult, SearchResponse
from app.services.face_detector import ScrfdDetector, YoloFaceDetector
from app.services.faiss_manager import get_faiss_manager
from app.services.image_utils import decode_image_bytes, validate_content_type

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/search", tags=["search"])

_MALAYSIA_TZ = ZoneInfo("Asia/Kuala_Lumpur")


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
    """Search for matching faces in a park's FAISS index (today only)."""
    settings = get_settings()
    k = top_k if top_k is not None else settings.top_k
    sim_threshold = threshold if threshold is not None else settings.similarity_threshold

    target_date = datetime.now(_MALAYSIA_TZ).date().isoformat()

    # ── Validate input ───────────────────────────────────────────
    err = validate_content_type(file.content_type)
    if err:
        raise HTTPException(status_code=400, detail=err)

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file body")

    t0 = time.perf_counter()

    fm = get_faiss_manager()
    pi = fm._parks.get(park_id)

    logger.info(
        "───────────────────────────────────────────────────────────────\n"
        "[Search] START  park=%s  detector=%s  date=%s  threshold=%.2f  top_k=%d\n"
        "  ├─ selfie size: %dKB\n"
        "  ├─ index state: %s",
        park_id, detector, target_date, sim_threshold, k,
        len(raw) // 1024,
        f"{pi.total_vectors} vectors, {len(pi.day_ranges)} days, next_id={pi._next_id}"
        if pi else "NOT LOADED",
    )

    if pi:
        if target_date in pi.day_ranges:
            r = pi.day_ranges[target_date]
            logger.info(
                "  ├─ date filter: %s → IDs [%d, %d) = %d vectors",
                target_date, r["start_id"], r["end_id"],
                r["end_id"] - r["start_id"],
            )
        else:
            logger.warning(
                "  ├─ date filter: %s → NOT FOUND in day_ranges!\n"
                "  │   available dates: %s\n"
                "  │   This search will return EMPTY — no vectors for this date.",
                target_date, sorted(pi.day_ranges.keys()),
            )

    # ── Decode image ─────────────────────────────────────────────
    try:
        image = decode_image_bytes(raw)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    t_decode = time.perf_counter()

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

    t_detect = time.perf_counter()

    if face is None:
        faces_in_index = pi.total_vectors if pi else 0
        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.warning(
            "[Search] NO FACE detected in selfie  park=%s  time=%.1fms\n"
            "───────────────────────────────────────────────────────────────",
            park_id, elapsed_ms,
        )
        return SearchResponse(
            status="no_face",
            park_id=park_id,
            matches=[],
            total_matches=0,
            query_face_confidence=0.0,
            search_time_ms=round(elapsed_ms, 2),
            faces_in_index=faces_in_index,
            detector_used=detector_used,
        )

    logger.info(
        "  ├─ face detected: confidence=%.4f  bbox=[%.0f,%.0f,%.0f,%.0f]",
        face.confidence,
        face.bbox[0], face.bbox[1], face.bbox[2], face.bbox[3],
    )

    # ── Search FAISS ─────────────────────────────────────────────
    if pi is None or pi.total_vectors == 0:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.warning(
            "[Search] EMPTY INDEX  park=%s  time=%.1fms\n"
            "───────────────────────────────────────────────────────────────",
            park_id, elapsed_ms,
        )
        return SearchResponse(
            status="empty_index",
            park_id=park_id,
            matches=[],
            total_matches=0,
            query_face_confidence=round(face.confidence, 4),
            search_time_ms=round(elapsed_ms, 2),
            faces_in_index=0,
            detector_used=detector_used,
        )

    matches = fm.search(
        park_id=park_id,
        query_embedding=face.embedding,
        top_k=k,
        threshold=sim_threshold,
        target_date=target_date,
    )

    t_faiss = time.perf_counter()

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
    if target_date in pi.day_ranges:
        r = pi.day_ranges[target_date]
        filtered_count = r["end_id"] - r["start_id"]

    decode_ms = (t_decode - t0) * 1000
    detect_ms = (t_detect - t_decode) * 1000
    faiss_ms = (t_faiss - t_detect) * 1000

    status = "found" if match_results else "no_match"

    logger.info(
        "[Search] %s  park=%s  matches=%d/%d (date_filtered=%d)  time=%.1fms\n"
        "  ├─ decode=%.1fms  detect+embed=%.1fms  faiss=%.1fms\n"
        "  ├─ detector=%s  threshold=%.2f  date=%s\n"
        "  └─ %s\n"
        "───────────────────────────────────────────────────────────────",
        status.upper(), park_id,
        len(match_results), pi.total_vectors, filtered_count,
        elapsed_ms,
        decode_ms, detect_ms, faiss_ms,
        detector_used, sim_threshold, target_date,
        f"top match: score={match_results[0].similarity_score:.4f} image={match_results[0].image_id}"
        if match_results
        else "no matches above threshold",
    )

    return SearchResponse(
        status=status,
        park_id=park_id,
        matches=match_results,
        total_matches=len(match_results),
        query_face_confidence=round(face.confidence, 4),
        search_time_ms=round(elapsed_ms, 2),
        faces_in_index=pi.total_vectors,
        filtered_faces=filtered_count,
        detector_used=detector_used,
        target_date=target_date,
    )

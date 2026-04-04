"""Park index management — cleanup, stats, rebuild."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Form, HTTPException, Query

from app.services.faiss_manager import get_faiss_manager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/manage", tags=["management"])


@router.post("/cleanup")
async def cleanup_old_faces(
    park_id: str = Form(...),
    keep_days: int = Form(default=1, description="Keep faces from the last N days"),
):
    """Remove faces older than *keep_days* from a park's FAISS index.

    The index is rebuilt in-place and saved to disk.  At typical daily
    volumes (~3 000 faces) the rebuild takes < 50 ms.
    """
    fm = get_faiss_manager()
    pi = fm._parks.get(park_id)
    if pi is None:
        raise HTTPException(status_code=404, detail=f"Park {park_id} not found")

    before = pi.total_vectors
    removed = fm.cleanup_old_faces(park_id, keep_days=keep_days)
    fm.save_index(park_id)
    after = pi.total_vectors

    logger.info(
        "[Cleanup] park=%s removed=%d before=%d after=%d keep_days=%d",
        park_id, removed, before, after, keep_days,
    )

    return {
        "status": "ok",
        "park_id": park_id,
        "removed": removed,
        "before": before,
        "after": after,
        "keep_days": keep_days,
    }


@router.get("/day-ranges")
async def get_day_ranges(park_id: str = Query(...)):
    """Return the date-to-ID-range mapping for a park's FAISS index."""
    fm = get_faiss_manager()
    pi = fm._parks.get(park_id)
    if pi is None:
        raise HTTPException(status_code=404, detail=f"Park {park_id} not found")

    return {
        "park_id": park_id,
        "total_vectors": pi.total_vectors,
        "day_ranges": pi.day_ranges,
    }

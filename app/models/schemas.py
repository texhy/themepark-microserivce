"""Shared API schemas."""

from __future__ import annotations

from typing import Any, List, Optional

from pydantic import BaseModel, Field


# ── Health ───────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = Field(..., description="Service version")
    models: dict[str, Any] = Field(default_factory=dict)
    gpu: dict[str, Any] = Field(default_factory=dict)
    faiss: dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    detail: str


# ── Ingest ───────────────────────────────────────────────────────────

class IngestFaceResult(BaseModel):
    face_id: str
    bbox: List[float]
    confidence: float
    embedding_stored: bool = True


class IngestResponse(BaseModel):
    status: str = "processed"
    image_id: str
    park_id: str
    faces_detected: int
    faces: List[IngestFaceResult]
    processing_time_ms: float


class BatchIngestImageResult(BaseModel):
    image_id: str
    faces_detected: int
    faces: List[IngestFaceResult]
    error: Optional[str] = None


class BatchIngestResponse(BaseModel):
    status: str = "processed"
    park_id: str
    images_processed: int
    total_faces: int
    results: List[BatchIngestImageResult]
    processing_time_ms: float


class BatchIngestAsyncResponse(BaseModel):
    """Response for async batch ingest — returns immediately with a batch_id."""
    status: str = "accepted"
    batch_id: str
    park_id: str
    images_queued: int
    message: str = "Images queued for processing. Poll /ingest/batch/status/{batch_id} for progress."


class BatchStatusResponse(BaseModel):
    """Status of an async batch ingest job."""
    batch_id: str
    status: str  # "processing" | "complete" | "not_found"
    total: int = 0
    done: int = 0
    failed: int = 0
    pending: int = 0


# ── Search ───────────────────────────────────────────────────────────

class SearchMatchResult(BaseModel):
    face_id: str
    image_id: str = ""
    similarity_score: float
    rank: int


class SearchResponse(BaseModel):
    status: str = "found"
    park_id: str
    matches: List[SearchMatchResult]
    total_matches: int
    query_face_confidence: float
    search_time_ms: float
    faces_in_index: int
    filtered_faces: int = 0
    detector_used: str
    target_date: Optional[str] = None

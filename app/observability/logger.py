"""Structured logging with structlog.

Dev  → colored, human-readable console output.
Prod → JSON to stdout (unchanged; just set APP_ENV=production).

All environments → append JSON to logs/YYYY-MM-DD.log (date-rotated).
"""

from __future__ import annotations

import hashlib
import logging
import os
import sys
import time
import traceback
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

import structlog

APP_ENV = os.getenv("APP_ENV", "development")
IS_PRODUCTION = APP_ENV in ("production", "staging")
LOG_DIR = Path(__file__).resolve().parent.parent.parent / "logs"


def configure_logging() -> None:
    """Call once at startup in main.py lifespan."""

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_level = logging.DEBUG if not IS_PRODUCTION else logging.INFO

    # ── stdlib root logger: file handler (JSON lines) ──────────────
    root = logging.getLogger()
    root.setLevel(log_level)

    # Remove any existing handlers to avoid duplicates on reload
    for h in root.handlers[:]:
        root.removeHandler(h)

    # Console handler (structured via structlog, see below)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    root.addHandler(console_handler)

    # Date-rotated file handler: logs/YYYY-MM-DD.log
    today_str = datetime.now().strftime("%Y-%m-%d")
    file_handler = TimedRotatingFileHandler(
        filename=str(LOG_DIR / f"{today_str}.log"),
        when="midnight",
        interval=1,
        backupCount=30,
        encoding="utf-8",
    )
    file_handler.setLevel(log_level)
    # File gets JSON lines — use a simple formatter (structlog does the rest)
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(file_handler)

    # ── structlog processors ───────────────────────────────────────
    shared_processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=False),
        structlog.processors.StackInfoRenderer(),
    ]

    if IS_PRODUCTION:
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=shared_processors + [
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


# ── Module-level logger ────────────────────────────────────────────

log = structlog.get_logger("ingest")


# ── Event emitters ─────────────────────────────────────────────────

def emit_image_enqueued(
    *,
    batch_id: str,
    image_id: str,
    park_id: str,
    file_size_bytes: int,
    position_in_batch: int,
    batch_total: int,
) -> None:
    log.info(
        "IMAGE_ENQUEUED",
        batch_id=batch_id,
        image_id=_short_image_id(image_id),
        park_id=park_id,
        file_size_kb=round(file_size_bytes / 1024, 1),
        pos=f"{position_in_batch}/{batch_total}",
    )


def emit_image_processed(
    *,
    batch_id: str,
    image_id: str,
    park_id: str,
    worker_id: str,
    faces_found: int,
    decode_ms: float,
    detect_ms: float,
    faiss_write_ms: float,
    total_ms: float,
    file_size_bytes: int,
    image_width_px: int,
    image_height_px: int,
    faiss_vectors_after: int,
    position_in_batch: int,
    batch_total: int,
) -> None:
    face_label = f"{faces_found} face{'s' if faces_found != 1 else ''}"
    if faces_found == 0:
        symbol = "-"
    else:
        symbol = "+"

    log.info(
        "IMAGE_PROCESSED",
        batch_id=batch_id[:8],
        pos=f"[{position_in_batch:02d}/{batch_total:02d}]",
        status=symbol,
        faces=face_label,
        decode=f"{decode_ms:.0f}ms",
        detect=f"{detect_ms:.0f}ms",
        faiss=f"{faiss_write_ms:.0f}ms",
        total=f"{total_ms:.0f}ms",
        image=_short_image_id(image_id),
        dims=f"{image_width_px}x{image_height_px}",
        size_kb=round(file_size_bytes / 1024, 1),
        worker=worker_id,
        faiss_total=faiss_vectors_after,
    )


def emit_image_failed(
    *,
    batch_id: str,
    image_id: str,
    park_id: str,
    worker_id: str,
    stage: str,
    exc: Exception,
    file_size_bytes: int = 0,
    position_in_batch: int = 0,
    batch_total: int = 0,
    retry_count: int = 0,
) -> None:
    tb = traceback.format_exc()
    tb_hash = hashlib.sha256(tb.encode()).hexdigest()[:12]
    log.error(
        "IMAGE_FAILED",
        batch_id=batch_id[:8],
        pos=f"[{position_in_batch:02d}/{batch_total:02d}]",
        image=_short_image_id(image_id),
        stage=stage,
        error_type=type(exc).__name__,
        error=str(exc)[:200],
        traceback_hash=tb_hash,
        size_kb=round(file_size_bytes / 1024, 1),
        worker=worker_id,
        retry=retry_count,
    )


def emit_batch_complete(
    *,
    batch_id: str,
    park_id: str,
    worker_id: str,
    images_total: int,
    images_processed: int,
    images_failed: int,
    images_no_face: int,
    total_faces_found: int,
    batch_duration_ms: float,
    faiss_vectors_before: int,
    faiss_vectors_after: int,
    faiss_save_ms: float,
    per_image_ms: list[float],
) -> None:
    sorted_ms = sorted(per_image_ms) if per_image_ms else [0.0]
    n = len(sorted_ms)

    def _pct(p: float) -> float:
        idx = int(p / 100 * n)
        return round(sorted_ms[min(idx, n - 1)], 1)

    duration_sec = batch_duration_ms / 1000
    img_per_sec = round(images_processed / duration_sec, 2) if duration_sec > 0 else 0
    faces_per_sec = round(total_faces_found / duration_sec, 2) if duration_sec > 0 else 0

    log.info(
        "BATCH_COMPLETE",
        batch_id=batch_id[:8],
        park_id=park_id,
        processed=images_processed,
        failed=images_failed,
        no_face=images_no_face,
        total=images_total,
        faces=total_faces_found,
        duration=f"{batch_duration_ms:.0f}ms",
        img_per_sec=img_per_sec,
        faces_per_sec=faces_per_sec,
        p50=f"{_pct(50):.0f}ms",
        p95=f"{_pct(95):.0f}ms",
        faiss_before=faiss_vectors_before,
        faiss_after=faiss_vectors_after,
        faiss_delta=f"+{faiss_vectors_after - faiss_vectors_before}",
        faiss_save=f"{faiss_save_ms:.0f}ms",
        worker=worker_id,
    )


def _short_image_id(image_id: str) -> str:
    """uploads/ParkName/1234-567.jpg → 1234-567.jpg"""
    if "/" in image_id:
        return image_id.rsplit("/", 1)[-1]
    return image_id

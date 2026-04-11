"""Embed worker — reads from Redis Stream `ingest:raw`, runs SCRFD+ArcFace,
publishes embeddings to `ingest:results`.

Usage:
    WORKER_ID=embed-1 python -m workers.embed_worker
    WORKER_ID=embed-2 python -m workers.embed_worker   # second instance
"""

from __future__ import annotations

import base64
import json
import os
import sys
import time
import uuid
from pathlib import Path

# Ensure project root is on sys.path so `app.*` imports work
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np
import redis

from app.config import get_settings
from app.core.model_loader import load_models
from app.observability.logger import (
    configure_logging,
    emit_image_enqueued,
    emit_image_failed,
    emit_image_processed,
    log,
)
from app.services.face_detector import ScrfdDetector
from app.services.image_utils import decode_image_bytes

# ── Config ────────────────────────────────────────────────────────────

WORKER_ID = os.getenv("WORKER_ID", f"embed-{os.getpid()}")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
STREAM_IN = "ingest:raw"
STREAM_OUT = "ingest:results"
CONSUMER_GROUP = "embed-workers"
BLOCK_MS = 5000          # block for 5s waiting for new messages
REAPER_IDLE_MS = 60_000  # reclaim messages idle > 60s
MAX_RETRIES = 3
DLQ_STREAM = "ingest:dead-letters"


def _ensure_groups(r: redis.Redis) -> None:
    """Create consumer groups idempotently."""
    for stream in (STREAM_IN,):
        try:
            r.xgroup_create(stream, CONSUMER_GROUP, id="0", mkstream=True)
            log.info("consumer_group_created", stream=stream, group=CONSUMER_GROUP)
        except redis.ResponseError as e:
            if "BUSYGROUP" not in str(e):
                raise


def _run_reaper(r: redis.Redis) -> None:
    """Claim messages stuck in PEL for > REAPER_IDLE_MS. DLQ after MAX_RETRIES."""
    try:
        result = r.xautoclaim(
            STREAM_IN, CONSUMER_GROUP, WORKER_ID,
            min_idle_time=REAPER_IDLE_MS, start_id="0-0", count=10,
        )
        if not result or not result[1]:
            return
        claimed = result[1]
        for msg_id, fields in claimed:
            # Check delivery count via xpending
            pending = r.xpending_range(STREAM_IN, CONSUMER_GROUP, msg_id, msg_id, 1)
            if pending and pending[0].get("times_delivered", 0) > MAX_RETRIES:
                log.warning(
                    "dead_letter",
                    msg_id=msg_id.decode() if isinstance(msg_id, bytes) else msg_id,
                    reason="max_retries_exceeded",
                )
                # Move to DLQ
                dlq_fields = {k: v for k, v in fields.items()}
                dlq_fields[b"original_stream"] = STREAM_IN.encode()
                dlq_fields[b"reason"] = b"max_retries"
                r.xadd(DLQ_STREAM, dlq_fields)
                r.xack(STREAM_IN, CONSUMER_GROUP, msg_id)
            else:
                log.info(
                    "reaper_reclaimed",
                    msg_id=msg_id.decode() if isinstance(msg_id, bytes) else msg_id,
                )
    except Exception:
        log.exception("reaper_error")


def _process_message(
    r: redis.Redis,
    msg_id: bytes | str,
    fields: dict,
    detector: ScrfdDetector,
) -> None:
    """Process a single ingest message: decode → detect → embed → publish."""

    image_path = fields[b"image_path"].decode()
    park_id = fields[b"park_id"].decode()
    image_id = fields[b"image_id"].decode()
    batch_id = fields[b"batch_id"].decode()
    batch_total = int(fields.get(b"batch_total", b"1"))
    position = int(fields.get(b"position", b"1"))

    t_start = time.perf_counter()

    # ── Read image from disk ──────────────────────────────────────
    try:
        raw = Path(image_path).read_bytes()
    except Exception as exc:
        emit_image_failed(
            batch_id=batch_id, image_id=image_id, park_id=park_id,
            worker_id=WORKER_ID, stage="read_file", exc=exc,
            position_in_batch=position, batch_total=batch_total,
        )
        r.xack(STREAM_IN, CONSUMER_GROUP, msg_id)
        r.hincrby(f"batch:{batch_id}", "failed", 1)
        r.hincrby(f"batch:{batch_id}", "pending", -1)
        return

    file_size = len(raw)

    # ── Decode ────────────────────────────────────────────────────
    try:
        image = decode_image_bytes(raw)
        t_decode = time.perf_counter()
    except Exception as exc:
        emit_image_failed(
            batch_id=batch_id, image_id=image_id, park_id=park_id,
            worker_id=WORKER_ID, stage="decode", exc=exc,
            file_size_bytes=file_size,
            position_in_batch=position, batch_total=batch_total,
        )
        r.xack(STREAM_IN, CONSUMER_GROUP, msg_id)
        r.hincrby(f"batch:{batch_id}", "failed", 1)
        r.hincrby(f"batch:{batch_id}", "pending", -1)
        return

    h, w = image.shape[:2]

    # ── Detect + Embed (fused in InsightFace) ─────────────────────
    try:
        detected = detector.detect(image)
        t_detect = time.perf_counter()
    except Exception as exc:
        emit_image_failed(
            batch_id=batch_id, image_id=image_id, park_id=park_id,
            worker_id=WORKER_ID, stage="detect", exc=exc,
            file_size_bytes=file_size,
            position_in_batch=position, batch_total=batch_total,
        )
        r.xack(STREAM_IN, CONSUMER_GROUP, msg_id)
        r.hincrby(f"batch:{batch_id}", "failed", 1)
        r.hincrby(f"batch:{batch_id}", "pending", -1)
        return

    # ── Build result payload ──────────────────────────────────────
    faces_payload = []
    for f in detected:
        faces_payload.append({
            "face_id": str(uuid.uuid4()),
            "embedding": base64.b64encode(f.embedding.tobytes()).decode(),
            "bbox": [round(v, 2) for v in f.bbox],
            "confidence": round(f.confidence, 4),
        })

    t_payload = time.perf_counter()

    # ── Publish to results stream ─────────────────────────────────
    r.xadd(STREAM_OUT, {
        "batch_id": batch_id,
        "image_id": image_id,
        "park_id": park_id,
        "faces_json": json.dumps(faces_payload),
        "face_count": str(len(faces_payload)),
        "image_width": str(w),
        "image_height": str(h),
        "file_size": str(file_size),
        "position": str(position),
        "batch_total": str(batch_total),
        "processed_at": str(int(time.time() * 1000)),
        "worker_id": WORKER_ID,
    })
    r.xack(STREAM_IN, CONSUMER_GROUP, msg_id)

    t_end = time.perf_counter()
    total_ms = (t_end - t_start) * 1000
    decode_ms = (t_decode - t_start) * 1000
    detect_ms = (t_detect - t_decode) * 1000

    emit_image_processed(
        batch_id=batch_id, image_id=image_id, park_id=park_id,
        worker_id=WORKER_ID,
        faces_found=len(detected),
        decode_ms=decode_ms,
        detect_ms=detect_ms,
        faiss_write_ms=0,  # FAISS write happens in faiss_writer
        total_ms=total_ms,
        file_size_bytes=file_size,
        image_width_px=w, image_height_px=h,
        faiss_vectors_after=0,  # not known here
        position_in_batch=position,
        batch_total=batch_total,
    )


def main() -> None:
    configure_logging()

    log.info(
        "embed_worker_starting",
        worker_id=WORKER_ID,
        redis=REDIS_URL,
        stream_in=STREAM_IN,
        stream_out=STREAM_OUT,
    )

    # Load ML models (each worker process gets its own copy)
    settings = get_settings()
    load_models(settings)
    detector = ScrfdDetector()

    r = redis.Redis.from_url(REDIS_URL, decode_responses=False)
    r.ping()
    log.info("redis_connected", url=REDIS_URL)

    _ensure_groups(r)

    log.info("embed_worker_ready", worker_id=WORKER_ID)

    idle_cycles = 0
    while True:
        try:
            messages = r.xreadgroup(
                CONSUMER_GROUP, WORKER_ID,
                {STREAM_IN: ">"},
                count=1, block=BLOCK_MS,
            )

            if not messages:
                idle_cycles += 1
                # Run reaper every ~30s of idle (6 cycles × 5s block)
                if idle_cycles % 6 == 0:
                    _run_reaper(r)
                continue

            idle_cycles = 0
            stream_name, entries = messages[0]

            for msg_id, fields in entries:
                _process_message(r, msg_id, fields, detector)

        except redis.ConnectionError:
            log.warning("redis_connection_lost", retry_in="5s")
            time.sleep(5)
        except KeyboardInterrupt:
            log.info("embed_worker_shutdown", worker_id=WORKER_ID)
            break
        except Exception:
            log.exception("embed_worker_unexpected_error")
            time.sleep(1)


if __name__ == "__main__":
    main()

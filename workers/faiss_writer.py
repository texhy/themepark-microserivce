"""FAISS writer worker — reads from Redis Stream `ingest:results`,
decodes embeddings, writes to FAISS index.

Single instance only — serialized FAISS writes avoid lock contention.

Usage:
    python -m workers.faiss_writer
"""

from __future__ import annotations

import base64
import json
import os
import sys
import time
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np
import redis

from app.config import get_settings
from app.observability.logger import (
    configure_logging,
    emit_batch_complete,
    emit_image_failed,
    log,
)
from app.services.faiss_manager import get_faiss_manager

# ── Config ────────────────────────────────────────────────────────────

WORKER_ID = os.getenv("WORKER_ID", "faiss-writer-1")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
STREAM_IN = "ingest:results"
CONSUMER_GROUP = "faiss-writers"
BLOCK_MS = 5000
DLQ_STREAM = "ingest:dead-letters"


def _ensure_groups(r: redis.Redis) -> None:
    try:
        r.xgroup_create(STREAM_IN, CONSUMER_GROUP, id="0", mkstream=True)
        log.info("consumer_group_created", stream=STREAM_IN, group=CONSUMER_GROUP)
    except redis.ResponseError as e:
        if "BUSYGROUP" not in str(e):
            raise


# ── Per-batch tracking ────────────────────────────────────────────────

class BatchTracker:
    """Track per-batch progress for BATCH_COMPLETE logging and FAISS save timing."""

    def __init__(self):
        self._batches: dict[str, dict] = {}

    def touch(self, batch_id: str, park_id: str, batch_total: int) -> dict:
        if batch_id not in self._batches:
            fm = get_faiss_manager()
            pi = fm._parks.get(park_id)
            self._batches[batch_id] = {
                "park_id": park_id,
                "batch_total": batch_total,
                "processed": 0,
                "failed": 0,
                "no_face": 0,
                "total_faces": 0,
                "per_image_ms": [],
                "faiss_vectors_before": pi.total_vectors if pi else 0,
                "t_start": time.perf_counter(),
            }
        return self._batches[batch_id]

    def is_complete(self, batch_id: str) -> bool:
        b = self._batches.get(batch_id)
        if not b:
            return False
        return (b["processed"] + b["failed"]) >= b["batch_total"]

    def pop(self, batch_id: str) -> dict | None:
        return self._batches.pop(batch_id, None)


def _process_message(
    r: redis.Redis,
    msg_id: bytes | str,
    fields: dict,
    tracker: BatchTracker,
) -> None:
    """Decode embeddings from the results stream and write to FAISS."""

    batch_id = fields[b"batch_id"].decode()
    image_id = fields[b"image_id"].decode()
    park_id = fields[b"park_id"].decode()
    face_count = int(fields.get(b"face_count", b"0"))
    batch_total = int(fields.get(b"batch_total", b"1"))
    position = int(fields.get(b"position", b"1"))
    embed_worker = fields.get(b"worker_id", b"unknown").decode()

    batch_info = tracker.touch(batch_id, park_id, batch_total)
    t_start = time.perf_counter()

    fm = get_faiss_manager()

    try:
        if face_count > 0:
            faces = json.loads(fields[b"faces_json"])
            face_ids = [f["face_id"] for f in faces]
            embeddings = np.stack([
                np.frombuffer(base64.b64decode(f["embedding"]), dtype=np.float32)
                for f in faces
            ])
            image_ids_list = [image_id] * len(faces)

            # This call is UNCHANGED from the original pipeline
            fm.add_embeddings_batch(
                park_id, face_ids, embeddings, image_ids=image_ids_list,
            )

            batch_info["total_faces"] += face_count
        else:
            batch_info["no_face"] += 1

        t_faiss = time.perf_counter()
        faiss_ms = (t_faiss - t_start) * 1000

        batch_info["processed"] += 1
        batch_info["per_image_ms"].append(faiss_ms)

        r.xack(STREAM_IN, CONSUMER_GROUP, msg_id)

        # Update batch hash in Redis
        r.hincrby(f"batch:{batch_id}", "done", 1)
        remaining = r.hincrby(f"batch:{batch_id}", "pending", -1)

        log.info(
            "faiss_write",
            batch_id=batch_id[:8],
            pos=f"[{position:02d}/{batch_total:02d}]",
            faces=face_count,
            faiss_ms=f"{faiss_ms:.1f}ms",
            image=image_id.rsplit("/", 1)[-1] if "/" in image_id else image_id,
            embed_worker=embed_worker,
            remaining=remaining,
        )

        # ── Batch complete? Save FAISS once ───────────────────────
        if tracker.is_complete(batch_id):
            t_save_start = time.perf_counter()
            fm.save_index(park_id)
            t_save_end = time.perf_counter()
            save_ms = (t_save_end - t_save_start) * 1000

            pi = fm._parks.get(park_id)
            faiss_vectors_after = pi.total_vectors if pi else 0

            emit_batch_complete(
                batch_id=batch_id,
                park_id=park_id,
                worker_id=WORKER_ID,
                images_total=batch_info["batch_total"],
                images_processed=batch_info["processed"],
                images_failed=batch_info["failed"],
                images_no_face=batch_info["no_face"],
                total_faces_found=batch_info["total_faces"],
                batch_duration_ms=(time.perf_counter() - batch_info["t_start"]) * 1000,
                faiss_vectors_before=batch_info["faiss_vectors_before"],
                faiss_vectors_after=faiss_vectors_after,
                faiss_save_ms=save_ms,
                per_image_ms=batch_info["per_image_ms"],
            )

            # Mark batch as complete in Redis
            r.hset(f"batch:{batch_id}", "status", "complete")
            r.publish("faiss:reloaded", park_id)
            log.info("faiss_reload_published", park_id=park_id, batch_id=batch_id[:8])
            tracker.pop(batch_id)

    except Exception as exc:
        log.exception("faiss_write_error", image_id=image_id, batch_id=batch_id[:8])
        emit_image_failed(
            batch_id=batch_id, image_id=image_id, park_id=park_id,
            worker_id=WORKER_ID, stage="faiss_write", exc=exc,
            position_in_batch=position, batch_total=batch_total,
        )
        r.xack(STREAM_IN, CONSUMER_GROUP, msg_id)
        batch_info["failed"] += 1

        r.hincrby(f"batch:{batch_id}", "failed", 1)
        r.hincrby(f"batch:{batch_id}", "pending", -1)

        # Still check if batch is done (even with errors)
        if tracker.is_complete(batch_id):
            fm.save_index(park_id)
            r.hset(f"batch:{batch_id}", "status", "complete")
            r.publish("faiss:reloaded", park_id)
            log.info("faiss_reload_published", park_id=park_id, batch_id=batch_id[:8])
            tracker.pop(batch_id)


def main() -> None:
    configure_logging()

    log.info(
        "faiss_writer_starting",
        worker_id=WORKER_ID,
        redis=REDIS_URL,
        stream_in=STREAM_IN,
    )

    # Initialize FAISS manager (loads existing indexes from disk)
    settings = get_settings()
    settings.resolved_faiss_index_dir().mkdir(parents=True, exist_ok=True)
    fm = get_faiss_manager()
    log.info("faiss_manager_ready", stats=fm.get_stats())

    r = redis.Redis.from_url(REDIS_URL, decode_responses=False)
    r.ping()
    log.info("redis_connected", url=REDIS_URL)

    _ensure_groups(r)

    tracker = BatchTracker()

    log.info("faiss_writer_ready", worker_id=WORKER_ID)

    while True:
        try:
            messages = r.xreadgroup(
                CONSUMER_GROUP, WORKER_ID,
                {STREAM_IN: ">"},
                count=1, block=BLOCK_MS,
            )

            if not messages:
                continue

            stream_name, entries = messages[0]
            for msg_id, fields in entries:
                _process_message(r, msg_id, fields, tracker)

        except redis.ConnectionError:
            log.warning("redis_connection_lost", retry_in="5s")
            time.sleep(5)
        except KeyboardInterrupt:
            log.info("faiss_writer_shutdown", worker_id=WORKER_ID)
            # Save all indexes before exit
            fm.save_all()
            break
        except Exception:
            log.exception("faiss_writer_unexpected_error")
            time.sleep(1)


if __name__ == "__main__":
    main()

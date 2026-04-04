"""Per-park FAISS index management with disk persistence.

Each park gets its own IndexFlatIP (inner product on L2-normalized vectors
= cosine similarity).  A parallel JSON sidecar maps FAISS integer IDs →
string face_ids so the Node.js backend can resolve them to images.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import faiss
import numpy as np

from app.config import get_settings

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 512

# Malaysia local calendar date — must match Node ingest_date / target_date (Asia/Kuala_Lumpur).
_MALAYSIA_TZ = ZoneInfo("Asia/Kuala_Lumpur")


def _malaysia_today_iso() -> str:
    return datetime.now(_MALAYSIA_TZ).date().isoformat()


@dataclass
class SearchMatch:
    face_id: str
    image_id: str
    score: float
    rank: int


def _normalize_id_entry(entry: Any) -> Dict[str, str]:
    """Handle both old format (plain string) and new format (dict)."""
    if isinstance(entry, dict):
        return entry
    return {"face_id": str(entry), "image_id": ""}


@dataclass
class ParkIndex:
    """Holds a FAISS index + its face_id mapping for one park.

    id_map values are dicts: {"face_id": "uuid", "image_id": "uploads/park/photo.jpg", "ingest_date": "2026-04-04"}
    Backward-compatible with old format where values were plain face_id strings.
    day_ranges maps ISO date strings to contiguous FAISS ID ranges for efficient filtering.
    """

    park_id: str
    index: faiss.Index
    id_map: Dict[int, Dict[str, str]]
    day_ranges: Dict[str, Dict[str, int]] = field(default_factory=dict)
    _next_id: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def _update_day_range(self, date_str: str, min_id: int, max_id: int) -> None:
        """Expand the day_ranges entry for *date_str* to cover [min_id, max_id+1)."""
        if date_str in self.day_ranges:
            r = self.day_ranges[date_str]
            r["start_id"] = min(r["start_id"], min_id)
            r["end_id"] = max(r["end_id"], max_id + 1)
        else:
            self.day_ranges[date_str] = {"start_id": min_id, "end_id": max_id + 1}

    def add(
        self, face_id: str, embedding: np.ndarray,
        image_id: str = "", ingest_date: str = "",
    ) -> int:
        """Add a single embedding. Returns the assigned FAISS integer id."""
        vec = np.asarray(embedding, dtype=np.float32).reshape(1, EMBEDDING_DIM)
        today = ingest_date or _malaysia_today_iso()
        with self._lock:
            fid = self._next_id
            self.index.add(vec)
            self.id_map[fid] = {
                "face_id": face_id, "image_id": image_id, "ingest_date": today,
            }
            self._next_id += 1
            self._update_day_range(today, fid, fid)
        return fid

    def add_batch(
        self,
        face_ids: List[str],
        embeddings: np.ndarray,
        image_ids: Optional[List[str]] = None,
        ingest_date: str = "",
    ) -> List[int]:
        """Add N embeddings at once. Returns list of assigned FAISS int ids."""
        vecs = np.asarray(embeddings, dtype=np.float32).reshape(-1, EMBEDDING_DIM)
        assert vecs.shape[0] == len(face_ids), "face_ids and embeddings count mismatch"
        if image_ids is None:
            image_ids = [""] * len(face_ids)
        today = ingest_date or _malaysia_today_iso()
        with self._lock:
            start = self._next_id
            self.index.add(vecs)
            assigned: list[int] = []
            for i, fid_str in enumerate(face_ids):
                fid = start + i
                self.id_map[fid] = {
                    "face_id": fid_str, "image_id": image_ids[i], "ingest_date": today,
                }
                assigned.append(fid)
            self._next_id = start + len(face_ids)
            self._update_day_range(today, start, start + len(face_ids) - 1)
        return assigned

    @property
    def total_vectors(self) -> int:
        return self.index.ntotal


class FaissManager:
    """Manages per-park FAISS indexes with disk persistence."""

    def __init__(self, index_dir: Optional[Path] = None):
        settings = get_settings()
        self._index_dir = index_dir or settings.resolved_faiss_index_dir()
        self._index_dir.mkdir(parents=True, exist_ok=True)
        self._parks: Dict[str, ParkIndex] = {}
        self._global_lock = threading.Lock()

    # ── Index lifecycle ──────────────────────────────────────────────

    def get_or_create_index(self, park_id: str) -> ParkIndex:
        """Return existing ParkIndex or create a fresh one."""
        with self._global_lock:
            if park_id in self._parks:
                return self._parks[park_id]

            park_dir = self._index_dir / park_id
            park_dir.mkdir(parents=True, exist_ok=True)

            idx_path = park_dir / "index.faiss"
            map_path = park_dir / "id_map.json"
            ranges_path = park_dir / "day_ranges.json"

            if idx_path.is_file() and map_path.is_file():
                index = faiss.read_index(str(idx_path))
                raw_map = json.loads(map_path.read_text(encoding="utf-8"))
                id_map = {int(k): _normalize_id_entry(v) for k, v in raw_map.items()}
                next_id = max(id_map.keys()) + 1 if id_map else 0

                # Load day_ranges from sidecar, or rebuild from id_map
                if ranges_path.is_file():
                    day_ranges = json.loads(ranges_path.read_text(encoding="utf-8"))
                else:
                    day_ranges: dict[str, dict[str, int]] = {}
                    for int_id, entry in id_map.items():
                        d = entry.get("ingest_date", "")
                        if not d:
                            continue
                        if d in day_ranges:
                            r = day_ranges[d]
                            r["start_id"] = min(r["start_id"], int_id)
                            r["end_id"] = max(r["end_id"], int_id + 1)
                        else:
                            day_ranges[d] = {"start_id": int_id, "end_id": int_id + 1}

                logger.info(
                    "[FAISS] Loaded park=%s from disk — %d vectors, %d day(s), next_id=%d",
                    park_id, index.ntotal, len(day_ranges), next_id,
                )
            else:
                index = faiss.IndexFlatIP(EMBEDDING_DIM)
                id_map = {}
                day_ranges = {}
                next_id = 0
                logger.info("[FAISS] Created new IndexFlatIP for park=%s", park_id)

            pi = ParkIndex(
                park_id=park_id,
                index=index,
                id_map=id_map,
                day_ranges=day_ranges,
                _next_id=next_id,
            )
            self._parks[park_id] = pi
            return pi

    def add_embedding(
        self, park_id: str, face_id: str, embedding: np.ndarray,
        image_id: str = "", ingest_date: str = "",
    ) -> int:
        pi = self.get_or_create_index(park_id)
        return pi.add(face_id, embedding, image_id=image_id, ingest_date=ingest_date)

    def add_embeddings_batch(
        self, park_id: str, face_ids: List[str], embeddings: np.ndarray,
        image_ids: Optional[List[str]] = None, ingest_date: str = "",
    ) -> List[int]:
        pi = self.get_or_create_index(park_id)
        return pi.add_batch(face_ids, embeddings, image_ids=image_ids, ingest_date=ingest_date)

    # ── Search ───────────────────────────────────────────────────────

    def search(
        self,
        park_id: str,
        query_embedding: np.ndarray,
        top_k: int = 20,
        threshold: float = 0.0,
        target_date: Optional[str] = None,
    ) -> List[SearchMatch]:
        """Search a park's index, optionally filtered to a single date.

        When *target_date* is provided (ISO ``YYYY-MM-DD``), only vectors
        ingested on that date are considered via ``faiss.IDSelectorRange``.
        """
        pi = self._parks.get(park_id)
        if pi is None or pi.total_vectors == 0:
            return []

        vec = np.asarray(query_embedding, dtype=np.float32).reshape(1, EMBEDDING_DIM)

        search_params = None
        effective_total = pi.total_vectors

        if target_date and target_date in pi.day_ranges:
            r = pi.day_ranges[target_date]
            sel = faiss.IDSelectorRange(r["start_id"], r["end_id"])
            search_params = faiss.SearchParameters(sel=sel)
            effective_total = r["end_id"] - r["start_id"]

        k = min(top_k, max(effective_total, 1))

        if search_params:
            scores, indices = pi.index.search(vec, k, params=search_params)
        else:
            scores, indices = pi.index.search(vec, k)

        matches: list[SearchMatch] = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < 0:
                continue
            if float(score) < threshold:
                continue
            entry = pi.id_map.get(int(idx))
            if entry is None:
                logger.warning("FAISS returned idx=%d with no id_map entry (park=%s)", idx, park_id)
                continue
            entry = _normalize_id_entry(entry)
            matches.append(SearchMatch(
                face_id=entry["face_id"],
                image_id=entry.get("image_id", ""),
                score=float(score),
                rank=rank,
            ))

        return matches

    # ── Cleanup ────────────────────────────────────────────────────────

    def cleanup_old_faces(self, park_id: str, keep_days: int = 1) -> int:
        """Remove faces older than *keep_days* and rebuild the index.

        Returns the number of vectors removed.  At ~3 000 vectors the full
        rebuild takes < 50 ms — negligible for a maintenance operation.
        """
        pi = self._parks.get(park_id)
        if not pi:
            return 0

        cutoff = (
            datetime.now(_MALAYSIA_TZ).date()
            - timedelta(days=max(keep_days - 1, 0))
        ).isoformat()

        ids_to_keep: dict[int, dict] = {}
        removed = 0
        for int_id, entry in pi.id_map.items():
            entry = _normalize_id_entry(entry)
            entry_date = entry.get("ingest_date", "")
            if entry_date and entry_date < cutoff:
                removed += 1
            else:
                ids_to_keep[int_id] = entry

        if removed == 0:
            return 0

        all_vecs = faiss.rev_swig_ptr(
            pi.index.get_xb(), pi.index.ntotal * EMBEDDING_DIM,
        ).reshape(-1, EMBEDDING_DIM).copy()

        new_index = faiss.IndexFlatIP(EMBEDDING_DIM)
        new_id_map: dict[int, dict] = {}
        new_day_ranges: dict[str, dict] = {}
        new_id = 0

        for old_id in sorted(ids_to_keep.keys()):
            new_index.add(all_vecs[old_id : old_id + 1])
            entry = ids_to_keep[old_id]
            entry_date = entry.get("ingest_date", "")
            new_id_map[new_id] = entry

            if entry_date:
                if entry_date in new_day_ranges:
                    new_day_ranges[entry_date]["end_id"] = new_id + 1
                else:
                    new_day_ranges[entry_date] = {"start_id": new_id, "end_id": new_id + 1}

            new_id += 1

        with pi._lock:
            pi.index = new_index
            pi.id_map = new_id_map
            pi.day_ranges = new_day_ranges
            pi._next_id = new_id

        logger.info(
            "[FAISS] Cleanup park=%s removed=%d remaining=%d",
            park_id, removed, new_id,
        )
        return removed

    # ── Persistence ──────────────────────────────────────────────────

    def save_index(self, park_id: str) -> None:
        pi = self._parks.get(park_id)
        if pi is None:
            return

        park_dir = self._index_dir / park_id
        park_dir.mkdir(parents=True, exist_ok=True)

        idx_path = park_dir / "index.faiss"
        map_path = park_dir / "id_map.json"
        ranges_path = park_dir / "day_ranges.json"

        with pi._lock:
            faiss.write_index(pi.index, str(idx_path))
            map_path.write_text(
                json.dumps(pi.id_map, indent=2),
                encoding="utf-8",
            )
            ranges_path.write_text(
                json.dumps(pi.day_ranges, indent=2),
                encoding="utf-8",
            )

        logger.info(
            "[FAISS] Saved park=%s — %d vectors, %d day(s) → %s",
            park_id, pi.total_vectors, len(pi.day_ranges), park_dir,
        )

    def save_all(self) -> None:
        for park_id in list(self._parks.keys()):
            self.save_index(park_id)

    def load_all_indexes(self) -> int:
        """Scan index_dir for existing parks and load them. Returns count loaded."""
        if not self._index_dir.is_dir():
            return 0
        count = 0
        for park_dir in sorted(self._index_dir.iterdir()):
            if not park_dir.is_dir():
                continue
            if (park_dir / "index.faiss").is_file():
                self.get_or_create_index(park_dir.name)
                count += 1
        logger.info("[FAISS] Loaded %d park indexes from %s", count, self._index_dir)
        return count

    # ── Stats ────────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        per_park: dict[str, int] = {}
        total = 0
        for pid, pi in self._parks.items():
            n = pi.total_vectors
            per_park[pid] = n
            total += n
        return {
            "total_vectors": total,
            "parks": per_park,
            "index_dir": str(self._index_dir),
        }


# ── Module-level singleton ───────────────────────────────────────────

_manager: Optional[FaissManager] = None


def get_faiss_manager() -> FaissManager:
    global _manager
    if _manager is None:
        _manager = FaissManager()
        _manager.load_all_indexes()
    return _manager

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
from pathlib import Path
from typing import Any, Dict, List, Optional

import faiss
import numpy as np

from app.config import get_settings

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 512


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

    id_map values are dicts: {"face_id": "uuid", "image_id": "uploads/park/photo.jpg"}
    Backward-compatible with old format where values were plain face_id strings.
    """

    park_id: str
    index: faiss.Index
    id_map: Dict[int, Dict[str, str]]
    _next_id: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, face_id: str, embedding: np.ndarray, image_id: str = "") -> int:
        """Add a single embedding. Returns the assigned FAISS integer id."""
        vec = np.asarray(embedding, dtype=np.float32).reshape(1, EMBEDDING_DIM)
        with self._lock:
            fid = self._next_id
            self.index.add(vec)
            self.id_map[fid] = {"face_id": face_id, "image_id": image_id}
            self._next_id += 1
        return fid

    def add_batch(
        self,
        face_ids: List[str],
        embeddings: np.ndarray,
        image_ids: Optional[List[str]] = None,
    ) -> List[int]:
        """Add N embeddings at once. Returns list of assigned FAISS int ids."""
        vecs = np.asarray(embeddings, dtype=np.float32).reshape(-1, EMBEDDING_DIM)
        assert vecs.shape[0] == len(face_ids), "face_ids and embeddings count mismatch"
        if image_ids is None:
            image_ids = [""] * len(face_ids)
        with self._lock:
            start = self._next_id
            self.index.add(vecs)
            assigned: list[int] = []
            for i, fid_str in enumerate(face_ids):
                fid = start + i
                self.id_map[fid] = {"face_id": fid_str, "image_id": image_ids[i]}
                assigned.append(fid)
            self._next_id = start + len(face_ids)
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

            if idx_path.is_file() and map_path.is_file():
                index = faiss.read_index(str(idx_path))
                raw_map = json.loads(map_path.read_text(encoding="utf-8"))
                id_map = {int(k): _normalize_id_entry(v) for k, v in raw_map.items()}
                next_id = max(id_map.keys()) + 1 if id_map else 0
                logger.info(
                    "[FAISS] Loaded park=%s from disk — %d vectors, next_id=%d",
                    park_id, index.ntotal, next_id,
                )
            else:
                index = faiss.IndexFlatIP(EMBEDDING_DIM)
                id_map = {}
                next_id = 0
                logger.info("[FAISS] Created new IndexFlatIP for park=%s", park_id)

            pi = ParkIndex(
                park_id=park_id,
                index=index,
                id_map=id_map,
                _next_id=next_id,
            )
            self._parks[park_id] = pi
            return pi

    def add_embedding(
        self, park_id: str, face_id: str, embedding: np.ndarray,
        image_id: str = "",
    ) -> int:
        pi = self.get_or_create_index(park_id)
        return pi.add(face_id, embedding, image_id=image_id)

    def add_embeddings_batch(
        self, park_id: str, face_ids: List[str], embeddings: np.ndarray,
        image_ids: Optional[List[str]] = None,
    ) -> List[int]:
        pi = self.get_or_create_index(park_id)
        return pi.add_batch(face_ids, embeddings, image_ids=image_ids)

    # ── Search ───────────────────────────────────────────────────────

    def search(
        self,
        park_id: str,
        query_embedding: np.ndarray,
        top_k: int = 20,
        threshold: float = 0.0,
    ) -> List[SearchMatch]:
        """Search a park's index. Returns matches sorted by descending score."""
        pi = self._parks.get(park_id)
        if pi is None or pi.total_vectors == 0:
            return []

        vec = np.asarray(query_embedding, dtype=np.float32).reshape(1, EMBEDDING_DIM)
        k = min(top_k, pi.total_vectors)
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

    # ── Persistence ──────────────────────────────────────────────────

    def save_index(self, park_id: str) -> None:
        pi = self._parks.get(park_id)
        if pi is None:
            return

        park_dir = self._index_dir / park_id
        park_dir.mkdir(parents=True, exist_ok=True)

        idx_path = park_dir / "index.faiss"
        map_path = park_dir / "id_map.json"

        with pi._lock:
            faiss.write_index(pi.index, str(idx_path))
            map_path.write_text(
                json.dumps(pi.id_map, indent=2),
                encoding="utf-8",
            )

        logger.info(
            "[FAISS] Saved park=%s — %d vectors → %s",
            park_id, pi.total_vectors, park_dir,
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

"""Malaysia calendar date (Asia/Kuala_Lumpur) for ingest/search day buckets."""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

_MALAYSIA_TZ = ZoneInfo("Asia/Kuala_Lumpur")


def malaysia_today_iso() -> str:
    """Current local calendar date in Malaysia as ``YYYY-MM-DD``.

    If ``FAISS_MALAYSIA_DATE_OVERRIDE`` is set (``YYYY-MM-DD``), returns that
    instead — for automated tests only; leave unset in production.
    """
    override = os.environ.get("FAISS_MALAYSIA_DATE_OVERRIDE", "").strip()
    if override:
        return override
    return datetime.now(_MALAYSIA_TZ).date().isoformat()


def malaysia_cutoff_date_iso(keep_days: int) -> str:
    """Cutoff ``ingest_date`` string for cleanup (same semantics as ``cleanup_old_faces``)."""
    d = datetime.now(_MALAYSIA_TZ).date() - timedelta(days=max(keep_days - 1, 0))
    return d.isoformat()

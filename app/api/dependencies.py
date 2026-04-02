"""FastAPI dependencies (Phase 2+ will add auth, DB, etc.)."""

from app.config import Settings, get_settings


def get_config() -> Settings:
    return get_settings()

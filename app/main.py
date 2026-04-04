"""FastAPI application entry."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.api.routes import health, ingest, manage, search
from app.config import get_settings
from app.core.model_loader import load_models, unload_models
from app.services.faiss_manager import get_faiss_manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _ensure_models_downloaded(model_dir: Path) -> None:
    """Check that required model weights exist; download if missing."""
    insightface_dir = model_dir / "insightface" / "models" / "buffalo_l"
    yolo_path = model_dir / "yolov8n-face.onnx"

    need_insightface = not insightface_dir.is_dir() or not any(insightface_dir.glob("*.onnx"))
    need_yolo = not yolo_path.is_file()

    if not need_insightface and not need_yolo:
        logger.info("[Models] All weights present in %s", model_dir)
        return

    import sys
    scripts_dir = Path(__file__).resolve().parent.parent / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

    from download_models import ensure_buffalo_l, ensure_yolov8_face_onnx

    if need_insightface:
        logger.info("[Models] buffalo_l not found — downloading to %s …", model_dir)
        ensure_buffalo_l(model_dir)

    if need_yolo:
        logger.info("[Models] yolov8n-face.onnx not found — downloading to %s …", model_dir)
        ensure_yolov8_face_onnx(model_dir, "yolov8n-face.onnx")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    settings.resolved_faiss_index_dir().mkdir(parents=True, exist_ok=True)
    model_dir = settings.resolved_model_dir()
    model_dir.mkdir(parents=True, exist_ok=True)

    _ensure_models_downloaded(model_dir)

    try:
        load_models(settings)
    except Exception:
        logger.exception("Model load failed — /health will report degraded state")
    fm = get_faiss_manager()
    logger.info("FAISS manager ready — %s", fm.get_stats())
    yield
    fm.save_all()
    unload_models()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(health.router)
    app.include_router(ingest.router)
    app.include_router(search.router)
    app.include_router(manage.router)

    @app.get("/")
    def root() -> dict[str, str]:
        return {"service": settings.app_name, "version": __version__, "docs": "/docs"}

    return app


app = create_app()

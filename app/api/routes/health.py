"""Health and readiness endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from app import __version__
from app.core.model_loader import dummy_face_probe, get_models, gpu_memory_info, models_loaded
from app.models.schemas import HealthResponse
from app.services.faiss_manager import get_faiss_manager

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    gpu = gpu_memory_info()
    models_info: dict = {
        "buffalo_l": False,
        "yolov8_face": False,
        "ctx_id": None,
        "det_size": None,
    }
    if models_loaded():
        m = get_models()
        models_info["buffalo_l"] = dummy_face_probe(m.face_analysis)
        models_info["yolov8_face"] = m.yolo_face is not None
        models_info["ctx_id"] = m.ctx_id
        models_info["det_size"] = list(m.det_size)
        if m.yolo_onnx_path:
            models_info["yolov8_onnx"] = str(m.yolo_onnx_path)
        if m.execution_report:
            models_info["execution"] = m.execution_report

    faiss_stats = get_faiss_manager().get_stats()

    return HealthResponse(
        status="ok",
        version=__version__,
        models=models_info,
        gpu=gpu,
        faiss=faiss_stats,
    )


@router.get("/ready")
def ready() -> dict[str, str]:
    if not models_loaded():
        return {"status": "not_ready"}
    return {"status": "ready"}

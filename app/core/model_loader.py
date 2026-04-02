"""Load InsightFace (buffalo_l) and optional YOLOv8-face ONNX at startup."""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)


@dataclass
class LoadedModels:
    """Holds references to loaded ML models."""

    face_analysis: Any  # insightface.app.FaceAnalysis
    yolo_face: Optional[Any]  # onnxruntime.InferenceSession or None
    yolo_onnx_path: Optional[Path]
    ctx_id: int
    det_size: tuple[int, int]
    # Populated at load time: ORT version, providers per submodule, GPU vs CPU summary
    execution_report: dict[str, Any] = field(default_factory=dict)


_models: Optional[LoadedModels] = None


def get_models() -> LoadedModels:
    if _models is None:
        raise RuntimeError("Models not loaded; call load_models() at startup")
    return _models


def models_loaded() -> bool:
    return _models is not None


def _try_import_insightface():
    from insightface.app import FaceAnalysis

    return FaceAnalysis


def _onnx_runtime_diagnostics() -> dict[str, Any]:
    """Snapshot of ONNX Runtime install (used to explain CPU vs GPU)."""
    import onnxruntime as ort

    avail = ort.get_available_providers()
    cuda_ok = "CUDAExecutionProvider" in avail
    return {
        "onnxruntime_version": ort.__version__,
        "available_execution_providers": avail,
        "cuda_execution_provider_available": cuda_ok,
        "note": (
            "GPU inference requires onnxruntime-gpu with CUDA/cuDNN matching your driver. "
            "If only CPUExecutionProvider appears, ORT runs on CPU."
        ),
    }


def _log_onnx_runtime_environment() -> dict[str, Any]:
    info = _onnx_runtime_diagnostics()
    logger.info(
        "[ORT] onnxruntime=%s | available providers=%s | CUDA EP present=%s",
        info["onnxruntime_version"],
        info["available_execution_providers"],
        info["cuda_execution_provider_available"],
    )
    if not info["cuda_execution_provider_available"]:
        logger.warning(
            "[ORT] CUDAExecutionProvider is NOT available — InsightFace/YOLO ONNX will use CPU "
            "(install onnxruntime-gpu on a CUDA machine to enable GPU)."
        )
    return info


def _session_providers_summary(session: Any) -> dict[str, Any]:
    """Best-effort: active ORT providers + input name for a session."""
    out: dict[str, Any] = {}
    try:
        out["active_providers"] = list(session.get_providers())
    except Exception as e:
        out["active_providers_error"] = str(e)
    try:
        inp = session.get_inputs()[0]
        out["input_name"] = inp.name
        out["input_shape"] = [str(d) for d in inp.shape]
    except Exception:
        pass
    return out


def _log_insightface_sessions(face_app: Any, ctx_id: int) -> dict[str, Any]:
    """Log each buffalo_l submodule's ONNX Runtime session after prepare()."""
    sub: dict[str, Any] = {
        "ctx_id": ctx_id,
        "insightface_cpu_forced": ctx_id < 0,
    }
    if ctx_id < 0:
        logger.info(
            "[InsightFace] ctx_id=%s (<0) — library forces CPUExecutionProvider on all submodels.",
            ctx_id,
        )
    else:
        logger.info(
            "[InsightFace] ctx_id=%s (>=0) — GPU sessions used when CUDA EP is available.",
            ctx_id,
        )

    models = getattr(face_app, "models", None)
    if not isinstance(models, dict):
        sub["submodels"] = {}
        logger.warning("[InsightFace] Could not introspect face_app.models for session logging.")
        return sub

    submodels: dict[str, Any] = {}
    for taskname, model in models.items():
        sess = getattr(model, "session", None)
        if sess is None:
            submodels[taskname] = {"error": "no session attribute"}
            continue
        detail = _session_providers_summary(sess)
        submodels[taskname] = detail
        prov = detail.get("active_providers", [])
        backend = "GPU" if any("CUDA" in p for p in prov) else "CPU"
        logger.info(
            "[InsightFace] submodel=%s | backend=%s | active_providers=%s",
            taskname,
            backend,
            prov,
        )

    sub["submodels"] = submodels
    return sub


def _load_yolo_face(onnx_path: Path) -> tuple[Optional[Any], dict[str, Any]]:
    """Load YOLOv8-face ONNX via ONNX Runtime; return session and diagnostics dict."""
    if not onnx_path.is_file():
        logger.warning("YOLOv8 face ONNX not found at %s — query pipeline will be limited", onnx_path)
        return None, {"loaded": False, "reason": "file_missing", "path": str(onnx_path)}
    try:
        import onnxruntime as ort

        avail = ort.get_available_providers()
        preferred = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        providers = [p for p in preferred if p in avail] or avail
        sess = ort.InferenceSession(str(onnx_path), providers=providers)
        detail = _session_providers_summary(sess)
        detail["loaded"] = True
        detail["path"] = str(onnx_path)
        prov = detail.get("active_providers", [])
        backend = "GPU" if any("CUDA" in p for p in prov) else "CPU"
        logger.info(
            "[YOLOv8-face] ONNX session | backend=%s | active_providers=%s | path=%s",
            backend,
            prov,
            onnx_path,
        )
        return sess, detail
    except Exception:
        logger.exception("Failed to load YOLOv8-face from %s", onnx_path)
        return None, {"loaded": False, "error": True, "path": str(onnx_path)}


def load_models(settings: Optional[Settings] = None) -> LoadedModels:
    """Initialize InsightFace buffalo_l and optional YOLOv8-face."""
    global _models
    if _models is not None:
        return _models

    settings = settings or get_settings()
    model_dir = settings.resolved_model_dir()
    model_dir.mkdir(parents=True, exist_ok=True)

    # InsightFace prints to stdout during model load; keep logs readable.
    ort_info = _log_onnx_runtime_environment()

    FaceAnalysis = _try_import_insightface()
    root = str(model_dir / "insightface")

    # InsightFace requests CUDA first; on CPU-only onnxruntime this emits a benign UserWarning.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*CUDAExecutionProvider.*",
            category=UserWarning,
            module=r"onnxruntime\.capi\.onnxruntime_inference_collection",
        )
        face_app = FaceAnalysis(name=settings.ingest_det_model, root=root)

    ctx_id = settings.resolved_ctx_id()
    det_size = settings.det_size()
    logger.info(
        "[Config] FACE_CTX_ID=%s -> resolved ctx_id=%d (%s)",
        settings.face_ctx_id,
        ctx_id,
        "GPU" if ctx_id >= 0 else "CPU",
    )
    face_app.prepare(ctx_id=ctx_id, det_size=det_size)

    if_info = _log_insightface_sessions(face_app, ctx_id)

    yolo_path = model_dir / settings.yolov8_face_onnx
    yolo, yolo_info = _load_yolo_face(yolo_path)

    summary_backend = {
        "insightface": (
            "CPU (ctx_id<0 forces CPU)"
            if ctx_id < 0
            else (
                "GPU-capable if CUDA EP available"
                if ort_info.get("cuda_execution_provider_available")
                else "CPU (no CUDA EP)"
            )
        ),
        "yolov8_face": (
            "GPU"
            if yolo is not None
            and any("CUDA" in p for p in (yolo.get_providers() if yolo else []))
            else ("CPU" if yolo is not None else "not_loaded")
        ),
    }
    logger.info(
        "[Summary] Expected inference backends — insightface: %s | yolov8_face: %s",
        summary_backend["insightface"],
        summary_backend["yolov8_face"],
    )

    execution_report: dict[str, Any] = {
        "onnxruntime": ort_info,
        "insightface_buffalo_l": if_info,
        "yolov8_face": yolo_info,
        "summary": summary_backend,
    }

    _models = LoadedModels(
        face_analysis=face_app,
        yolo_face=yolo,
        yolo_onnx_path=yolo_path if yolo_path.is_file() else None,
        ctx_id=ctx_id,
        det_size=det_size,
        execution_report=execution_report,
    )
    logger.info(
        "Models ready: buffalo_l root=%s ctx_id=%s det_size=%s yolo=%s",
        root,
        ctx_id,
        det_size,
        yolo_path.name if yolo is not None else "off",
    )
    return _models


def unload_models() -> None:
    global _models
    _models = None


def gpu_memory_info() -> dict[str, Optional[float]]:
    """Return used/total GPU memory in MiB if nvidia-smi is available."""
    import shutil
    import subprocess

    if not shutil.which("nvidia-smi"):
        return {"available": False, "used_mib": None, "total_mib": None}

    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        )
        line = out.strip().splitlines()[0]
        parts = [p.strip() for p in line.split(",")]
        used, total = float(parts[0]), float(parts[1])
        return {"available": True, "used_mib": used, "total_mib": total}
    except Exception:
        return {"available": True, "used_mib": None, "total_mib": None}


def dummy_face_probe(face_app: Any) -> bool:
    """Run a tiny tensor through the stack to verify recognition works."""
    try:
        img = np.zeros((640, 640, 3), dtype=np.uint8)
        faces = face_app.get(img)
        return isinstance(faces, list)
    except Exception:
        logger.exception("Face model probe failed")
        return False

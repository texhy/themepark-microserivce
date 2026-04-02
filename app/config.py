"""Application configuration from environment / .env."""

from functools import lru_cache
from pathlib import Path
from typing import Tuple

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=("settings_",),
    )

    app_name: str = Field(default="theme-park-msp", validation_alias="APP_NAME")
    app_env: str = Field(default="development", validation_alias="APP_ENV")
    host: str = Field(default="0.0.0.0", validation_alias="HOST")
    port: int = Field(default=8000, validation_alias="PORT")

    # Paths (resolved relative to project root if not absolute)
    faiss_index_dir: str = Field(default="data/faiss_indexes", validation_alias="FAISS_INDEX_DIR")
    model_dir: str = Field(default="models", validation_alias="MODEL_DIR")

    # InsightFace: 0 = first GPU, -1 = CPU, "auto" = detect at runtime
    # Default "auto": uses GPU (0) if CUDAExecutionProvider is available, else CPU (-1).
    face_ctx_id: str = Field(default="auto", validation_alias="FACE_CTX_ID")

    def resolved_ctx_id(self) -> int:
        """Return integer ctx_id: 0+ for GPU, -1 for CPU. 'auto' probes ORT."""
        raw = self.face_ctx_id.strip().lower()
        if raw == "auto":
            try:
                import onnxruntime as ort
                if "CUDAExecutionProvider" in ort.get_available_providers():
                    return 0
            except Exception:
                pass
            return -1
        return int(raw)
    det_size_width: int = Field(default=640, validation_alias="DET_SIZE_WIDTH")
    det_size_height: int = Field(default=640, validation_alias="DET_SIZE_HEIGHT")

    ingest_det_model: str = Field(default="buffalo_l", validation_alias="INGEST_DET_MODEL")
    query_det_model: str = Field(default="yolov8n", validation_alias="QUERY_DET_MODEL")
    yolov8_face_onnx: str = Field(
        default="yolov8n-face.onnx",
        validation_alias="YOLOV8_FACE_ONNX",
    )

    similarity_threshold: float = Field(default=0.45, validation_alias="SIMILARITY_THRESHOLD")
    top_k: int = Field(default=20, validation_alias="TOP_K")

    cors_origins: str = Field(default="*", validation_alias="CORS_ORIGINS")

    def resolved_faiss_index_dir(self) -> Path:
        p = Path(self.faiss_index_dir)
        if not p.is_absolute():
            p = _project_root() / p
        return p

    def resolved_model_dir(self) -> Path:
        p = Path(self.model_dir)
        if not p.is_absolute():
            p = _project_root() / p
        return p

    def det_size(self) -> Tuple[int, int]:
        return (self.det_size_width, self.det_size_height)

    def cors_origin_list(self) -> list[str]:
        raw = self.cors_origins.strip()
        if raw == "*":
            return ["*"]
        return [o.strip() for o in raw.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()

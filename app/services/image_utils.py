"""Image decode, resize, validation utilities."""

from __future__ import annotations

import logging
from io import BytesIO

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

MAX_IMAGE_BYTES = 20 * 1024 * 1024  # 20 MB
SUPPORTED_MIME = {"image/jpeg", "image/png", "image/webp", "image/bmp"}


def validate_content_type(content_type: str | None) -> str | None:
    """Return an error message if the content type is unsupported, else None."""
    if not content_type:
        return "Missing Content-Type header"
    base = content_type.split(";")[0].strip().lower()
    if base not in SUPPORTED_MIME:
        return f"Unsupported image type '{base}'. Accepted: {', '.join(sorted(SUPPORTED_MIME))}"
    return None


def decode_image_bytes(raw: bytes) -> np.ndarray:
    """Decode raw bytes → BGR numpy array (OpenCV convention).

    Raises ValueError on corrupt / unsupported data.
    """
    import time

    if len(raw) > MAX_IMAGE_BYTES:
        raise ValueError(f"Image too large ({len(raw)} bytes, max {MAX_IMAGE_BYTES})")

    _t0 = time.perf_counter()

    try:
        pil_img = Image.open(BytesIO(raw))
        pil_img.verify()
        pil_img = Image.open(BytesIO(raw))  # re-open after verify()
    except Exception as exc:
        raise ValueError(f"Cannot decode image: {exc}") from exc

    _t_open = time.perf_counter()

    if pil_img.mode == "RGBA":
        pil_img = pil_img.convert("RGB")
    elif pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")

    arr = np.asarray(pil_img, dtype=np.uint8)
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    _t_done = time.perf_counter()

    logger.info(
        "[Decode] size=%dx%d bytes=%d open+verify=%.1fms convert=%.1fms total=%.1fms",
        bgr.shape[1], bgr.shape[0], len(raw),
        (_t_open - _t0) * 1000,
        (_t_done - _t_open) * 1000,
        (_t_done - _t0) * 1000,
    )

    return bgr

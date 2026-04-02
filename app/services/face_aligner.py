"""Face alignment via 5-point landmark affine transformation.

InsightFace handles alignment internally when you call `face_app.get()`,
but this module exposes standalone alignment for the YOLOv8-face detection
path (Phase 3) where we detect with YOLO but still need ArcFace-compatible
112×112 crops.
"""

from __future__ import annotations

import cv2
import numpy as np

# ArcFace reference landmarks for a 112×112 output (standard 5-point template).
ARCFACE_REF_LANDMARKS = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)

OUTPUT_SIZE = (112, 112)


def align_face(
    image: np.ndarray,
    landmarks_5: np.ndarray,
    output_size: tuple[int, int] = OUTPUT_SIZE,
) -> np.ndarray:
    """Warp a face crop to a canonical 112×112 pose using 5-point landmarks.

    Args:
        image: Full BGR frame (uint8).
        landmarks_5: shape (5, 2) — left eye, right eye, nose, left mouth,
                      right mouth in pixel coordinates.
        output_size: Target (w, h). Default 112×112 for ArcFace.

    Returns:
        Aligned face crop as BGR uint8 array of shape (output_size[1], output_size[0], 3).
    """
    src = np.asarray(landmarks_5, dtype=np.float32).reshape(5, 2)
    dst = ARCFACE_REF_LANDMARKS.copy()

    if output_size != (112, 112):
        scale_x = output_size[0] / 112.0
        scale_y = output_size[1] / 112.0
        dst[:, 0] *= scale_x
        dst[:, 1] *= scale_y

    # estimateAffinePartial2D gives a similarity transform (rotation + uniform
    # scale + translation) which is what ArcFace expects.
    tform, _ = cv2.estimateAffinePartial2D(src, dst)
    if tform is None:
        tform = cv2.getAffineTransform(src[:3], dst[:3])

    aligned = cv2.warpAffine(
        image,
        tform,
        output_size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return aligned

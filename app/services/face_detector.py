"""Face detection for both ingestion (SCRFD) and query (YOLOv8-face) paths."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np

from app.core.model_loader import get_models

logger = logging.getLogger(__name__)

DEFAULT_CONFIDENCE_THRESHOLD = 0.5


@dataclass
class DetectedFace:
    """Single detected face with all metadata needed downstream."""

    bbox: list[float]           # [x1, y1, x2, y2] in pixel coords
    confidence: float
    landmarks: np.ndarray       # shape (5, 2) — 5 key-points
    aligned_face: np.ndarray    # 112×112 aligned crop (BGR, uint8)
    embedding: np.ndarray       # 512-dim L2-normalized ArcFace vector


# ═══════════════════════════════════════════════════════════════════
#  SCRFD Detector  — ingestion (photographer) path
# ═══════════════════════════════════════════════════════════════════

class ScrfdDetector:
    """Wraps InsightFace FaceAnalysis for the ingestion (photographer) path.

    Uses the SCRFD det_10g model inside buffalo_l for detection,
    and w600k_r50 ArcFace for embedding — both already loaded in
    `model_loader.LoadedModels.face_analysis`.
    """

    def __init__(self, confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD):
        self._threshold = confidence_threshold

    def detect(self, image: np.ndarray) -> List[DetectedFace]:
        """Run detection + alignment + embedding on a single image.

        Args:
            image: BGR uint8 numpy array of any resolution.

        Returns:
            List of DetectedFace, one per face above the confidence threshold.
            The list may be empty if no faces are found.
        """
        models = get_models()
        face_app = models.face_analysis

        raw_faces = face_app.get(image)

        results: list[DetectedFace] = []
        for face in raw_faces:
            score = float(face.det_score)
            if score < self._threshold:
                continue

            bbox = [float(v) for v in face.bbox]

            kps = getattr(face, "kps", np.zeros((5, 2), dtype=np.float32))

            aligned = getattr(face, "normed_face", None)
            if aligned is None:
                aligned = np.zeros((112, 112, 3), dtype=np.uint8)

            embedding = getattr(face, "normed_embedding", None)
            if embedding is None:
                embedding = getattr(face, "embedding", None)
            if embedding is None:
                embedding = np.zeros(512, dtype=np.float32)
            else:
                embedding = np.asarray(embedding, dtype=np.float32)
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm

            results.append(
                DetectedFace(
                    bbox=bbox,
                    confidence=score,
                    landmarks=kps,
                    aligned_face=aligned,
                    embedding=embedding,
                )
            )

        logger.debug(
            "SCRFD detected %d faces (%d above threshold %.2f)",
            len(raw_faces),
            len(results),
            self._threshold,
        )
        return results


# ═══════════════════════════════════════════════════════════════════
#  YOLOv8-face Detector  — query (selfie / kiosk) path
# ═══════════════════════════════════════════════════════════════════

class YoloFaceDetector:
    """YOLOv8n-face ONNX detector optimized for single-face selfie detection.

    Full decode pipeline: letterbox → ONNX inference → DFL decode → NMS →
    scale back to original coords → align → ArcFace embed.

    No torch/ultralytics dependency — pure numpy + cv2.
    """

    STRIDES = [8, 16, 32]
    REG_MAX = 16
    INPUT_SIZE = (640, 640)  # (H, W)

    def __init__(
        self,
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.45,
    ):
        self._conf_threshold = conf_threshold
        self._iou_threshold = iou_threshold

    # ── Public API ───────────────────────────────────────────────

    def detect_best_face(
        self,
        image: np.ndarray,
    ) -> Optional[DetectedFace]:
        """Detect faces and return the single best one (for selfie/kiosk).

        Selection: highest (area × confidence) — favours large, confident faces.

        Returns None if no face is found above the confidence threshold.
        """
        from app.services.face_aligner import align_face
        from app.services.face_embedder import ArcFaceEmbedder

        models = get_models()
        session = models.yolo_face
        if session is None:
            logger.warning("YOLOv8-face ONNX not loaded — falling back to SCRFD")
            scrfd = ScrfdDetector(confidence_threshold=self._conf_threshold)
            faces = scrfd.detect(image)
            if not faces:
                return None
            return max(faces, key=lambda f: _bbox_area(f.bbox) * f.confidence)

        letterboxed, scale, (dw, dh) = self._letterbox(image)
        input_tensor = self._preprocess(letterboxed)
        outputs = session.run(None, {session.get_inputs()[0].name: input_tensor})

        boxes, scores, landmarks = self._postprocess(outputs)

        if len(boxes) == 0:
            return None

        self._scale_to_original(boxes, landmarks, scale, dw, dh, image.shape[:2])

        best_idx = int(np.argmax(
            [(x2 - x1) * (y2 - y1) * s for (x1, y1, x2, y2), s in zip(boxes, scores)]
        ))

        bbox = boxes[best_idx].tolist()
        conf = float(scores[best_idx])
        lm5 = landmarks[best_idx].reshape(5, 2)

        aligned = align_face(image, lm5)
        embedder = ArcFaceEmbedder()
        embedding = embedder.embed(aligned)

        return DetectedFace(
            bbox=bbox,
            confidence=conf,
            landmarks=lm5,
            aligned_face=aligned,
            embedding=embedding,
        )

    def detect_all(self, image: np.ndarray) -> List[DetectedFace]:
        """Detect all faces (used less often; search needs detect_best_face)."""
        from app.services.face_aligner import align_face
        from app.services.face_embedder import ArcFaceEmbedder

        models = get_models()
        session = models.yolo_face
        if session is None:
            return ScrfdDetector(confidence_threshold=self._conf_threshold).detect(image)

        letterboxed, scale, (dw, dh) = self._letterbox(image)
        input_tensor = self._preprocess(letterboxed)
        outputs = session.run(None, {session.get_inputs()[0].name: input_tensor})

        boxes, scores, landmarks = self._postprocess(outputs)

        if len(boxes) == 0:
            return []

        self._scale_to_original(boxes, landmarks, scale, dw, dh, image.shape[:2])

        embedder = ArcFaceEmbedder()
        results: list[DetectedFace] = []
        for i in range(len(boxes)):
            lm5 = landmarks[i].reshape(5, 2)
            aligned = align_face(image, lm5)
            embedding = embedder.embed(aligned)
            results.append(DetectedFace(
                bbox=boxes[i].tolist(),
                confidence=float(scores[i]),
                landmarks=lm5,
                aligned_face=aligned,
                embedding=embedding,
            ))
        return results

    # ── Pre-processing ───────────────────────────────────────────

    @classmethod
    def _letterbox(
        cls,
        image: np.ndarray,
        color: tuple[int, int, int] = (114, 114, 114),
    ) -> tuple[np.ndarray, float, tuple[float, float]]:
        """Resize + pad to INPUT_SIZE preserving aspect ratio."""
        h, w = image.shape[:2]
        th, tw = cls.INPUT_SIZE
        scale = min(th / h, tw / w)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        dw = (tw - new_w) / 2
        dh = (th - new_h) / 2
        top, bottom = int(dh), int(th - new_h - int(dh))
        left, right = int(dw), int(tw - new_w - int(dw))
        padded = cv2.copyMakeBorder(
            resized, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=color,
        )
        return padded, scale, (dw, dh)

    @staticmethod
    def _preprocess(img: np.ndarray) -> np.ndarray:
        """BGR HWC uint8 → RGB CHW float32 [0,1] with batch dim."""
        img = img[:, :, ::-1].astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))[np.newaxis]
        return np.ascontiguousarray(img)

    # ── Post-processing ──────────────────────────────────────────

    def _postprocess(
        self,
        predictions: list[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Decode 3 multi-scale feature maps → (boxes, scores, landmarks).

        Each feature map is (1, 80, H, W):
          64 channels  DFL bbox distribution (4 sides × 16 bins)
           1 channel   face class logit
          15 channels  5 keypoints × (x, y, visibility)
        """
        all_boxes, all_scores, all_lm = [], [], []

        for pred, stride in zip(predictions, self.STRIDES):
            _, channels, fh, fw = pred.shape
            pred = pred.reshape(channels, -1).T  # (H*W, 80)

            # Anchor grid (0.5 offset = center of each grid cell)
            gy, gx = np.meshgrid(
                np.arange(fh, dtype=np.float32) + 0.5,
                np.arange(fw, dtype=np.float32) + 0.5,
                indexing="ij",
            )
            gx = gx.ravel()
            gy = gy.ravel()

            # DFL box decode
            dfl = pred[:, :64].reshape(-1, 4, self.REG_MAX)
            dfl = _softmax(dfl, axis=-1) @ np.arange(self.REG_MAX, dtype=np.float32)
            x1 = (gx - dfl[:, 0]) * stride
            y1 = (gy - dfl[:, 1]) * stride
            x2 = (gx + dfl[:, 2]) * stride
            y2 = (gy + dfl[:, 3]) * stride
            boxes = np.stack([x1, y1, x2, y2], axis=-1)

            # Confidence (sigmoid)
            scores = _sigmoid(pred[:, 64])

            # Keypoints: 5 × (x, y, vis)
            kpt_raw = pred[:, 65:].reshape(-1, 5, 3)
            kpt_gx = np.arange(fw, dtype=np.float32)[np.newaxis, :].repeat(fh, axis=0).ravel()
            kpt_gy = np.arange(fh, dtype=np.float32)[:, np.newaxis].repeat(fw, axis=1).ravel()
            kpt_x = (kpt_raw[:, :, 0] * 2.0 + kpt_gx[:, None]) * stride
            kpt_y = (kpt_raw[:, :, 1] * 2.0 + kpt_gy[:, None]) * stride
            lm = np.stack([kpt_x, kpt_y], axis=-1).reshape(-1, 10)

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_lm.append(lm)

        boxes = np.concatenate(all_boxes)
        scores = np.concatenate(all_scores)
        landmarks = np.concatenate(all_lm)

        # Confidence filter
        mask = scores >= self._conf_threshold
        boxes = boxes[mask]
        scores = scores[mask]
        landmarks = landmarks[mask]

        if len(boxes) == 0:
            return np.empty((0, 4)), np.empty(0), np.empty((0, 10))

        # NMS (pure numpy)
        keep = _nms_numpy(boxes, scores, self._iou_threshold)
        return boxes[keep], scores[keep], landmarks[keep]

    # ── Coordinate rescaling ─────────────────────────────────────

    @staticmethod
    def _scale_to_original(
        boxes: np.ndarray,
        landmarks: np.ndarray,
        scale: float,
        dw: float,
        dh: float,
        orig_hw: tuple[int, int],
    ) -> None:
        """Undo letterbox: translate by -pad, divide by scale, clip."""
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - dw) / scale
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - dh) / scale
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_hw[1])
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_hw[0])

        landmarks[:, 0::2] = (landmarks[:, 0::2] - dw) / scale
        landmarks[:, 1::2] = (landmarks[:, 1::2] - dh) / scale


# ═══════════════════════════════════════════════════════════════════
#  Shared helpers
# ═══════════════════════════════════════════════════════════════════

def _bbox_area(bbox: list[float]) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def _nms_numpy(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float,
) -> np.ndarray:
    """Pure-numpy Non-Maximum Suppression."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep: list[int] = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[1:][iou <= iou_threshold]

    return np.array(keep, dtype=np.intp)

#!/usr/bin/env python3
"""Download InsightFace buffalo_l and YOLOv8-face ONNX into ./models/."""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

# Project root = parent of scripts/
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Pre-exported YOLOv8-face ONNX (yakhyo/yolov8-face-onnx-inference releases)
YOLOV8_FACE_ONNX_URL = (
    "https://github.com/yakhyo/yolov8-face-onnx-inference/releases/download/weights/yolov8n-face.onnx"
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_checksum(path: Path, digest: str) -> None:
    path.with_suffix(path.suffix + ".sha256").write_text(digest + "  " + path.name + "\n", encoding="utf-8")


def download_url(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {dest}")
    req = urllib.request.Request(url, headers={"User-Agent": "theme-park-msp-download-models/1.0"})
    with urllib.request.urlopen(req, timeout=600) as resp, dest.open("wb") as out:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)


def ensure_buffalo_l(model_dir: Path) -> None:
    """Trigger InsightFace to download buffalo_l under model_dir/insightface."""
    from insightface.app import FaceAnalysis

    root = str(model_dir / "insightface")
    print(f"Preparing InsightFace buffalo_l (root={root}) …")
    app = FaceAnalysis(name="buffalo_l", root=root)
    app.prepare(ctx_id=-1, det_size=(640, 640))
    print("buffalo_l ready.")


def ensure_yolov8_face_onnx(model_dir: Path, onnx_name: str) -> None:
    onnx_path = model_dir / onnx_name
    if onnx_path.is_file():
        print(f"YOLOv8 ONNX already exists: {onnx_path}")
        write_checksum(onnx_path, sha256_file(onnx_path))
        return

    download_url(YOLOV8_FACE_ONNX_URL, onnx_path)
    print(f"ONNX written: {onnx_path}")
    write_checksum(onnx_path, sha256_file(onnx_path))


def main() -> int:
    parser = argparse.ArgumentParser(description="Download ML weights for theme-park-msp")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=ROOT / "models",
        help="Directory for weights (default: ./models)",
    )
    parser.add_argument(
        "--skip-yolo",
        action="store_true",
        help="Only download buffalo_l",
    )
    parser.add_argument(
        "--skip-insightface",
        action="store_true",
        help="Only download YOLOv8-face ONNX",
    )
    args = parser.parse_args()
    model_dir: Path = args.model_dir
    model_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_insightface:
        ensure_buffalo_l(model_dir)
    if not args.skip_yolo:
        ensure_yolov8_face_onnx(model_dir, "yolov8n-face.onnx")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

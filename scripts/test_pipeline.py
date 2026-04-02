#!/usr/bin/env python3
"""Smoke test: import app and print health (requires server or inline check)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.core.model_loader import load_models, unload_models
from app.config import get_settings


def main() -> int:
    settings = get_settings()
    try:
        load_models(settings)
        from app.core.model_loader import get_models

        m = get_models()
        print("buffalo_l:", m.face_analysis is not None)
        print("yolo:", m.yolo_face is not None)
    finally:
        unload_models()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

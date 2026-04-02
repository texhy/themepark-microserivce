# theme-park-msp

Python microservice for theme park face recognition (FastAPI + InsightFace + FAISS).

## GPU / CPU auto-detection

`FACE_CTX_ID=auto` (the default) probes ONNX Runtime at startup:
- If `CUDAExecutionProvider` is present → GPU (ctx_id=0)
- Otherwise → CPU (ctx_id=-1)

No manual config needed. It will use GPU on RunPod and CPU on your laptop automatically.

## Local dev setup (CPU)

```bash
cd theme-park-msp
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt     # onnxruntime (CPU)
cp .env.example .env
python scripts/download_models.py   # buffalo_l + yolov8n-face.onnx
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## RunPod deployment

### Docker build & push

```bash
docker build -t theme-park-msp .
docker tag theme-park-msp your-registry/theme-park-msp:latest
docker push your-registry/theme-park-msp:latest
```

### RunPod pod configuration

| Setting | Value |
| --- | --- |
| GPU | RTX 4090 (24 GB VRAM) |
| Container image | `your-registry/theme-park-msp:latest` |
| Network volume | 10–20 GB, EU region (closest to VPS) |
| Expose port | TCP 8000 |

### Environment variables (set in RunPod pod template)

```
FACE_CTX_ID=auto
MODEL_DIR=/runpod-volume/models
FAISS_INDEX_DIR=/runpod-volume/faiss_indexes
```

### First-time setup on a new network volume

Models are **auto-downloaded on first startup** if missing from `MODEL_DIR`.
Alternatively, SSH into the pod and run manually:

```bash
python scripts/download_models.py --model-dir /runpod-volume/models
```

### Startup sequence

1. Directories created (`MODEL_DIR`, `FAISS_INDEX_DIR`)
2. Model weights checked — auto-downloaded if missing
3. InsightFace (buffalo_l) + YOLOv8-face loaded into **GPU VRAM**
4. All FAISS indexes loaded from disk into **RAM**
5. Server ready — all inference is warm, no cold-start on first request

### Shutdown

Graceful shutdown flushes all FAISS indexes back to the network volume.

## Endpoints

| Endpoint   | Description                        |
| ---------- | ---------------------------------- |
| `/docs`    | Swagger UI                         |
| `/health`  | Model load status, GPU info, ONNX Runtime providers per submodel |
| `/ready`   | Simple readiness probe             |
| `/ingest/` | Pipeline 1 — ingestion (Phase 2)   |
| `/search/` | Pipeline 2 — face search (Phase 3) |

## Startup logs to confirm GPU

Look for these lines after `uvicorn` starts:

```
[ORT] onnxruntime=1.20.1 | available providers=['CUDAExecutionProvider', 'CPUExecutionProvider'] | CUDA EP present=True
[Config] FACE_CTX_ID=auto -> resolved ctx_id=0 (GPU)
[InsightFace] submodel=detection  | backend=GPU | active_providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
[InsightFace] submodel=recognition| backend=GPU | active_providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
[YOLOv8-face] ONNX session        | backend=GPU | active_providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
[Summary] Expected inference backends — insightface: GPU | yolov8_face: GPU
```

If you see `CPU` instead, check:
1. Is `onnxruntime-gpu` installed (not `onnxruntime`)?
2. Does `nvidia-smi` work?
3. Do CUDA and cuDNN versions match the ORT build?

## Files

| File | Role |
| ---- | ---- |
| `requirements.txt` | CPU deps (local dev) |
| `requirements-gpu.txt` | GPU deps (RunPod — `onnxruntime-gpu`) |
| `Dockerfile` | GPU image (`nvidia/cuda` base) |
| `Dockerfile.cpu` | CPU image (slim Python) |
| `.env.example` | Config template (`FACE_CTX_ID=auto`) |

## Smoke test

```bash
python scripts/test_pipeline.py
```

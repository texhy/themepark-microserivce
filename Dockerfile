# ---- GPU image for RunPod (CUDA 12.x + cuDNN 9) ----
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    libgl1 libglib2.0-0 \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-gpu.txt .
RUN pip install --no-cache-dir -r requirements-gpu.txt

COPY app ./app
COPY scripts ./scripts

# Default dirs inside the container (overridden by env vars on RunPod).
# On RunPod, set MODEL_DIR and FAISS_INDEX_DIR to network volume paths.
RUN mkdir -p /app/data/faiss_indexes /app/models

ENV PYTHONUNBUFFERED=1
ENV FACE_CTX_ID=auto

# RunPod network volume paths (override in pod env or .env):
#   MODEL_DIR=/runpod-volume/models
#   FAISS_INDEX_DIR=/runpod-volume/faiss_indexes

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

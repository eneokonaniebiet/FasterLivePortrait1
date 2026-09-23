# syntax=docker/dockerfile:1.7

# NOTE:
# Render web services provide Docker execution, but the current Render
# compute plans documented by Render are CPU/RAM plans. A CUDA image does
# not provide an NVIDIA GPU by itself. This image therefore supports the
# FasterLivePortrait CUDA/TensorRT runtime when deployed on GPU-capable
# infrastructure, but must not be represented as "30 FPS guaranteed" on
# a CPU-only Render instance.

FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/data/huggingface \
    FLIP_CHECKPOINT_DIR=/data/checkpoints \
    PORT=10000

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev git ffmpeg libgl1 libglib2.0-0 libsm6 libxext6 \
    libxrender1 libgomp1 ca-certificates curl && \
    rm -rf /var/lib/apt/lists/* && \
    python3 -m pip install --upgrade pip setuptools wheel

WORKDIR /app

COPY requirements.txt /tmp/requirements.txt

# PyTorch CUDA 11.8 wheels. FasterLivePortrait's remaining dependencies
# are installed from its upstream requirements plus FastAPI runtime deps.
RUN python3 -m pip install --no-cache-dir \
    torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118 && \
    python3 -m pip install --no-cache-dir -r /tmp/requirements.txt \
    fastapi uvicorn[standard] python-multipart

COPY . /app

# Render supplies PORT at runtime. One worker is intentional because the
# FasterLivePortrait pipeline contains stateful motion/smoothing state.
EXPOSE 10000

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
  CMD curl -fsS http://127.0.0.1:10000/health || exit 1

CMD ["python3", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "10000", "--workers", "1"]

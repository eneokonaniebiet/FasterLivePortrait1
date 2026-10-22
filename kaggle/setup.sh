#!/bin/bash
# One-time setup for a free Kaggle/Colab T4.
# Kaggle cell:  !bash /kaggle/working/FasterLivePortrait1/kaggle/setup.sh
set -e
ROOT=/kaggle/working
[ -d "$ROOT" ] || ROOT=/content
cd "$ROOT/FasterLivePortrait1"

# Python deps. pycuda/TensorRT are only needed for the TRT path, so failures there are not fatal.
grep -viE "^(pycuda|kokoro|misaki|phonemizer|soundfile|munch)" requirements.txt > /tmp/req_live.txt
pip install -q -r /tmp/req_live.txt
pip install -q onnxruntime-gpu fastapi "uvicorn[standard]" python-multipart || true
pip install -q pycuda || echo "pycuda not installed (ok for ONNX mode)"

# Model weights
huggingface-cli download warmshao/FasterLivePortrait --local-dir "$ROOT/checkpoints"

# Free public https tunnel (needed so the browser camera is allowed)
wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O /tmp/cloudflared
chmod +x /tmp/cloudflared

echo "Setup done."

"""Start the live server (app.py) + a free public tunnel from a Kaggle/Colab notebook.
Kaggle cell:  %run /kaggle/working/FasterLivePortrait1/kaggle/start.py
"""
import os
import re
import subprocess
import sys
import threading

BASE = "/kaggle/working" if os.path.isdir("/kaggle/working") else "/content"
REPO = os.path.join(BASE, "FasterLivePortrait1")
os.chdir(REPO)
sys.path.insert(0, REPO)

os.environ.setdefault("FLIP_CONFIG", "configs/onnx_infer.yaml")
os.environ.setdefault("FLIP_CHECKPOINT_DIR", os.path.join(BASE, "checkpoints"))
os.environ.setdefault("PORT", "8000")

tunnel = subprocess.Popen(
    ["/tmp/cloudflared", "tunnel", "--url", f"http://localhost:{os.environ['PORT']}"],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
)

def watch():
    for line in tunnel.stdout:
        m = re.search(r"https://[-a-z0-9]+\.trycloudflare\.com", line)
        if m:
            print("\n>>> OPEN THIS LINK:", m.group(0), "\n")
            return

threading.Thread(target=watch, daemon=True).start()

import uvicorn
from app import app  # loads the model; takes about a minute

uvicorn.run(app, host="0.0.0.0", port=int(os.environ["PORT"]), log_level="warning")

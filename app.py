import asyncio
import logging
import os
import shutil
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from omegaconf import OmegaConf
from src.pipelines.faster_live_portrait_pipeline import FasterLivePortraitPipeline

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
PORT = int(os.getenv("PORT", "10000"))
HOST = os.getenv("HOST", "0.0.0.0")
CHECKPOINT_DIR = Path(os.getenv("FLIP_CHECKPOINT_DIR", "/data/checkpoints"))
CONFIG_PATH = Path(os.getenv("FLIP_CONFIG", "configs/trt_infer.yaml"))
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "20"))
MAX_FRAME_BYTES = int(os.getenv("MAX_FRAME_BYTES", "500000"))
TARGET_FPS = float(os.getenv("TARGET_FPS", "30"))
ALLOWED_ORIGINS = [x.strip() for x in os.getenv("CORS_ORIGINS", "*").split(",") if x.strip()]

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("faster-liveportrait-api")

PIPELINE = None
PIPELINE_LOCK = asyncio.Lock()
STARTUP_ERROR = None
ACTIVE_SESSION = None
SESSIONS = {}


def _load_pipeline():
    if not CONFIG_PATH.exists():
        raise RuntimeError(f"Missing FasterLivePortrait config: {CONFIG_PATH}")
    cfg = OmegaConf.load(str(CONFIG_PATH))
    for section in ("models", "animal_models"):
        if section not in cfg:
            continue
        for name in cfg[section]:
            model_path = cfg[section][name].get("model_path")
            if isinstance(model_path, str):
                cfg[section][name].model_path = model_path.replace("./checkpoints", str(CHECKPOINT_DIR))
            elif model_path is not None:
                cfg[section][name].model_path = [
                    p.replace("./checkpoints", str(CHECKPOINT_DIR)) for p in model_path
                ]
    cfg.infer_params.flag_pasteback = True
    return FasterLivePortraitPipeline(cfg=cfg, is_animal=False)


@asynccontextmanager
async def lifespan(app):
    global PIPELINE, STARTUP_ERROR
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        log.info("Loading FasterLivePortrait pipeline from %s", CONFIG_PATH)
        PIPELINE = await asyncio.to_thread(_load_pipeline)
        log.info("FasterLivePortrait pipeline ready on %s", PIPELINE.device)
    except Exception as exc:
        STARTUP_ERROR = f"{type(exc).__name__}: {exc}"
        log.exception("Pipeline initialization failed")
    yield
    PIPELINE = None
    SESSIONS.clear()


app = FastAPI(title="FasterLivePortrait Live API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=ALLOWED_ORIGINS != ["*"],
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {
        "status": "ok" if PIPELINE is not None else "degraded",
        "pipeline_ready": PIPELINE is not None,
        "device": str(PIPELINE.device) if PIPELINE else None,
        "target_fps": TARGET_FPS,
        "startup_error": STARTUP_ERROR,
    }


@app.get("/")
async def dashboard():
    return FileResponse("index.html", media_type="text/html")


@app.post("/api/source")
async def upload_source(file: UploadFile = File(...)):
    global ACTIVE_SESSION
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(415, "Upload a portrait image (JPEG/PNG/WebP).")
    if PIPELINE is None:
        raise HTTPException(503, "FasterLivePortrait pipeline is not ready.")

    # The upstream pipeline is stateful. This worker intentionally supports
    # one active realtime portrait session so sources cannot overwrite each other.
    if ACTIVE_SESSION is not None:
        old = SESSIONS.pop(ACTIVE_SESSION, None)
        if old:
            shutil.rmtree(old["dir"], ignore_errors=True)
        ACTIVE_SESSION = None

    session_id = uuid.uuid4().hex
    session_dir = Path(tempfile.mkdtemp(prefix=f"flp-{session_id}-"))
    source_path = session_dir / "source.jpg"
    size = 0

    try:
        with source_path.open("wb") as out:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_MB * 1024 * 1024:
                    raise HTTPException(413, f"Maximum source size is {MAX_UPLOAD_MB} MB.")
                out.write(chunk)

        async with PIPELINE_LOCK:
            ok = await asyncio.to_thread(
                PIPELINE.prepare_source, str(source_path), realtime=True
            )

        if not ok or not PIPELINE.src_imgs or not PIPELINE.src_infos:
            raise HTTPException(422, "No usable face was detected in the source portrait.")

        SESSIONS[session_id] = {
            "dir": str(session_dir),
            "source": str(source_path),
            "created": time.time(),
        }
        ACTIVE_SESSION = session_id
        return {"session_id": session_id, "status": "ready"}
    except HTTPException:
        shutil.rmtree(session_dir, ignore_errors=True)
        raise
    except Exception as exc:
        shutil.rmtree(session_dir, ignore_errors=True)
        log.exception("Source preparation failed")
        raise HTTPException(500, f"Source preparation failed: {exc}") from exc


def _decode_jpeg(data):
    frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Invalid JPEG frame")
    return frame


def _encode_jpeg(frame_rgb):
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return encoded.tobytes()


@app.websocket("/api/stream-live")
async def stream_live(ws: WebSocket):
    global ACTIVE_SESSION
    await ws.accept()

    session_id = ws.query_params.get("session_id")
    if session_id != ACTIVE_SESSION or session_id not in SESSIONS:
        await ws.send_json({"type": "error", "message": "Invalid or expired session_id"})
        await ws.close(code=1008)
        return
    if PIPELINE is None:
        await ws.send_json({"type": "error", "message": "Pipeline is not ready"})
        await ws.close(code=1013)
        return

    first_frame = True
    frames = 0
    started = time.perf_counter()

    try:
        while True:
            message = await ws.receive()
            if message.get("type") == "websocket.disconnect":
                break

            data = message.get("bytes")
            if not data:
                continue
            if len(data) > MAX_FRAME_BYTES:
                await ws.send_json({"type": "error", "message": "Frame too large"})
                continue

            try:
                frame = _decode_jpeg(data)
                async with PIPELINE_LOCK:
                    # FasterLivePortrait's full-pasteback output is returned
                    # when flag_pasteback=True and realtime=False.
                    _, output_crop, output_full, _ = await asyncio.to_thread(
                        PIPELINE.run,
                        frame,
                        PIPELINE.src_imgs[0],
                        PIPELINE.src_infos[0],
                        first_frame=first_frame,
                        realtime=False,
                    )

                first_frame = False
                if output_crop is None or output_full is None:
                    await ws.send_json({"type": "status", "message": "No face detected"})
                    continue

                await ws.send_bytes(await asyncio.to_thread(_encode_jpeg, output_full))
                frames += 1

                elapsed = max(time.perf_counter() - started, 1e-6)
                if frames % 30 == 0:
                    await ws.send_json({
                        "type": "stats",
                        "fps": round(frames / elapsed, 2),
                        "frames": frames,
                    })
            except Exception as exc:
                log.exception("Frame processing error")
                await ws.send_json({"type": "error", "message": str(exc)})
    except WebSocketDisconnect:
        pass
    finally:
        if ACTIVE_SESSION == session_id:
            ACTIVE_SESSION = None
        log.info("WebSocket closed for session %s", session_id)


@app.delete("/api/source/{session_id}")
async def delete_source(session_id: str):
    global ACTIVE_SESSION
    session = SESSIONS.pop(session_id, None)
    if session:
        shutil.rmtree(session["dir"], ignore_errors=True)
    if ACTIVE_SESSION == session_id:
        ACTIVE_SESSION = None
    return JSONResponse({"status": "deleted"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host=HOST, port=PORT, workers=1)

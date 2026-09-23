import asyncio
import base64
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

PIPELINE: FasterLivePortraitPipeline | None = None
PIPELINE_LOCK = asyncio.Lock()
STARTUP_ERROR: str | None = None
SESSIONS: dict[str, dict] = {}


def _prepare_checkpoints_dir() -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


def _load_pipeline() -> FasterLivePortraitPipeline:
    cfg_path = CONFIG_PATH
    if not cfg_path.exists():
        raise RuntimeError(f"Missing FasterLivePortrait config: {cfg_path}")

    cfg = OmegaConf.load(str(cfg_path))

    # The upstream configs use ./checkpoints paths. Redirect them to the
    # persistent/runtime checkpoint directory without changing the upstream tree.
    for section in ("models", "animal_models"):
        if section not in cfg:
            continue
        for name in cfg[section]:
            model_path = cfg[section][name].get("model_path")
            if isinstance(model_path, str):
                cfg[section][name].model_path = model_path.replace(
                    "./checkpoints", str(CHECKPOINT_DIR)
                )
            elif model_path is not None:
                cfg[section][name].model_path = [
                    p.replace("./checkpoints", str(CHECKPOINT_DIR)) for p in model_path
                ]

    if "infer_params" in cfg:
        cfg.infer_params.flag_pasteback = True

    # Human portrait mode is the requested production path.
    return FasterLivePortraitPipeline(cfg=cfg, is_animal=False)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global PIPELINE, STARTUP_ERROR
    _prepare_checkpoints_dir()
    try:
        log.info("Loading FasterLivePortrait pipeline from %s", CONFIG_PATH)
        PIPELINE = await asyncio.to_thread(_load_pipeline)
        log.info("FasterLivePortrait pipeline ready")
    except Exception as exc:
        STARTUP_ERROR = f"{type(exc).__name__}: {exc}"
        log.exception("Pipeline initialization failed")
        # Keep HTTP/WebSocket server alive so health/diagnostic endpoints work.
    yield
    PIPELINE = None
    SESSIONS.clear()


app = FastAPI(
    title="FasterLivePortrait Live API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=ALLOWED_ORIGINS != ["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {
        "status": "ok" if PIPELINE is not None else "degraded",
        "pipeline_ready": PIPELINE is not None,
        "cuda_available": bool(PIPELINE and str(PIPELINE.device) == "cuda"),
        "target_fps": TARGET_FPS,
        "startup_error": STARTUP_ERROR,
    }


@app.get("/")
async def dashboard():
    return FileResponse("index.html", media_type="text/html")


@app.post("/api/source")
async def upload_source(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(415, "Upload a portrait image (JPEG/PNG/WebP).")

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

        if PIPELINE is None:
            raise HTTPException(503, "FasterLivePortrait pipeline is not ready.")

        # prepare_source is stateful, therefore serialize access.
        async with PIPELINE_LOCK:
            ok = await asyncio.to_thread(PIPELINE.prepare_source, str(source_path), realtime=True)

        if not ok or not PIPELINE.src_imgs or not PIPELINE.src_infos:
            raise HTTPException(422, "No usable face was detected in the source portrait.")

        SESSIONS[session_id] = {
            "dir": str(session_dir),
            "source": str(source_path),
            "created": time.time(),
        }
        return {"session_id": session_id, "status": "ready"}
    except HTTPException:
        shutil.rmtree(session_dir, ignore_errors=True)
        raise
    except Exception as exc:
        shutil.rmtree(session_dir, ignore_errors=True)
        log.exception("Source preparation failed")
        raise HTTPException(500, f"Source preparation failed: {exc}") from exc


def _decode_jpeg(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Invalid JPEG frame")
    return frame


def _encode_jpeg(frame_rgb: np.ndarray) -> bytes:
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(
        ".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 82]
    )
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return encoded.tobytes()


@app.websocket("/api/stream-live")
async def stream_live(ws: WebSocket):
    await ws.accept()
    session_id = ws.query_params.get("session_id")
    session = SESSIONS.get(session_id or "")
    if not session:
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

                # FasterLivePortraitPipeline.run is stateful (motion smoothing,
                # reference rotation and frame state), so one process must
                # serialize inference. This avoids corrupting its internal state.
                async with PIPELINE_LOCK:
                    _, output_crop, output_full, _ = await asyncio.to_thread(
                        PIPELINE.run,
                        frame,
                        PIPELINE.src_imgs[0],
                        PIPELINE.src_infos[0],
                        first_frame=first_frame,
                        realtime=True,
                    )

                first_frame = False
                if output_full is None:
                    await ws.send_json({"type": "status", "message": "No face detected"})
                    continue

                jpeg = await asyncio.to_thread(_encode_jpeg, output_full.cpu().numpy().astype(np.uint8))
                await ws.send_bytes(jpeg)
                frames += 1

                elapsed = max(time.perf_counter() - started, 1e-6)
                if frames % 30 == 0:
                    await ws.send_json(
                        {"type": "stats", "fps": round(frames / elapsed, 2), "frames": frames}
                    )
            except Exception as exc:
                log.exception("Frame processing error")
                await ws.send_json({"type": "error", "message": str(exc)})
    except WebSocketDisconnect:
        pass
    finally:
        log.info("WebSocket closed for session %s", session_id)


@app.delete("/api/source/{session_id}")
async def delete_source(session_id: str):
    session = SESSIONS.pop(session_id, None)
    if session:
        shutil.rmtree(session["dir"], ignore_errors=True)
    return JSONResponse({"status": "deleted"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host=HOST, port=PORT, workers=1)

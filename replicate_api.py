from __future__ import annotations

import hmac
import os
import re
from typing import Any

import httpx
from fastapi import APIRouter, File, Header, HTTPException, UploadFile

router = APIRouter(prefix="/api/replicate", tags=["Replicate FasterLivePortrait"])

REPLICATE_API = "https://api.replicate.com/v1"
MODEL_VERSION = "87c30186211a69bb4636d86823fec8e261f77e82999c868efad71fe9b4464142"
MAX_UPLOAD_BYTES = int(os.getenv("REPLICATE_MAX_UPLOAD_MB", "20")) * 1024 * 1024


def _credentials(access_key: str | None) -> str:
    expected = (os.getenv("KEMZY_REPLICATE_ACCESS_KEY") or "").strip()
    token = (os.getenv("REPLICATE_API_TOKEN") or "").strip()
    if not expected or not token:
        raise HTTPException(status_code=503, detail="Replicate integration is not configured")
    if not access_key or not hmac.compare_digest(access_key, expected):
        raise HTTPException(status_code=401, detail="Unauthorized")
    return token


async def _read_upload(file: UploadFile, kind: str) -> bytes:
    content_type = (file.content_type or "").lower()
    if kind == "image" and not content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="source_image must be an image")
    if kind == "video" and not content_type.startswith("video/"):
        raise HTTPException(status_code=415, detail="driving_video must be a video")
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if not data:
        raise HTTPException(status_code=400, detail=f"{kind} upload is empty")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"Each upload must be {MAX_UPLOAD_BYTES // (1024 * 1024)} MB or smaller")
    return data


async def _upload_file(client: httpx.AsyncClient, token: str, file: UploadFile, data: bytes, default_name: str) -> str:
    response = await client.post(
        f"{REPLICATE_API}/files",
        headers={"Authorization": f"Bearer {token}"},
        files={"content": (file.filename or default_name, data, file.content_type or "application/octet-stream")},
    )
    if response.is_error:
        raise HTTPException(status_code=502, detail=f"Replicate file upload failed (HTTP {response.status_code})")
    payload = response.json()
    url = (payload.get("urls") or {}).get("get") or payload.get("url")
    if not url:
        raise HTTPException(status_code=502, detail="Replicate did not return a file URL")
    return str(url)


@router.post("/animate")
async def animate(
    source_image: UploadFile = File(...),
    driving_video: UploadFile = File(...),
    fps: int = 30,
    driving_multiplier: float = 1.0,
    x_kemzy_access_key: str | None = Header(default=None),
) -> dict[str, Any]:
    token = _credentials(x_kemzy_access_key)
    if not 1 <= fps <= 60:
        raise HTTPException(status_code=422, detail="fps must be between 1 and 60")
    if not 0.1 <= driving_multiplier <= 3:
        raise HTTPException(status_code=422, detail="driving_multiplier must be between 0.1 and 3")

    image_data = await _read_upload(source_image, "image")
    video_data = await _read_upload(driving_video, "video")
    timeout = httpx.Timeout(120.0, connect=20.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        image_url = await _upload_file(client, token, source_image, image_data, "source-image.jpg")
        video_url = await _upload_file(client, token, driving_video, video_data, "driving-video.mp4")
        response = await client.post(
            f"{REPLICATE_API}/predictions",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "Prefer": "wait=1",
            },
            json={
                "version": MODEL_VERSION,
                "input": {
                    "source_image": image_url,
                    "driving_video": video_url,
                    "fps": fps,
                    "driving_multiplier": driving_multiplier,
                },
            },
        )
    if response.is_error:
        raise HTTPException(status_code=502, detail=f"Replicate prediction request failed (HTTP {response.status_code})")
    result = response.json()
    return {
        "prediction_id": result.get("id"),
        "status": result.get("status", "starting"),
        "output": result.get("output"),
        "error": result.get("error"),
        "urls": result.get("urls", {}),
    }


@router.get("/predictions/{prediction_id}")
async def prediction_status(
    prediction_id: str,
    x_kemzy_access_key: str | None = Header(default=None),
) -> dict[str, Any]:
    token = _credentials(x_kemzy_access_key)
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", prediction_id):
        raise HTTPException(status_code=400, detail="Invalid prediction ID")
    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
        response = await client.get(
            f"{REPLICATE_API}/predictions/{prediction_id}",
            headers={"Authorization": f"Bearer {token}"},
        )
    if response.is_error:
        raise HTTPException(status_code=502, detail=f"Replicate prediction lookup failed (HTTP {response.status_code})")
    result = response.json()
    return {
        "prediction_id": result.get("id"),
        "status": result.get("status"),
        "output": result.get("output"),
        "error": result.get("error"),
        "urls": result.get("urls", {}),
        "metrics": result.get("metrics", {}),
    }

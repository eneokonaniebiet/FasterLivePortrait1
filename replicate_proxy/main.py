from __future__ import annotations

import hmac
import os
import re
from typing import Any

import httpx
from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

REPLICATE_API = "https://api.replicate.com/v1"
MODEL_VERSION = "87c30186211a69bb4636d86823fec8e261f77e82999c868efad71fe9b4464142"
MAX_UPLOAD_BYTES = int(os.getenv("REPLICATE_MAX_UPLOAD_MB", "20")) * 1024 * 1024

app = FastAPI(title="FasterLivePortrait Replicate Gateway", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[x.strip() for x in os.getenv("CORS_ORIGINS", "*").split(",") if x.strip()],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-Kemzy-Access-Key"],
)


def credentials(access_key: str | None) -> str:
    expected = (os.getenv("KEMZY_REPLICATE_ACCESS_KEY") or "").strip()
    token = (os.getenv("REPLICATE_API_TOKEN") or "").strip()
    if not expected or not token:
        raise HTTPException(status_code=503, detail="Replicate integration is not configured")
    if not access_key or not hmac.compare_digest(access_key, expected):
        raise HTTPException(status_code=401, detail="Unauthorized")
    return token


async def read_upload(file: UploadFile, kind: str) -> bytes:
    content_type = (file.content_type or "").lower()
    if kind == "image" and not content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="source_image must be an image")
    if kind == "video" and not content_type.startswith("video/"):
        raise HTTPException(status_code=415, detail="driving_video must be a video")
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if not data:
        raise HTTPException(status_code=400, detail=f"{kind} upload is empty")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"Each file must be {MAX_UPLOAD_BYTES // (1024 * 1024)} MB or smaller")
    return data


async def upload_to_replicate(client: httpx.AsyncClient, token: str, file: UploadFile, data: bytes, fallback: str) -> str:
    response = await client.post(
        f"{REPLICATE_API}/files",
        headers={"Authorization": f"Bearer {token}"},
        files={"content": (file.filename or fallback, data, file.content_type or "application/octet-stream")},
    )
    if response.is_error:
        raise HTTPException(status_code=502, detail=f"Replicate file upload failed (HTTP {response.status_code})")
    payload = response.json()
    url = (payload.get("urls") or {}).get("get") or payload.get("url")
    if not url:
        raise HTTPException(status_code=502, detail="Replicate did not return a file URL")
    return str(url)



@app.get("/", response_class=HTMLResponse)
async def home() -> str:
    return """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>FasterLivePortrait • Replicate test</title>
<style>
body{margin:0;background:#10131b;color:#f4f6fb;font:16px system-ui,sans-serif;padding:22px}
main{max-width:640px;margin:auto}h1{font-size:1.7rem}p{color:#b8c0d4;line-height:1.5}
form{background:#1b2130;border:1px solid #343d52;border-radius:16px;padding:18px}
label{display:block;margin:15px 0 7px;font-weight:650}input,button{width:100%;box-sizing:border-box;padding:12px;border-radius:10px;border:1px solid #48536b;background:#111722;color:#fff}
button{margin-top:18px;background:#72f0bd;color:#07130e;font-weight:800;border:0}
#result{white-space:pre-wrap;overflow-wrap:anywhere;background:#0a0d13;border-radius:12px;padding:14px;margin-top:18px}
small{color:#b8c0d4}a{color:#72f0bd}
</style></head><body><main>
<h1>FasterLivePortrait · Replicate test</h1>
<p>This page sends a portrait image and a driving video to your configured Replicate model. It is an animation job, not a continuous live-camera stream.</p>
<form id="job">
<label for="key">Kémzy access key</label><input id="key" type="password" autocomplete="off" required>
<small>Use the value you saved as KEMZY_REPLICATE_ACCESS_KEY in Render. Never enter your Replicate API token here.</small>
<label for="image">Portrait image</label><input id="image" type="file" accept="image/*" required>
<label for="video">Driving video</label><input id="video" type="file" accept="video/*" required>
<label for="fps">Output FPS (1–60)</label><input id="fps" type="number" min="1" max="60" value="30" required>
<label for="multiplier">Driving multiplier (0.1–3)</label><input id="multiplier" type="number" min="0.1" max="3" step="0.1" value="1" required>
<button id="submit" type="submit">Generate animation</button>
</form><div id="result" aria-live="polite">Ready. Your request may use your Replicate credit.</div>
<script>
const form=document.getElementById('job'),result=document.getElementById('result'),button=document.getElementById('submit');
form.addEventListener('submit',async e=>{
e.preventDefault();button.disabled=true;result.textContent='Uploading files and submitting prediction…';
try{
const data=new FormData();data.append('source_image',document.getElementById('image').files[0]);data.append('driving_video',document.getElementById('video').files[0]);data.append('fps',document.getElementById('fps').value);data.append('driving_multiplier',document.getElementById('multiplier').value);
const key=document.getElementById('key').value;
let r=await fetch('/api/replicate/animate',{method:'POST',headers:{'X-Kemzy-Access-Key':key},body:data});let j=await r.json();
if(!r.ok)throw new Error(j.detail||('Request failed: '+r.status));
result.textContent=JSON.stringify(j,null,2);
if(j.prediction_id && !['succeeded','failed','canceled'].includes(j.status)){
for(let i=0;i<20;i++){await new Promise(resolve=>setTimeout(resolve,4000));r=await fetch('/api/replicate/predictions/'+encodeURIComponent(j.prediction_id),{headers:{'X-Kemzy-Access-Key':key}});j=await r.json();if(!r.ok)throw new Error(j.detail||('Status check failed: '+r.status));result.textContent=JSON.stringify(j,null,2);if(['succeeded','failed','canceled'].includes(j.status))break;}
}
}catch(err){result.textContent='Error: '+err.message;}finally{button.disabled=false;}
});
</script></main></body></html>"""


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"status": "ok", "service": "fasterliveportrait-replicate-api"}


@app.post("/api/replicate/animate")
async def animate(
    source_image: UploadFile = File(...),
    driving_video: UploadFile = File(...),
    fps: int = 30,
    driving_multiplier: float = 1.0,
    x_kemzy_access_key: str | None = Header(default=None),
) -> dict[str, Any]:
    token = credentials(x_kemzy_access_key)
    if not 1 <= fps <= 60:
        raise HTTPException(status_code=422, detail="fps must be between 1 and 60")
    if not 0.1 <= driving_multiplier <= 3:
        raise HTTPException(status_code=422, detail="driving_multiplier must be between 0.1 and 3")
    image_data = await read_upload(source_image, "image")
    video_data = await read_upload(driving_video, "video")
    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=20.0)) as client:
        image_url = await upload_to_replicate(client, token, source_image, image_data, "source-image.jpg")
        video_url = await upload_to_replicate(client, token, driving_video, video_data, "driving-video.mp4")
        response = await client.post(
            f"{REPLICATE_API}/predictions",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json", "Prefer": "wait=1"},
            json={"version": MODEL_VERSION, "input": {
                "source_image": image_url, "driving_video": video_url,
                "fps": fps, "driving_multiplier": driving_multiplier,
            }},
        )
    if response.is_error:
        raise HTTPException(status_code=502, detail=f"Replicate prediction request failed (HTTP {response.status_code})")
    result = response.json()
    return {
        "prediction_id": result.get("id"), "status": result.get("status", "starting"),
        "output": result.get("output"), "error": result.get("error"), "urls": result.get("urls", {}),
    }


@app.get("/api/replicate/predictions/{prediction_id}")
async def prediction_status(
    prediction_id: str,
    x_kemzy_access_key: str | None = Header(default=None),
) -> dict[str, Any]:
    token = credentials(x_kemzy_access_key)
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
        "prediction_id": result.get("id"), "status": result.get("status"),
        "output": result.get("output"), "error": result.get("error"),
        "urls": result.get("urls", {}), "metrics": result.get("metrics", {}),
    }

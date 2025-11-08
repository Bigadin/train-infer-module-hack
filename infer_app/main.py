import os
import time
import threading
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException, APIRouter
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.staticfiles import StaticFiles

from .manager import InferenceManager


DATA_DIR = os.path.abspath(os.getenv("INFER_DATA_DIR", "./data_infer"))
os.makedirs(DATA_DIR, exist_ok=True)

app = FastAPI(title="Inference API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

mgr = InferenceManager(base_dir=DATA_DIR)

# Provide an "/api"-prefixed router so calls to :8001/api/* also work
api = APIRouter()


@app.get("/", response_class=HTMLResponse)
def index():
    html_path = Path(__file__).parent / "static" / "index.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Inference API</h1>")


@app.post("/infer")
@api.post("/infer")
async def infer(
    background_tasks: BackgroundTasks,
    model_type: str = Form(..., description="yolo|detr"),
    mode: str = Form(..., description="image|video-render|video-realtime"),
    threshold: float = Form(0.5),
    fps: int = Form(1, description="Frame stride for realtime video; 10 means 1 of every 10 frames"),
    weights: UploadFile = File(...),
    source: UploadFile = File(...),
):
    model_type = model_type.lower().strip()
    if model_type not in {"yolo", "detr"}:
        raise HTTPException(400, "model_type must be yolo or detr")
    mode = mode.lower().strip()
    if mode not in {"image", "video-render", "video-realtime"}:
        raise HTTPException(400, "mode must be image|video-render|video-realtime")
    if fps < 1:
        fps = 1

    job_id = mgr.create_job(model_type=model_type, mode=mode, threshold=threshold, fps=fps)

    # Save uploads
    weights_path = mgr.get_job_dir(job_id) / f"weights_{weights.filename}"
    with open(weights_path, "wb") as f:
        f.write(await weights.read())
    source_path = mgr.get_job_dir(job_id) / f"src_{source.filename}"
    with open(source_path, "wb") as f:
        f.write(await source.read())

    background_tasks.add_task(mgr.run_job, job_id, str(weights_path), str(source_path))
    return {"job_id": job_id}


@app.get("/status/{job_id}")
@api.get("/status/{job_id}")
def status(job_id: str):
    job = mgr.get(job_id)
    if not job:
        raise HTTPException(404, "Unknown job_id")
    return mgr.public_status(job_id)


@app.get("/logs/{job_id}", response_class=PlainTextResponse)
@api.get("/logs/{job_id}", response_class=PlainTextResponse)
def logs(job_id: str, tail: int = 2000):
    path = mgr.get_job_log(job_id)
    if not path.exists():
        raise HTTPException(404, "No logs")
    data = path.read_text(encoding="utf-8", errors="ignore")
    if tail > 0 and len(data) > tail:
        return data[-tail:]
    return data


@app.get("/result/{job_id}")
@api.get("/result/{job_id}")
def result(job_id: str):
    info = mgr.get(job_id)
    if not info:
        raise HTTPException(404, "Unknown job_id")
    out = info.get("result")
    if not out or not Path(out).exists():
        raise HTTPException(404, "No result yet")
    fn = Path(out).name
    media = "image/jpeg" if fn.lower().endswith((".jpg", ".jpeg", ".png")) else "video/mp4"
    return FileResponse(out, filename=fn, media_type=media)


@app.get("/rt/{job_id}/frame.jpg")
@api.get("/rt/{job_id}/frame.jpg")
def rt_latest_frame(job_id: str):
    p = mgr.latest_realtime_frame(job_id)
    if not p:
        raise HTTPException(404, "No frame")
    return FileResponse(str(p), media_type="image/jpeg")


@app.get("/rt/{job_id}/frames")
@api.get("/rt/{job_id}/frames")
def rt_list(job_id: str, limit: int = 50):
    files = mgr.realtime_frames(job_id, limit=limit)
    return {"frames": [f"/rt/{job_id}/frame.jpg?t={int(Path(f).stat().st_mtime)}" for f in files]}


# Serve static
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Mount the API router under /api as well
app.include_router(api, prefix="/api")

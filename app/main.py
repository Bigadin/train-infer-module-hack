import io
import os
import shutil
import threading
import time
import uuid
from typing import Optional, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.staticfiles import StaticFiles

from .training.manager import JobManager
from .schemas import TrainRequest


DATA_DIR = os.path.abspath(os.getenv("DATA_DIR", "./data"))
os.makedirs(DATA_DIR, exist_ok=True)

app = FastAPI(title="Simple Training API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

job_manager = JobManager(base_dir=DATA_DIR)


@app.get("/", response_class=HTMLResponse)
def index():
    # Minimal UI for manual testing
    html = (
        (Path(__file__).parent / "static" / "index.html").read_text(encoding="utf-8")
        if (Path(__file__).parent / "static" / "index.html").exists()
        else "<h1>Training API</h1><p>Run container with static files.</p>"
    )
    return HTMLResponse(content=html)


@app.post("/train")
async def train(
    background_tasks: BackgroundTasks,
    model_type: str = Form(..., description="'yolo' or 'detr'"),
    dataset_url: Optional[str] = Form(None, description="Optional HTTP URL to a zip of the dataset in the expected format"),
    epochs: int = Form(20),
    batch_size: int = Form(4),
    lr: float = Form(1e-4),
    grad_accum_steps: int = Form(1),
    img_size: int = Form(640),
    dataset: Optional[UploadFile] = File(None),
):
    """
    Start a training job. One of dataset_url OR dataset file can be provided.

    - model_type: 'yolo' uses YOLO small; 'detr' uses RFDETR Base.
    - epochs, batch_size, lr, grad_accum_steps, img_size are basic params.

    Dataset format must match the trainer expectations:
      - YOLO: a zip of a YOLO-format dataset containing a data.yaml
      - DETR: a zip of a COCO-format dataset (images + annotations json)
    """
    model_type = model_type.lower().strip()
    if model_type not in {"yolo", "detr"}:
        raise HTTPException(status_code=400, detail="model_type must be 'yolo' or 'detr'")
    print(f"[API] Received training request: model_type={model_type}, dataset_url={dataset_url}, dataset_file={'yes' if dataset else 'no'}, epochs={epochs}, batch_size={batch_size}, lr={lr}, grad_accum_steps={grad_accum_steps}, img_size={img_size}")
    if not dataset_url and not dataset:
        raise HTTPException(status_code=400, detail="Provide dataset_url or upload a dataset file")

    job_id = job_manager.create_job(
        model_type=model_type,
        params={
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "grad_accum_steps": grad_accum_steps,
            "img_size": img_size,
        },
    )

    # Save uploaded dataset (if any)
    if dataset is not None:
        target_path = job_manager.get_job_dataset_zip_path(job_id)
        with open(target_path, "wb") as f:
            f.write(await dataset.read())
        dataset_url_local = None
    else:
        dataset_url_local = dataset_url
    
    print(f"[API] Starting job {job_id} in background...")

    # Kick background training
    background_tasks.add_task(job_manager.run_job, job_id, dataset_url_local)
    return {"job_id": job_id}


@app.get("/status/{job_id}")
def status(job_id: str):
    job = job_manager.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Unknown job_id")
    return job_manager.public_status(job_id)


@app.get("/logs/{job_id}", response_class=PlainTextResponse)
def logs(job_id: str, tail: int = 2000):
    path = job_manager.get_job_log_path(job_id)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="No logs for job")
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            data = f.read()
        if tail > 0 and len(data) > tail:
            return data[-tail:]
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/download/{job_id}")
def download(job_id: str):
    info = job_manager.get(job_id)
    if not info:
        raise HTTPException(status_code=404, detail="Unknown job_id")
    weights = info.get("weights_path")
    if not weights or not os.path.exists(weights):
        raise HTTPException(status_code=404, detail="Weights not available yet")
    filename = os.path.basename(weights)
    return FileResponse(weights, filename=filename, media_type="application/octet-stream")


def _job_paths(job_id: str):
    from pathlib import Path
    job_dir = Path(job_manager.get_job_dir(job_id))
    outputs = Path(job_manager.get_job_output_dir(job_id))
    return job_dir, outputs


def _find_yolo_run_dir(outputs_dir: "Path") -> Optional["Path"]:
    # Look for runs/*/weights
    candidates = []
    runs_dir = outputs_dir / "runs"
    if runs_dir.exists():
        for p in runs_dir.rglob("weights"):
            # parent of weights is the run directory (like train)
            candidates.append(p.parent)
    if candidates:
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return candidates[0]
    return None


@app.get("/artifacts/{job_id}")
def artifacts(job_id: str):
    info = job_manager.get(job_id)
    if not info:
        raise HTTPException(status_code=404, detail="Unknown job_id")
    job_dir, outputs = _job_paths(job_id)
    run_dir = _find_yolo_run_dir(outputs) or outputs
    images = []
    csvs = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        for p in run_dir.rglob(ext):
            rel = p.relative_to(job_dir)
            images.append(str(rel).replace("\\", "/"))
    for p in run_dir.rglob("*.csv"):
        rel = p.relative_to(job_dir)
        csvs.append(str(rel).replace("\\", "/"))
    # Build URLs
    imgs = [f"/artifact/{job_id}?file={f}" for f in images]
    csv_urls = [f"/artifact/{job_id}?file={f}" for f in csvs]
    return {"images": imgs, "csvs": csv_urls}


@app.get("/artifact/{job_id}")
def artifact(job_id: str, file: str):
    # Serve a specific artifact file under the job directory
    job = job_manager.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Unknown job_id")
    base = Path(job_manager.get_job_dir(job_id)).resolve()
    target = (base / file).resolve()
    if not str(target).startswith(str(base)):
        raise HTTPException(status_code=400, detail="Invalid path")
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(str(target))


@app.get("/metrics/{job_id}")
def metrics(job_id: str):
    # For YOLO, parse results.csv (last row)
    from pathlib import Path
    import csv

    info = job_manager.get(job_id)
    if not info:
        raise HTTPException(status_code=404, detail="Unknown job_id")
    job_dir, outputs = _job_paths(job_id)
    run_dir = _find_yolo_run_dir(outputs)
    if not run_dir:
        return {"metrics": {}, "found": False}
    results_csv = run_dir / "results.csv"
    if not results_csv.exists():
        return {"metrics": {}, "found": False}
    rows = []
    with open(results_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    if not rows:
        return {"metrics": {}, "found": False}
    last = rows[-1]
    # Return a subset of commonly used metrics if present
    keys = [
        "metrics/precision(B)",
        "metrics/recall(B)",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
        "train/box_loss",
        "train/cls_loss",
        "train/dfl_loss",
        "val/box_loss",
        "val/cls_loss",
        "val/dfl_loss",
        "epoch",
    ]
    filtered = {k: last.get(k) for k in keys if k in last}
    return {"metrics": filtered, "found": True}


# Late import to avoid circular import at module init
from pathlib import Path  # noqa: E402

# Serve static if present
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

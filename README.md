Simple Training API (YOLO / DETR)

Overview
- FastAPI service to start training jobs for either YOLO (small) or DETR (RFDETR Base) via HTTP.
- Upload a dataset zip or provide a dataset URL.
- Poll job status/logs and download the trained weights at completion.
- Minimal HTML UI available at `/` for manual use.

Endpoints
- POST `/train` (multipart/form-data)
  - model_type: `yolo` or `detr`
  - dataset: file (zip) [optional]
  - dataset_url: string (zip URL) [optional]
  - epochs: int (default 20)
  - batch_size: int (default 4)
  - lr: float (default 1e-4 for DETR, 1e-3 for YOLO)
  - grad_accum_steps: int (DETR only, default 1)
  - img_size: int (YOLO only, default 640)
  Returns: `{ job_id }`

- GET `/status/{job_id}`: job status JSON
- GET `/logs/{job_id}`: plain text logs (tail)
- GET `/download/{job_id}`: download weights (when ready)

Dataset Format
- YOLO: Provide a YOLO-format dataset zip containing a `data.yaml` that defines `train`, `val`, and `names`.
- DETR: Provide a COCO-format dataset zip (images + COCO annotations json). No conversion is performed by the API.

Models
- YOLO: Uses Ultralytics `yolov8s` (small).
- DETR: Uses `RFDETRBase` directly (import: `from rfdetr import RFDETRBase`). Ensure `rfdetr` is installable in the container or vendored.

Run with Docker (GPU)
1) Build and run:
   - `docker compose up --build`
2) Open: `http://localhost:8000/`

Notes
- GPU: The compose file requests NVIDIA GPU (`gpus: all`) and the Dockerfile is CUDA runtime based. Install the NVIDIA Container Toolkit on the host.
- Torch/Torchvision: As requested, Dockerfile installs CUDA wheels explicitly:
  `pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128`
- Output/weights and datasets persist in `./data`.

Local Dev (without Docker)
- `pip install -r requirements.txt`
- Also install torch/torchvision with the CUDA wheels appropriate for your system (or CPU wheels if testing):
  `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128`
- `uvicorn app.main:app --reload`

Limitations (by design for simplicity)
- Progress is coarse (Created/Running/Completed). Logs are available for more detail.
- No dataset conversion between COCO and YOLO formats.
- Error handling is minimal; intended for non-production usage.


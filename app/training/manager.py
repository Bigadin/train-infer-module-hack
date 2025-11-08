import contextlib
import io
import json
import os
import sys
import threading
import time
import traceback
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Any

import requests
import shutil

from .yolo_trainer import train_yolo
from .detr_trainer import train_detr


@contextlib.contextmanager
def redirect_output(to_path: Path):
    to_path.parent.mkdir(parents=True, exist_ok=True)
    with open(to_path, "a", buffering=1) as f:
        class Stream:
            def write(self, s):
                try:
                    f.write(s)
                    f.flush()
                except Exception:
                    pass

            def flush(self):
                try:
                    f.flush()
                except Exception:
                    pass

        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = Stream(), Stream()
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_out, old_err


def _now_ts() -> float:
    return time.time()


class JobManager:
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir).absolute()
        self.jobs: Dict[str, Dict[str, Any]] = {}
        (self.base_dir / "jobs").mkdir(parents=True, exist_ok=True)

    # Paths helpers
    def get_job_dir(self, job_id: str) -> Path:
        return self.base_dir / "jobs" / job_id

    def get_job_dataset_zip_path(self, job_id: str) -> str:
        return str(self.get_job_dir(job_id) / "dataset.zip")

    def get_job_dataset_dir(self, job_id: str) -> Path:
        return self.get_job_dir(job_id) / "dataset"

    def get_job_log_path(self, job_id: str) -> str:
        return str(self.get_job_dir(job_id) / "train.log")

    def get_job_output_dir(self, job_id: str) -> Path:
        return self.get_job_dir(job_id) / "outputs"

    def create_job(self, model_type: str, params: Dict[str, Any]) -> str:
        job_id = time.strftime("%Y%m%d-%H%M%S") + f"-{os.urandom(3).hex()}"
        job_dir = self.get_job_dir(job_id)
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "outputs").mkdir(exist_ok=True)
        (job_dir / "dataset").mkdir(exist_ok=True)
        self.jobs[job_id] = {
            "job_id": job_id,
            "model_type": model_type,
            "params": params,
            "status": "created",
            "progress": 0,
            "message": "",
            "weights_path": None,
            "created_at": _now_ts(),
            "updated_at": _now_ts(),
        }
        return job_id

    def set_status(self, job_id: str, status: str, message: str = "", progress: Optional[int] = None):
        job = self.jobs.get(job_id)
        if not job:
            return
        job["status"] = status
        if message:
            job["message"] = message
        if progress is not None:
            job["progress"] = int(max(0, min(100, progress)))
        job["updated_at"] = _now_ts()

    def set_weights(self, job_id: str, path: Optional[str]):
        job = self.jobs.get(job_id)
        if not job:
            return
        job["weights_path"] = path
        job["updated_at"] = _now_ts()

    def get(self, job_id: str) -> Optional[Dict[str, Any]]:
        return self.jobs.get(job_id)

    def public_status(self, job_id: str) -> Dict[str, Any]:
        job = self.jobs.get(job_id)
        if not job:
            raise KeyError(job_id)
        return {
            "job_id": job["job_id"],
            "status": job["status"],
            "progress": job.get("progress", 0),
            "message": job.get("message", ""),
            "weights_ready": bool(job.get("weights_path")),
            "created_at": job.get("created_at"),
            "updated_at": job.get("updated_at"),
        }

    def _download_dataset(self, url: str, to_zip_path: Path):
        self.set_status(to_zip_path.parent.name, "downloading", f"Downloading dataset from {url}")
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(to_zip_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        self.set_status(to_zip_path.parent.name, "downloading", "Download complete", progress=20)

    def _extract_zip(self, zip_path: Path, dest_dir: Path):
        self.set_status(zip_path.parent.name, "preparing", "Extracting dataset", progress=25)
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(dest_dir)
        self.set_status(zip_path.parent.name, "preparing", "Dataset ready", progress=30)

    def run_job(self, job_id: str, dataset_url: Optional[str]):
        job = self.jobs.get(job_id)
        if not job:
            return
        job_dir = self.get_job_dir(job_id)
        log_path = Path(self.get_job_log_path(job_id))
        ds_zip = Path(self.get_job_dataset_zip_path(job_id))
        ds_dir = self.get_job_dataset_dir(job_id)
        out_dir = self.get_job_output_dir(job_id)

        with redirect_output(log_path):
            try:
                print(f"[JOB {job_id}] Starting training for model_type={job['model_type']}")
                # Download if URL provided
                if dataset_url:
                    self._download_dataset(dataset_url, ds_zip)
                # If we have a zip, extract it
                if ds_zip.exists():
                    self._extract_zip(ds_zip, ds_dir)

                # Update status and run trainer
                self.set_status(job_id, "running", "Training in progress", progress=50)

                # Change CWD so trainers write under job folder
                old_cwd = os.getcwd()
                os.chdir(job_dir)
                try:
                    if job["model_type"] == "yolo":
                        weights = train_yolo(
                            dataset_dir=str(ds_dir),
                            output_dir=str(out_dir),
                            epochs=int(job["params"].get("epochs", 20)),
                            batch_size=int(job["params"].get("batch_size", 4)),
                            lr=float(job["params"].get("lr", 1e-3)),
                            img_size=int(job["params"].get("img_size", 640)),
                            device=os.getenv("YOLO_DEVICE", "0"),
                            workers=int(os.getenv("YOLO_WORKERS", "2")),
                        )
                    else:
                        weights = train_detr(
                            dataset_dir=str(ds_dir),
                            output_dir=str(out_dir),
                            epochs=int(job["params"].get("epochs", 20)),
                            batch_size=int(job["params"].get("batch_size", 4)),
                            grad_accum_steps=int(job["params"].get("grad_accum_steps", 1)),
                            lr=float(job["params"].get("lr", 1e-4)),
                            device=os.getenv("DETR_DEVICE", "cuda" if _torch_cuda_available() else "cpu"),
                        )
                finally:
                    os.chdir(old_cwd)

                self.set_weights(job_id, weights)
                self.set_status(job_id, "completed", "Training complete", progress=100)
                print(f"[JOB {job_id}] Completed. Weights: {weights}")
                # Enforce retention after completion
                self._enforce_retention(max_keep=6)
            except Exception as e:
                print(f"[JOB {job_id}] ERROR: {e}")
                traceback.print_exc()
                self.set_status(job_id, "failed", f"Error: {e}")
                # Also enforce retention, in case many failed jobs accumulate
                self._enforce_retention(max_keep=6)

    # Retention: keep only the most recent N job folders, never deleting active ones
    def _enforce_retention(self, max_keep: int = 6):
        try:
            jobs_root = self.base_dir / "jobs"
            if not jobs_root.exists():
                return
            entries = [p for p in jobs_root.iterdir() if p.is_dir()]
            if len(entries) <= max_keep:
                return

            # Determine active jobs to never delete
            active_ids = set(
                jid for jid, j in self.jobs.items() if j.get("status") not in {"completed", "failed"}
            )

            # Sort by mtime (newest first)
            entries.sort(key=lambda p: p.stat().st_mtime, reverse=True)

            # Build keep set: start with all active, then newest completed until limit
            keep = set(active_ids)
            for p in entries:
                jid = p.name
                if jid in keep:
                    continue
                if len(keep) >= max_keep:
                    break
                keep.add(jid)

            # Delete the rest
            for p in entries:
                jid = p.name
                if jid in keep:
                    continue
                try:
                    shutil.rmtree(p)
                    print(f"[RETENTION] Removed old job folder: {p}")
                except Exception as e:
                    print(f"[RETENTION] Failed to remove {p}: {e}")
        except Exception as e:
            print(f"[RETENTION] Error enforcing retention: {e}")


def _torch_cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False

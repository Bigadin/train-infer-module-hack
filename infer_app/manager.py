import os
import cv2
import time
import json
import shutil
import zipfile
import traceback
from pathlib import Path
from typing import Dict, Optional, List

from contextlib import contextmanager
import sys

from .workers import run_image_yolo, run_image_detr, run_video_render_yolo, run_video_render_detr, run_video_realtime_yolo, run_video_realtime_detr


@contextmanager
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


def _ts() -> float:
    return time.time()


class InferenceManager:
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir).absolute()
        (self.base_dir / "jobs").mkdir(parents=True, exist_ok=True)
        self.jobs: Dict[str, Dict] = {}

    def create_job(self, model_type: str, mode: str, threshold: float, fps: int) -> str:
        job_id = time.strftime("%Y%m%d-%H%M%S") + f"-{os.urandom(3).hex()}"
        jd = self.get_job_dir(job_id)
        (jd / "rt").mkdir(parents=True, exist_ok=True)
        self.jobs[job_id] = {
            "job_id": job_id,
            "model_type": model_type,
            "mode": mode,
            "threshold": threshold,
            "fps": fps,
            "status": "created",
            "message": "",
            "result": None,
            "created_at": _ts(),
            "updated_at": _ts(),
        }
        return job_id

    def get_job_dir(self, job_id: str) -> Path:
        return self.base_dir / "jobs" / job_id

    def get_job_log(self, job_id: str) -> Path:
        return self.get_job_dir(job_id) / "infer.log"

    def set_status(self, job_id: str, status: str, message: str = ""):
        j = self.jobs.get(job_id)
        if not j:
            return
        j["status"] = status
        if message:
            j["message"] = message
        j["updated_at"] = _ts()

    def set_result(self, job_id: str, path: Optional[str]):
        j = self.jobs.get(job_id)
        if not j:
            return
        j["result"] = path
        j["updated_at"] = _ts()

    def get(self, job_id: str) -> Optional[Dict]:
        return self.jobs.get(job_id)

    def public_status(self, job_id: str) -> Dict:
        j = self.jobs.get(job_id)
        return {
            "job_id": job_id,
            "status": j.get("status"),
            "message": j.get("message", ""),
            "result_ready": bool(j.get("result")),
        }

    def run_job(self, job_id: str, weights_path: str, source_path: str):
        j = self.jobs.get(job_id)
        if not j:
            return
        log = self.get_job_log(job_id)
        with redirect_output(log):
            try:
                print(f"[INFER {job_id}] Starting: {j['model_type']} {j['mode']}")
                self.set_status(job_id, "running", "Loading model...")
                res_path = None
                if j["mode"] == "image":
                    out_image = self.get_job_dir(job_id) / "result.jpg"
                    if j["model_type"] == "yolo":
                        run_image_yolo(weights_path, source_path, str(out_image), threshold=j["threshold"])
                    else:
                        run_image_detr(weights_path, source_path, str(out_image), threshold=j["threshold"])
                    res_path = str(out_image)
                elif j["mode"] == "video-render":
                    out_video = self.get_job_dir(job_id) / "result.mp4"
                    if j["model_type"] == "yolo":
                        run_video_render_yolo(weights_path, source_path, str(out_video), threshold=j["threshold"]) 
                    else:
                        run_video_render_detr(weights_path, source_path, str(out_video), threshold=j["threshold"]) 
                    res_path = str(out_video)
                else:  # realtime
                    rt_dir = self.get_job_dir(job_id) / "rt"
                    if j["model_type"] == "yolo":
                        run_video_realtime_yolo(weights_path, source_path, rt_dir, threshold=j["threshold"], frame_stride=j["fps"], keep_last=100)
                    else:
                        run_video_realtime_detr(weights_path, source_path, rt_dir, threshold=j["threshold"], frame_stride=j["fps"], keep_last=100)
                    res_path = None
                self.set_result(job_id, res_path)
                self.set_status(job_id, "completed", "Done")
                self._enforce_retention(max_keep=6)
            except Exception as e:
                print(f"[INFER {job_id}] ERROR: {e}")
                traceback.print_exc()
                self.set_status(job_id, "failed", str(e))
                self._enforce_retention(max_keep=6)

    # Realtime helpers
    def realtime_frames(self, job_id: str, limit: int = 50) -> List[str]:
        rt = self.get_job_dir(job_id) / "rt"
        if not rt.exists():
            return []
        files = sorted(rt.glob("*.jpg"), key=lambda p: p.stat().st_mtime, reverse=True)
        return [str(p) for p in files[: max(1, limit)]]

    def latest_realtime_frame(self, job_id: str) -> Optional[Path]:
        lst = self.realtime_frames(job_id, limit=1)
        return Path(lst[0]) if lst else None

    # Retention policy: keep only last N jobs
    def _enforce_retention(self, max_keep: int = 6):
        try:
            jobs_root = self.base_dir / "jobs"
            if not jobs_root.exists():
                return
            entries = [p for p in jobs_root.iterdir() if p.is_dir()]
            if len(entries) <= max_keep:
                return
            # Keep newest folders
            entries.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            for p in entries[max_keep:]:
                try:
                    shutil.rmtree(p)
                    print(f"[INFER RETENTION] Removed {p}")
                except Exception as e:
                    print(f"[INFER RETENTION] Failed to remove {p}: {e}")
        except Exception as e:
            print(f"[INFER RETENTION] Error: {e}")


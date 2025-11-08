import os
import cv2
import numpy as np
from pathlib import Path
from typing import Optional

from PIL import Image

import supervision as sv


# ------------- Utilities -------------

def _ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def _annotate_frame(image_bgr: np.ndarray, detections: sv.Detections, names: Optional[dict] = None) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=(w, h))
    thickness = sv.calculate_optimal_line_thickness(resolution_wh=(w, h))
    bbox_annotator = sv.BoxAnnotator(thickness=thickness)
    label_annotator = sv.LabelAnnotator(text_color=sv.Color.BLACK, text_scale=text_scale, text_thickness=thickness)
    labels = []
    for i in range(len(detections)):
        cls_id = int(detections.class_id[i]) if detections.class_id is not None else -1
        conf = float(detections.confidence[i]) if detections.confidence is not None else 0.0
        if names and cls_id in names:
            labels.append(f"{names[cls_id]} {conf:.2f}")
        else:
            labels.append(f"{cls_id} {conf:.2f}")
    annotated = bbox_annotator.annotate(image_bgr.copy(), detections)
    annotated = label_annotator.annotate(annotated, detections, labels)
    return annotated


# ------------- YOLO -------------

def _yolo_load(weights_path: str):
    from ultralytics import YOLO
    return YOLO(weights_path)


def _yolo_detect_to_sv(res) -> sv.Detections:
    return sv.Detections.from_ultralytics(res)


def run_image_yolo(weights_path: str, src_path: str, out_path: str, threshold: float = 0.5):
    model = _yolo_load(weights_path)
    img = cv2.imread(src_path)
    res = model.predict(img, conf=threshold, verbose=False)[0]
    det = _yolo_detect_to_sv(res)
    annotated = _annotate_frame(img, det, names=res.names if hasattr(res, 'names') else None)
    _ensure_parent(Path(out_path))
    cv2.imwrite(out_path, annotated)


def run_video_render_yolo(weights_path: str, src_path: str, out_path: str, threshold: float = 0.5):
    model = _yolo_load(weights_path)
    cap = cv2.VideoCapture(src_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    _ensure_parent(Path(out_path))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        res = model.predict(frame, conf=threshold, verbose=False)[0]
        det = _yolo_detect_to_sv(res)
        annotated = _annotate_frame(frame, det, names=res.names if hasattr(res, 'names') else None)
        writer.write(annotated)
    writer.release()
    cap.release()


def run_video_realtime_yolo(weights_path: str, src_path: str, rt_dir: Path, threshold: float = 0.5, frame_stride: int = 1, keep_last: int = 100):
    model = _yolo_load(weights_path)
    cap = cv2.VideoCapture(src_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video")
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        i += 1
        if frame_stride > 1 and (i % frame_stride) != 0:
            continue
        res = model.predict(frame, conf=threshold, verbose=False)[0]
        det = _yolo_detect_to_sv(res)
        annotated = _annotate_frame(frame, det, names=res.names if hasattr(res, 'names') else None)
        name = f"{int(cv2.getTickCount())}.jpg"
        out = rt_dir / name
        _ensure_parent(out)
        cv2.imwrite(str(out), annotated)
        # retention on frames
        files = sorted(rt_dir.glob("*.jpg"), key=lambda p: p.stat().st_mtime, reverse=True)
        for p in files[keep_last:]:
            try:
                p.unlink()
            except Exception:
                pass
    cap.release()


# ------------- DETR -------------

def _detr_load(weights_path: str):
    from rfdetr import RFDETRBase
    return RFDETRBase(pretrain_weights=weights_path)


def _detr_predict(model, image_bgr: np.ndarray, threshold: float):
    pil_image = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)).convert("RGB")
    detections = model.predict(pil_image, threshold=threshold)
    # Try to build SV detections
    boxes = None
    for attr in ("xyxy", "bbox", "boxes"):
        if hasattr(detections, attr):
            boxes = getattr(detections, attr)
            break
    cls = getattr(detections, "class_id", None)
    conf = getattr(detections, "confidence", None)
    if boxes is None:
        # fallback: return empty
        return sv.Detections.empty()
    arr = np.array(boxes)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 4)
    det = sv.Detections(xyxy=arr)
    if cls is not None:
        det.class_id = np.array(cls, dtype=int)
    if conf is not None:
        det.confidence = np.array(conf, dtype=float)
    return det


def run_image_detr(weights_path: str, src_path: str, out_path: str, threshold: float = 0.5):
    model = _detr_load(weights_path)
    img = cv2.imread(src_path)
    det = _detr_predict(model, img, threshold)
    annotated = _annotate_frame(img, det, names=None)
    _ensure_parent(Path(out_path))
    cv2.imwrite(out_path, annotated)


def run_video_render_detr(weights_path: str, src_path: str, out_path: str, threshold: float = 0.5):
    model = _detr_load(weights_path)
    cap = cv2.VideoCapture(src_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    _ensure_parent(Path(out_path))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        det = _detr_predict(model, frame, threshold)
        annotated = _annotate_frame(frame, det, names=None)
        writer.write(annotated)
    writer.release()
    cap.release()


def run_video_realtime_detr(weights_path: str, src_path: str, rt_dir: Path, threshold: float = 0.5, frame_stride: int = 1, keep_last: int = 100):
    model = _detr_load(weights_path)
    cap = cv2.VideoCapture(src_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video")
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        i += 1
        if frame_stride > 1 and (i % frame_stride) != 0:
            continue
        det = _detr_predict(model, frame, threshold)
        annotated = _annotate_frame(frame, det, names=None)
        name = f"{int(cv2.getTickCount())}.jpg"
        out = rt_dir / name
        _ensure_parent(out)
        cv2.imwrite(str(out), annotated)
        # retention on frames
        files = sorted(rt_dir.glob("*.jpg"), key=lambda p: p.stat().st_mtime, reverse=True)
        for p in files[keep_last:]:
            try:
                p.unlink()
            except Exception:
                pass
    cap.release()


import os
from pathlib import Path
from typing import Optional


def _find_yaml(dataset_dir: Path) -> Optional[str]:
    # Find a data.yaml in dataset root or one level below
    for depth in [dataset_dir, *dataset_dir.glob("*")]:
        cand = list(depth.glob("*.yaml")) + list(depth.glob("*.yml"))
        for c in cand:
            if c.name.lower() in {"data.yaml", "data.yml"}:
                return str(c)
    return None


def _find_weight_prefer_best(search_dir: Path) -> Optional[str]:
    best = list(search_dir.rglob("best.pt"))
    if best:
        best.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return str(best[0])
    last = list(search_dir.rglob("last.pt"))
    if last:
        last.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return str(last[0])
    cand = []
    for ext in ("*.pt", "*.pth"):
        cand.extend(search_dir.rglob(ext))
    if not cand:
        return None
    cand.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(cand[0])


def train_yolo(
    dataset_dir: str,
    output_dir: str,
    epochs: int = 20,
    batch_size: int = 4,
    lr: float = 1e-3,
    img_size: int = 640,
    device: str = "0",
    workers: int = 2,
) -> str:
    """
    Trains YOLO (small) on a YOLO-format dataset with data.yaml.
    Returns the path to the best/last weights found under the job folder.
    """
    from ultralytics import YOLO
    import torch

    ds = Path(dataset_dir)
    data_yaml = _find_yaml(ds)
    if not data_yaml:
        raise FileNotFoundError("data.yaml not found in dataset; ensure YOLO-format dataset")

    print(
        f"[YOLO] Training yolov8s with epochs={epochs}, batch={batch_size}, lr0={lr}, imgsz={img_size}, device={device}, workers={workers}"
    )
    print(f"[YOLO] torch.cuda.is_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        try:
            print(f"[YOLO] GPU: {torch.cuda.get_device_name(0)}")
        except Exception:
            pass
    # Choose weights from a persistent models directory if available
    models_dir = Path(os.getenv("MODELS_DIR", "/app/data/models"))
    models_dir.mkdir(parents=True, exist_ok=True)
    weights_name = os.getenv("YOLO_WEIGHTS", "yolo11s.pt")
    candidate = models_dir / weights_name
    if candidate.exists():
        model_source = str(candidate)
    else:
        model_source = weights_name  # let Ultralytics download & cache internally
    print(f"[YOLO] Using weights: {model_source}")
    model = YOLO(model_source)
    res = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        lr0=lr,
        workers=workers,
        project=str(Path(output_dir) / "runs"),
        name="train",
        exist_ok=True,
    )

    # Typical ultralytics output: runs/detect/train/weights/best.pt or runs/train/weights/best.pt
    out_path = Path(output_dir)
    w = _find_weight_prefer_best(out_path)
    if w:
        return w
    # search job dir as fallback
    w = _find_weight_prefer_best(out_path.parent)
    if w:
        return w
    placeholder = Path(output_dir) / "weights_not_found.txt"
    placeholder.write_text("No weights were detected after YOLO training.")
    return str(placeholder)

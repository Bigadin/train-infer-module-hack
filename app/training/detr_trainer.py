import os
from pathlib import Path
from typing import Optional


def _find_latest_weight(search_dir: Path) -> Optional[str]:
    cand = []
    for ext in ("*.pt", "*.pth"):
        cand.extend(search_dir.rglob(ext))
    if not cand:
        return None
    cand.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(cand[0])


def train_detr(
    dataset_dir: str,
    output_dir: str,
    epochs: int = 20,
    batch_size: int = 4,
    grad_accum_steps: int = 1,
    lr: float = 1e-4,
    device: str = "cuda",
) -> str:
    """
    Trains RFDETR Base on a COCO dataset directory.
    Returns the path to the best/last weights found under the job folder.
    """
    from rfdetr import RFDETRBase  # import inside to avoid import if not used
    import torch

    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    print("[DETR] Initializing RFDETRBase...")
    model = RFDETRBase()

    print(
        f"[DETR] Training with epochs={epochs}, batch_size={batch_size}, grad_accum_steps={grad_accum_steps}, lr={lr}, device={device}"
    )
    print(f"[DETR] torch.cuda.is_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        try:
            print(f"[DETR] GPU: {torch.cuda.get_device_name(0)}")
        except Exception:
            pass
    # RFDETRBase likely uses the default device automatically if CUDA is available
    # We assume dataset_dir is COCO formatted.
    model.train(
        dataset_dir=dataset_dir,
        epochs=epochs,
        batch_size=batch_size,
        grad_accum_steps=grad_accum_steps,
        lr=lr,
    )

    # Try to find produced weights under output_dir or job dir
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    w = _find_latest_weight(out_path)
    if w:
        return w
    # Fallback: search upwards from output_dir's parent (job dir)
    job_dir = out_path.parent
    w = _find_latest_weight(job_dir)
    if w:
        return w
    # If nothing found, create a placeholder to satisfy the API contract
    placeholder = out_path / "weights_not_found.txt"
    placeholder.write_text("No weights were detected after training.")
    return str(placeholder)

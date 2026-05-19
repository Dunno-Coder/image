import csv
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import torch

from config import RESULTS_DIR, WEIGHTS_DIR


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_device(device_arg: str = "auto") -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    requested = torch.device(device_arg)
    if requested.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return requested


def ensure_project_dirs() -> None:
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def weights_path(dataset_name: str, model_name: str) -> Path:
    return WEIGHTS_DIR / f"{dataset_name}_{model_name}_best.pt"


def count_trainable_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def append_results_row(csv_path: Path, row: Dict[str, object]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    normalized = dict(row)
    normalized.setdefault("timestamp", datetime.now().isoformat(timespec="seconds"))

    fieldnames = [
        "timestamp",
        "dataset_name",
        "model_name",
        "split",
        "accuracy",
        "balanced_accuracy",
        "f1",
        "auroc",
        "uncertainty_mean",
        "inference_time_ms",
        "image_size",
        "epochs",
        "batch_size",
        "lr",
        "weights_path",
        "selected_backbone",
        "base_backbone_name",
        "trainable_params",
        "total_params",
    ]

    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(normalized)


def save_checkpoint(
    model: torch.nn.Module,
    path: Path,
    *,
    dataset_name: str,
    model_name: str,
    image_size: int,
    extra: Optional[Dict[str, object]] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset_name": dataset_name,
        "model_name": model_name,
        "image_size": image_size,
        "state_dict": model.state_dict(),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint_state(path: Path, device: torch.device) -> Dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(
            f"Model weights were not found at {path}. Train the model first or place a checkpoint there."
        )
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint
    if isinstance(checkpoint, dict):
        return {"state_dict": checkpoint}
    raise ValueError(f"Unsupported checkpoint format at {path}")


def format_metric(value: object, digits: int = 4) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if np.isnan(v):
        return "n/a"
    return f"{v:.{digits}f}"


def iter_existing(paths: Iterable[Path]) -> Iterable[Path]:
    for path in paths:
        if path.exists():
            yield path

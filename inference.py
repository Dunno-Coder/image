from pathlib import Path
from typing import Dict, Optional

import torch
from PIL import Image

from config import DEFAULT_UNCERTAINTY_THRESHOLD
from dataset import get_eval_transform
from metrics import edl_from_logits
from model import get_model, resolve_base_backbone_name
from utils import get_device, load_checkpoint_state, weights_path


LABELS = {
    0: "Real human-created / real-world image",
    1: "AI-generated image",
}


def load_model_for_inference(
    dataset_name: str,
    model_name: str,
    image_size: int,
    device_arg: str = "auto",
    checkpoint_path: Optional[Path] = None,
    require_weights: bool = True,
    base_backbone_name: str = "auto",
) -> torch.nn.Module:
    device = get_device(device_arg)

    ckpt_path = checkpoint_path or weights_path(dataset_name, model_name)
    checkpoint = None
    if require_weights or ckpt_path.exists():
        checkpoint = load_checkpoint_state(ckpt_path, device)
        ckpt_image_size = checkpoint.get("image_size")
        if ckpt_image_size is not None and int(ckpt_image_size) != int(image_size):
            raise ValueError(
                f"Checkpoint was trained with image_size={ckpt_image_size}, "
                f"but the app requested image_size={image_size}."
            )

    selected_backbone = checkpoint.get("selected_backbone") if checkpoint else None
    checkpoint_base_backbone = checkpoint.get("base_backbone_name") if checkpoint else None
    resolved_base_backbone = base_backbone_name
    if model_name == "proposed_mnff_edl":
        resolved_base_backbone = str(checkpoint_base_backbone or resolve_base_backbone_name(base_backbone_name))
    model = get_model(
        model_name,
        pretrained=False,
        freeze_backbone=True,
        base_backbone_name=resolved_base_backbone,
        forced_backbone_name=selected_backbone if selected_backbone else None,
        image_size=image_size,
    ).to(device)

    if checkpoint is not None:
        model.load_state_dict(checkpoint["state_dict"])

    model.eval()
    return model


@torch.no_grad()
def predict_pil_image(
    model: torch.nn.Module,
    image: Image.Image,
    image_size: int,
    device_arg: str = "auto",
    uncertainty_threshold: float = DEFAULT_UNCERTAINTY_THRESHOLD,
) -> Dict[str, object]:
    device = get_device(device_arg)
    transform = get_eval_transform(image_size)
    tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)

    logits = model(tensor)
    outputs = edl_from_logits(logits)
    probs = outputs["prob"][0].detach().cpu()
    uncertainty = float(outputs["uncertainty"][0].detach().cpu())
    confidence = float(outputs["confidence"][0].detach().cpu())
    pred_idx = int(outputs["predicted_class"][0].detach().cpu())

    action = "manual_review" if uncertainty > uncertainty_threshold else "auto_decision"
    return {
        "label": LABELS[pred_idx],
        "class_index": pred_idx,
        "prob_real": float(probs[0]),
        "prob_ai": float(probs[1]),
        "confidence": confidence,
        "uncertainty": uncertainty,
        "action": action,
    }

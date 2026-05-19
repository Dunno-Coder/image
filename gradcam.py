from typing import Optional

import numpy as np
import torch
from PIL import Image

from dataset import get_eval_transform
from metrics import edl_from_logits
from utils import get_device


def _normalize_map(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x = x - float(x.min())
    denom = float(x.max()) + 1e-8
    return x / denom


def _overlay_heatmap(base: Image.Image, heatmap: np.ndarray) -> Image.Image:
    base_rgb = base.convert("RGB")
    heat_img = Image.fromarray(np.uint8(_normalize_map(heatmap) * 255)).resize(base_rgb.size)

    try:
        import cv2

        heat_np = np.asarray(heat_img)
        colored = cv2.applyColorMap(heat_np, cv2.COLORMAP_JET)
        colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
        base_np = np.asarray(base_rgb).astype(np.float32)
        overlay = (0.55 * base_np + 0.45 * colored.astype(np.float32)).clip(0, 255).astype(np.uint8)
        return Image.fromarray(overlay)
    except Exception:
        red = Image.new("RGB", base_rgb.size, (255, 0, 0))
        mask = heat_img.convert("L")
        colored = Image.composite(red, base_rgb, mask)
        return Image.blend(base_rgb, colored, 0.45)


def generate_heatmap(
    model: torch.nn.Module,
    image: Image.Image,
    image_size: int,
    device_arg: str = "auto",
    class_index: Optional[int] = None,
) -> Optional[Image.Image]:
    """Return a gradient saliency overlay.

    This fallback works for ViT-style foundation backbones when classic Grad-CAM
    target layers are not obvious. Failures are swallowed so Streamlit remains
    usable during demos.
    """
    try:
        device = get_device(device_arg)
        model.eval()
        transform = get_eval_transform(image_size)
        tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
        tensor.requires_grad_(True)

        model.zero_grad(set_to_none=True)
        logits = model(tensor)
        outputs = edl_from_logits(logits)
        if class_index is None:
            class_index = int(outputs["predicted_class"][0].detach().cpu())
        score = logits[0, class_index]
        score.backward()

        if tensor.grad is None:
            return None
        saliency = tensor.grad.detach().abs().mean(dim=1)[0].cpu().numpy()
        return _overlay_heatmap(image, saliency)
    except Exception:
        return None

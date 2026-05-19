import time
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score


def edl_from_logits(logits: torch.Tensor, num_classes: int = 2) -> Dict[str, torch.Tensor]:
    evidence = F.softplus(logits)
    alpha = evidence + 1.0
    strength = alpha.sum(dim=1, keepdim=True)
    prob = alpha / strength
    uncertainty = num_classes / strength
    confidence = 1.0 - uncertainty
    predicted_class = torch.argmax(prob, dim=1)
    return {
        "evidence": evidence,
        "alpha": alpha,
        "prob": prob,
        "uncertainty": uncertainty.squeeze(1),
        "confidence": confidence.squeeze(1),
        "predicted_class": predicted_class,
    }


def edl_nll_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    outputs = edl_from_logits(logits, num_classes=2)
    probs = outputs["prob"].clamp_min(1e-8)
    return F.nll_loss(torch.log(probs), targets)


def compute_binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob_ai: np.ndarray,
    uncertainty: np.ndarray,
    inference_time_ms: float,
) -> Dict[str, float]:
    if len(y_true) == 0:
        raise ValueError("Cannot compute metrics on an empty dataset.")

    try:
        auroc = float(roc_auc_score(y_true, y_prob_ai))
    except ValueError:
        auroc = float("nan")

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "auroc": auroc,
        "uncertainty_mean": float(np.mean(uncertainty)),
        "inference_time_ms": float(inference_time_ms),
    }


@torch.no_grad()
def evaluate_loader(model: torch.nn.Module, loader, device: torch.device) -> Dict[str, float]:
    model.eval()
    y_true = []
    y_pred = []
    y_prob_ai = []
    uncertainties = []
    total_inference_ms = 0.0
    total_images = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        logits = model(images)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        outputs = edl_from_logits(logits)
        probs = outputs["prob"]

        batch_size = images.size(0)
        total_images += batch_size
        total_inference_ms += elapsed_ms

        y_true.extend(targets.detach().cpu().numpy().tolist())
        y_pred.extend(outputs["predicted_class"].detach().cpu().numpy().tolist())
        y_prob_ai.extend(probs[:, 1].detach().cpu().numpy().tolist())
        uncertainties.extend(outputs["uncertainty"].detach().cpu().numpy().tolist())

    avg_time_ms = total_inference_ms / max(total_images, 1)
    return compute_binary_metrics(
        np.asarray(y_true),
        np.asarray(y_pred),
        np.asarray(y_prob_ai),
        np.asarray(uncertainties),
        avg_time_ms,
    )


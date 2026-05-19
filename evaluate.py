import argparse
from pathlib import Path

import torch

from config import RESULTS_CSV
from dataset import get_dataloaders
from metrics import evaluate_loader
from model import get_model
from utils import (
    append_results_row,
    count_total_parameters,
    count_trainable_parameters,
    ensure_project_dirs,
    get_device,
    load_checkpoint_state,
    weights_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained AI-generated vs real image classifier.")
    parser.add_argument("--dataset_name", required=True, choices=["genimage", "cifake", "ai_vs_real"])
    parser.add_argument("--model_name", required=True, choices=["dinov3_linear", "siglip2_linear", "aimv2_linear", "proposed_mnff_edl"])
    parser.add_argument("--batch_size", required=True, type=int)
    parser.add_argument("--image_size", required=True, type=int)
    parser.add_argument("--device", required=True, help="Use 'auto', 'cpu', or 'cuda'.")
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument("--weights", default=None, type=str)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_project_dirs()
    device = get_device(args.device)
    _, _, test_loader = get_dataloaders(
        dataset_name=args.dataset_name,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    ckpt_path = Path(args.weights) if args.weights else weights_path(args.dataset_name, args.model_name)
    checkpoint = load_checkpoint_state(ckpt_path, device)
    selected_backbone = checkpoint.get("selected_backbone")
    base_backbone_name = checkpoint.get("base_backbone_name", "auto")
    model = get_model(
        args.model_name,
        pretrained=False,
        freeze_backbone=True,
        base_backbone_name=str(base_backbone_name),
        forced_backbone_name=selected_backbone if selected_backbone else None,
        image_size=args.image_size,
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])

    metrics = evaluate_loader(model, test_loader, device)
    selected_backbone = selected_backbone or getattr(model, "selected_backbone_name", args.model_name)

    append_results_row(
        RESULTS_CSV,
        {
            "dataset_name": args.dataset_name,
            "model_name": args.model_name,
            "split": "test",
            **metrics,
            "image_size": args.image_size,
            "epochs": "",
            "batch_size": args.batch_size,
            "lr": "",
            "weights_path": str(ckpt_path),
            "selected_backbone": selected_backbone,
            "base_backbone_name": base_backbone_name,
            "trainable_params": count_trainable_parameters(model),
            "total_params": count_total_parameters(model),
        },
    )

    print(metrics)
    print(f"Appended results to {RESULTS_CSV}")


if __name__ == "__main__":
    main()

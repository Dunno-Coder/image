import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from config import RESULTS_CSV
from dataset import get_dataloaders
from metrics import edl_nll_loss, evaluate_loader
from model import get_model, resolve_base_backbone_name
from utils import (
    append_results_row,
    count_total_parameters,
    count_trainable_parameters,
    ensure_project_dirs,
    get_device,
    save_checkpoint,
    set_seed,
    weights_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train AI-generated vs real image classifiers.")
    parser.add_argument("--dataset_name", required=True, choices=["genimage", "cifake", "ai_vs_real"])
    parser.add_argument("--model_name", required=True, choices=["dinov3_linear", "siglip2_linear", "aimv2_linear", "proposed_mnff_edl"])
    parser.add_argument("--epochs", required=True, type=int)
    parser.add_argument("--batch_size", required=True, type=int)
    parser.add_argument("--lr", required=True, type=float)
    parser.add_argument("--image_size", required=True, type=int)
    parser.add_argument("--device", required=True, help="Use 'auto', 'cpu', or 'cuda'.")
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--no_pretrained", action="store_true")
    parser.add_argument("--finetune_backbone", action="store_true")
    parser.add_argument(
        "--base_backbone_name",
        default="auto",
        choices=["auto", "dinov3_linear", "siglip2_linear", "aimv2_linear"],
        help="Base backbone for proposed_mnff_edl. 'auto' uses results/best_backbone_config.json.",
    )
    parser.add_argument(
        "--proposed_backbone",
        default=None,
        choices=["auto", "dinov3_linear", "siglip2_linear", "aimv2_linear"],
        help="Deprecated alias for --base_backbone_name.",
    )
    return parser.parse_args()


def resolve_training_base_backbone(args: argparse.Namespace) -> str:
    requested = args.proposed_backbone if args.proposed_backbone is not None else args.base_backbone_name
    return resolve_base_backbone_name(requested)


def train_one_epoch(model, loader, optimizer, device, use_edl_loss: bool) -> float:
    model.train()
    running_loss = 0.0
    num_samples = 0

    for images, targets in tqdm(loader, desc="train", leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = edl_nll_loss(logits, targets) if use_edl_loss else F.cross_entropy(logits, targets)
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        num_samples += batch_size

    return running_loss / max(num_samples, 1)


def main() -> None:
    args = parse_args()
    ensure_project_dirs()
    set_seed(args.seed)
    device = get_device(args.device)

    train_loader, val_loader, test_loader = get_dataloaders(
        dataset_name=args.dataset_name,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    base_backbone_name = resolve_training_base_backbone(args)
    model = get_model(
        args.model_name,
        pretrained=not args.no_pretrained,
        freeze_backbone=not args.finetune_backbone,
        base_backbone_name=base_backbone_name,
        image_size=args.image_size,
    ).to(device)

    trainable_params = count_trainable_parameters(model)
    total_params = count_total_parameters(model)
    selected_backbone = getattr(model, "selected_backbone_name", args.model_name)
    print(f"Device: {device}")
    if args.model_name == "proposed_mnff_edl":
        print(f"Proposed base backbone key: {base_backbone_name}")
    print(f"Selected backbone: {selected_backbone}")
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=1e-4,
    )

    best_f1 = -1.0
    best_path = weights_path(args.dataset_name, args.model_name)
    use_edl_loss = args.model_name == "proposed_mnff_edl"

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, use_edl_loss)
        val_metrics = evaluate_loader(model, val_loader, device)
        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} "
            f"loss={train_loss:.4f} val_f1={val_metrics['f1']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f}"
        )

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            save_checkpoint(
                model,
                best_path,
                dataset_name=args.dataset_name,
                model_name=args.model_name,
                image_size=args.image_size,
                extra={
                    "best_val_f1": best_f1,
                    "selected_backbone": selected_backbone,
                    "base_backbone_name": base_backbone_name,
                    "epoch": epoch,
                },
            )
            print(f"Saved new best checkpoint: {best_path}")

    try:
        checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    test_metrics = evaluate_loader(model, test_loader, device)

    append_results_row(
        RESULTS_CSV,
        {
            "dataset_name": args.dataset_name,
            "model_name": args.model_name,
            "split": "test",
            **test_metrics,
            "image_size": args.image_size,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weights_path": str(Path(best_path)),
            "selected_backbone": selected_backbone,
            "base_backbone_name": base_backbone_name,
            "trainable_params": trainable_params,
            "total_params": total_params,
        },
    )
    print(f"Best validation F1: {best_f1:.4f}")
    print(f"Test metrics: {test_metrics}")
    print(f"Appended results to {RESULTS_CSV}")


if __name__ == "__main__":
    main()

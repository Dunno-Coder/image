from pathlib import Path
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from config import (
    CLASS_TO_IDX,
    DATA_DIR,
    IMAGENET_MEAN,
    IMAGENET_STD,
    SUPPORTED_DATASETS,
)
from utils import seed_worker


def get_train_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def get_eval_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def _remap_imagefolder_targets(dataset: datasets.ImageFolder) -> datasets.ImageFolder:
    missing = [name for name in CLASS_TO_IDX if name not in dataset.class_to_idx]
    if missing:
        raise ValueError(
            f"Dataset split at {dataset.root} is missing class folder(s): {missing}. "
            "Expected folders named 'real' and 'ai'."
        )

    remapped_samples = []
    for image_path, _ in dataset.samples:
        class_name = Path(image_path).parent.name.lower()
        if class_name not in CLASS_TO_IDX:
            raise ValueError(f"Unexpected class folder '{class_name}' in {image_path}")
        remapped_samples.append((image_path, CLASS_TO_IDX[class_name]))

    dataset.samples = remapped_samples
    dataset.imgs = remapped_samples
    dataset.targets = [target for _, target in remapped_samples]
    dataset.class_to_idx = dict(CLASS_TO_IDX)
    dataset.classes = ["real", "ai"]
    return dataset


def build_imagefolder(split_dir: Path, transform: transforms.Compose) -> datasets.ImageFolder:
    if not split_dir.exists():
        raise FileNotFoundError(
            f"Dataset split folder not found: {split_dir}. "
            "Expected datasets/{dataset_name}/{train,val,test}/{real,ai}."
        )
    dataset = datasets.ImageFolder(root=str(split_dir), transform=transform)
    return _remap_imagefolder_targets(dataset)


def get_dataloaders(
    dataset_name: str,
    image_size: int = 224,
    batch_size: int = 16,
    num_workers: int = 2,
    data_dir: Path = DATA_DIR,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    if dataset_name not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported dataset_name '{dataset_name}'. Use one of: {SUPPORTED_DATASETS}")

    dataset_root = data_dir / dataset_name
    train_dataset = build_imagefolder(dataset_root / "train", get_train_transform(image_size))
    val_dataset = build_imagefolder(dataset_root / "val", get_eval_transform(image_size))
    test_dataset = build_imagefolder(dataset_root / "test", get_eval_transform(image_size))

    generator = torch.Generator()
    generator.manual_seed(42)

    common_kwargs: Dict[str, object] = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "worker_init_fn": seed_worker,
        "generator": generator,
    }

    train_loader = DataLoader(train_dataset, shuffle=True, **common_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **common_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **common_kwargs)
    return train_loader, val_loader, test_loader


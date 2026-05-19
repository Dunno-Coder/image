import json
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import BEST_BACKBONE_CONFIG_JSON, DEFAULT_PROPOSED_BACKBONE, NUM_CLASSES, SUPPORTED_MODELS


BACKBONE_CANDIDATES: Dict[str, List[str]] = {
    # DINOv3 names are kept first for environments that provide them.
    # DINOv2 and strong ViT checkpoints are fallbacks for timm installations
    # that do not yet expose DINOv3.
    "dinov3_linear": [
        "vit_base_patch16_224.dinov3",
        "vit_base_patch14_dinov2.lvd142m",
        "vit_small_patch14_dinov2.lvd142m",
        "vit_base_patch16_224.augreg_in21k_ft_in1k",
    ],
    "siglip2_linear": [
        "vit_base_patch16_siglip_224.v2_webli",
        "vit_base_patch16_siglip_224.webli",
        "vit_base_patch16_clip_224.laion2b_ft_in12k_in1k",
        "vit_base_patch16_224.augreg_in21k_ft_in1k",
    ],
    "aimv2_linear": [
        "vit_base_patch14_aimv2_224",
        "vit_large_patch14_aimv2_224",
        "vit_base_patch16_224.mae",
        "vit_base_patch16_224.augreg_in21k_ft_in1k",
    ],
}

SUPPORTED_BASE_BACKBONES = ("dinov3_linear", "siglip2_linear", "aimv2_linear")


def resolve_base_backbone_name(base_backbone_name: str = "auto") -> str:
    """Resolve the logical foundation backbone key for proposed_mnff_edl."""
    if base_backbone_name != "auto":
        if base_backbone_name not in SUPPORTED_BASE_BACKBONES:
            raise ValueError(
                f"Unsupported base_backbone_name '{base_backbone_name}'. "
                f"Use one of: ('auto', {', '.join(SUPPORTED_BASE_BACKBONES)})"
            )
        return base_backbone_name

    if BEST_BACKBONE_CONFIG_JSON.exists():
        try:
            payload = json.loads(BEST_BACKBONE_CONFIG_JSON.read_text(encoding="utf-8"))
            selected = str(payload.get("best_model_name", "")).strip()
            if selected in SUPPORTED_BASE_BACKBONES:
                return selected
        except Exception:
            pass

    return DEFAULT_PROPOSED_BACKBONE


class MNFFModule(nn.Module):
    """Multi-Nonlinear Feature Fusion for subtle artifact enhancement."""

    def __init__(self, channels: int = 3) -> None:
        super().__init__()
        self.conv_d1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, dilation=1)
        self.conv_d2 = nn.Conv2d(channels, channels, kernel_size=3, padding=2, dilation=2)
        self.conv_d4 = nn.Conv2d(channels, channels, kernel_size=3, padding=4, dilation=4)
        self.restore = nn.Conv2d(channels * 3, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gelu_x = F.gelu(x)
        elu_x = F.elu(x)
        relu_x = F.relu(x)
        x_f = gelu_x * elu_x * relu_x
        multi_scale = torch.cat(
            [self.conv_d1(x_f), self.conv_d2(x_f), self.conv_d4(x_f)],
            dim=1,
        )
        return self.restore(multi_scale)


class TimmFeatureBackbone(nn.Module):
    def __init__(
        self,
        model_key: str,
        pretrained: bool = True,
        freeze: bool = True,
        forced_model_name: Optional[str] = None,
        image_size: int = 224,
    ) -> None:
        super().__init__()
        try:
            import timm
        except ImportError as exc:
            raise ImportError(
                "The 'timm' package is required for the foundation backbone fallback. "
                "Install requirements.txt first."
            ) from exc

        if model_key not in BACKBONE_CANDIDATES:
            raise ValueError(f"Unknown backbone key '{model_key}'.")

        errors = []
        self.selected_backbone_name: Optional[str] = None
        self.backbone: Optional[nn.Module] = None
        self.out_dim: Optional[int] = None
        candidates = [forced_model_name] if forced_model_name else BACKBONE_CANDIDATES[model_key]
        self.image_size = image_size

        for candidate in candidates:
            try:
                self.backbone = self._create_timm_model(timm, candidate, pretrained, image_size)
                self.out_dim = self._infer_feature_dim()
                self.selected_backbone_name = candidate
                break
            except Exception as exc:
                errors.append(f"{candidate}: {exc}")
                self.backbone = None
                self.out_dim = None

        if self.backbone is None:
            raise RuntimeError(
                "Could not create any candidate backbone for "
                f"{model_key}. Tried: {candidates}. "
                f"Errors: {' | '.join(errors)}"
            )

        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()

    @staticmethod
    def _create_timm_model(timm, candidate: str, pretrained: bool, image_size: int) -> nn.Module:
        attempts = [
            {
                "pretrained": pretrained,
                "num_classes": 0,
                "global_pool": "avg",
                "img_size": image_size,
            },
            {
                "pretrained": pretrained,
                "num_classes": 0,
                "global_pool": "avg",
            },
            {
                "pretrained": pretrained,
                "num_classes": 0,
            },
        ]
        last_error = None
        for kwargs in attempts:
            try:
                return timm.create_model(candidate, **kwargs)
            except TypeError as exc:
                last_error = exc
        raise last_error if last_error is not None else RuntimeError(f"Could not create {candidate}")

    def train(self, mode: bool = True):
        super().train(mode)
        if self.backbone is not None and all(not p.requires_grad for p in self.backbone.parameters()):
            self.backbone.eval()
        return self

    def _infer_feature_dim(self) -> int:
        assert self.backbone is not None
        num_features = getattr(self.backbone, "num_features", None)
        if isinstance(num_features, int) and num_features > 0:
            return num_features

        with torch.no_grad():
            dummy = torch.zeros(1, 3, self.image_size, self.image_size)
            features = self.forward(dummy)
        return int(features.shape[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert self.backbone is not None
        features = self.backbone(x)
        if isinstance(features, (tuple, list)):
            features = features[0]
        if isinstance(features, dict):
            for key in ("pooled_output", "last_hidden_state", "features"):
                if key in features:
                    features = features[key]
                    break
        if features.ndim == 4:
            features = features.mean(dim=(2, 3))
        elif features.ndim == 3:
            features = features[:, 0]
        return features.flatten(1)


class LinearFoundationClassifier(nn.Module):
    def __init__(
        self,
        model_key: str,
        num_classes: int = NUM_CLASSES,
        pretrained: bool = True,
        freeze_backbone: bool = True,
        forced_backbone_name: Optional[str] = None,
        image_size: int = 224,
    ) -> None:
        super().__init__()
        self.model_key = model_key
        self.feature_extractor = TimmFeatureBackbone(
            model_key=model_key,
            pretrained=pretrained,
            freeze=freeze_backbone,
            forced_model_name=forced_backbone_name,
            image_size=image_size,
        )
        feature_dim = int(self.feature_extractor.out_dim)
        self.classifier = nn.Linear(feature_dim, num_classes)

    @property
    def selected_backbone_name(self) -> str:
        return self.feature_extractor.selected_backbone_name or self.model_key

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)
        return self.classifier(features)


class ProposedMNFFEDLClassifier(nn.Module):
    def __init__(
        self,
        base_backbone_name: str = DEFAULT_PROPOSED_BACKBONE,
        num_classes: int = NUM_CLASSES,
        pretrained: bool = True,
        freeze_backbone: bool = True,
        forced_backbone_name: Optional[str] = None,
        image_size: int = 224,
    ) -> None:
        super().__init__()
        if base_backbone_name == "proposed_mnff_edl":
            base_backbone_name = DEFAULT_PROPOSED_BACKBONE
        base_backbone_name = resolve_base_backbone_name(base_backbone_name)
        self.model_key = "proposed_mnff_edl"
        self.base_backbone_name = base_backbone_name
        self.backbone_key = base_backbone_name
        self.mnff = MNFFModule(channels=3)
        self.feature_extractor = TimmFeatureBackbone(
            model_key=base_backbone_name,
            pretrained=pretrained,
            freeze=freeze_backbone,
            forced_model_name=forced_backbone_name,
            image_size=image_size,
        )
        feature_dim = int(self.feature_extractor.out_dim)
        self.classifier = nn.Linear(feature_dim, num_classes)

    @property
    def selected_backbone_name(self) -> str:
        return self.feature_extractor.selected_backbone_name or self.backbone_key

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enhanced = self.mnff(x)
        features = self.feature_extractor(enhanced)
        return self.classifier(features)


def get_model(
    model_name: str,
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
    freeze_backbone: bool = True,
    proposed_backbone: str = DEFAULT_PROPOSED_BACKBONE,
    base_backbone_name: str = "auto",
    forced_backbone_name: Optional[str] = None,
    image_size: int = 224,
) -> nn.Module:
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model_name '{model_name}'. Use one of: {SUPPORTED_MODELS}")

    if model_name == "proposed_mnff_edl":
        if base_backbone_name == "auto" and proposed_backbone != DEFAULT_PROPOSED_BACKBONE:
            base_backbone_name = proposed_backbone
        return ProposedMNFFEDLClassifier(
            base_backbone_name=base_backbone_name,
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            forced_backbone_name=forced_backbone_name,
            image_size=image_size,
        )

    return LinearFoundationClassifier(
        model_key=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        forced_backbone_name=forced_backbone_name,
        image_size=image_size,
    )

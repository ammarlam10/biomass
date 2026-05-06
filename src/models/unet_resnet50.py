"""
UNet with ResNet50 encoder – baseline pixel-wise regression model.

Uses segmentation-models-pytorch (smp) which handles the first-conv weight
adaptation when in_channels != 3 automatically.

Output: [B, 2, H, W]  (tree_count channel, mean_height channel)
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

from src.models.factory import register_model


@register_model("unet_resnet50")
def build_unet_resnet50(cfg: Dict, num_input_channels: int) -> nn.Module:
    unet_cfg = cfg.get("model", {}).get("unet", {})
    targets = cfg.get("data", {}).get("targets", ["tree_count", "mean_height"])
    return UNetResNet50(
        in_channels=num_input_channels,
        encoder_name=unet_cfg.get("encoder_name", "resnet50"),
        encoder_weights=unet_cfg.get("encoder_weights", "imagenet"),
        decoder_channels=unet_cfg.get("decoder_channels", [256, 128, 64, 32, 16]),
        num_targets=len(targets),
    )


class UNetResNet50(nn.Module):
    """
    Thin wrapper around smp.Unet that exposes a clean forward signature.

    The smp library adapts the first convolutional layer to accept
    `in_channels` inputs while optionally preserving imagenet-pretrained
    weights for the remaining encoder layers.
    """

    def __init__(
        self,
        in_channels: int,
        encoder_name: str = "resnet50",
        encoder_weights: str = "imagenet",
        decoder_channels: list | None = None,
        num_targets: int = 2,
    ) -> None:
        super().__init__()
        if decoder_channels is None:
            decoder_channels = [256, 128, 64, 32, 16]

        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_targets,
            activation=None,     # raw regression – no sigmoid/softmax
            decoder_channels=decoder_channels,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
        Returns:
            pred: [B, 2, H, W]
        """
        return self.model(x)

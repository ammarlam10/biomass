"""
UNet with ResNet50 encoder – baseline pixel-wise regression model.

Uses segmentation-models-pytorch (smp) which handles the first-conv weight
adaptation when in_channels != 3 automatically.

Output: [B, 2, H, W]  (tree_count channel, mean_height channel)

Multi-task variant (unet_resnet50_multitask):
  forward() returns (primary, auxiliary) where auxiliary is the prediction of
  frozen ResNet50 DOP20 feature maps used as an auxiliary reconstruction target.
  At inference only primary is needed; auxiliary can be discarded.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

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


@register_model("unet_resnet50_multitask")
def build_unet_resnet50_multitask(cfg: Dict, num_input_channels: int) -> nn.Module:
    unet_cfg = cfg.get("model", {}).get("unet", {})
    aux_cfg = cfg.get("model", {}).get("aux", {})
    targets = cfg.get("data", {}).get("targets", ["tree_count", "mean_height"])
    return UNetResNet50MultiTask(
        in_channels=num_input_channels,
        encoder_name=unet_cfg.get("encoder_name", "resnet50"),
        encoder_weights=unet_cfg.get("encoder_weights", "imagenet"),
        decoder_channels=unet_cfg.get("decoder_channels", [256, 128, 64, 32, 16]),
        num_targets=len(targets),
        aux_feature_dim=aux_cfg.get("feature_dim", 256),
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
            pred: [B, num_targets, H, W]
        """
        return self.model(x)


class UNetResNet50MultiTask(nn.Module):
    """
    Multi-task UNet variant with two output heads:

    - **Primary head** (``segmentation_head``): predicts LiDAR targets
      [B, num_targets, H, W] — identical to the single-task model.
    - **Auxiliary head** (``aux_head``): predicts DOP20 RGB feature maps
      [B, aux_feature_dim, H, W] — used only during training.

    ``forward()`` returns a tuple ``(primary, auxiliary)`` so the trainer
    can compute both losses. At inference, discard ``auxiliary`` and use
    only ``primary``.

    The encoder and decoder are shared between both heads; only the final
    Conv2d layers differ.
    """

    def __init__(
        self,
        in_channels: int,
        encoder_name: str = "resnet50",
        encoder_weights: str = "imagenet",
        decoder_channels: Optional[List[int]] = None,
        num_targets: int = 2,
        aux_feature_dim: int = 256,
    ) -> None:
        super().__init__()
        if decoder_channels is None:
            decoder_channels = [256, 128, 64, 32, 16]

        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_targets,
            activation=None,
            decoder_channels=decoder_channels,
        )

        last_ch = decoder_channels[-1]
        self.aux_head = nn.Conv2d(last_ch, aux_feature_dim, kernel_size=1)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, C, H, W]
        Returns:
            primary  : [B, num_targets, H, W]  — LiDAR regression targets
            auxiliary: [B, aux_feature_dim, H, W] — DOP20 feature reconstruction
        """
        features = self.model.encoder(x)
        decoder_out = self.model.decoder(features)
        primary = self.model.segmentation_head(decoder_out)
        auxiliary = self.aux_head(decoder_out)
        return primary, auxiliary

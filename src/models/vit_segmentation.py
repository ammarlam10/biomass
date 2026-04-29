"""
ViT-based pixel regression model – Stage 1 future extension stub.

Planned implementation options:
  A) SegFormer (MiT encoder + lightweight MLP decoder) via HuggingFace
       transformers.SegformerForSemanticSegmentation adapted for regression
  B) ViT-UNet hybrid using timm ViT encoder + UNet-style decoder

Required interface (identical to all other models):
    forward(x: Tensor[B, C, H, W]) -> Tensor[B, 2, H, W]

Implementation notes for Stage 1:
  - Patch embed size must align with 128x128 input
    (e.g. patch_size=16 → 8x8 = 64 tokens for SegFormer)
  - Output must be bilinearly upsampled to [B, 2, 128, 128]
  - Species channel can be passed as an extra token embedding or ignored
  - Use timm or HuggingFace weights for pretrained initialisation
  - Add to src/models/__init__.py to auto-register
"""

from __future__ import annotations

import torch.nn as nn

from src.models.factory import register_model


@register_model("vit_segmentation")
def build_vit_segmentation(cfg: dict, num_input_channels: int) -> nn.Module:
    raise NotImplementedError(
        "ViT segmentation model is not yet implemented (Stage 1 roadmap). "
        "See src/models/vit_segmentation.py for planned architecture notes."
    )


class ViTSegmentation(nn.Module):
    """
    Placeholder for ViT-based pixel regression.

    Interface: forward(x: Tensor[B, C, H, W]) -> Tensor[B, 2, H, W]
    """

    def __init__(self) -> None:
        super().__init__()

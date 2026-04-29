"""
Prithvi foundation model adapter – Stage 2 future extension stub.

Prithvi (ibm-nasa-geospatial/Prithvi-100M on HuggingFace) is a geospatial
foundation model pretrained on Sentinel-2 time-series via masked-patch
autoencoding.

Required interface (identical to all other models):
    forward(x: Tensor[B, C, H, W]) -> Tensor[B, 2, H, W]

Implementation plan for Stage 2:
  1. Load pretrained backbone:
       from transformers import AutoModel
       backbone = AutoModel.from_pretrained("ibm-nasa-geospatial/Prithvi-100M")
  2. Channel adapter:
       - Prithvi expects 6-band HLS input (B02 B03 B04 B8A B11 B12) per frame
       - Map our S2 bands to Prithvi's expected bands; ignore S1 or concatenate
         as extra tokens
       - Optional: learnable linear projector from N-channel input to 6 channels
  3. Temporal adapter:
       - Prithvi accepts T time frames; map our 4 seasons to T=4 frames
  4. Regression head:
       - Attach a lightweight decoder/head on top of Prithvi features
       - Output bilinearly upsampled to [B, 2, 128, 128]
  5. Freezing schedule:
       - Phase 1: freeze backbone, train head only
       - Phase 2: unfreeze last N transformer blocks
       - Phase 3: full fine-tune (optional)

Config key: model.name = prithvi_adapter
Model-specific params live under model.prithvi.*
"""

from __future__ import annotations

import torch.nn as nn

from src.models.factory import register_model


@register_model("prithvi_adapter")
def build_prithvi_adapter(cfg: dict, num_input_channels: int) -> nn.Module:
    raise NotImplementedError(
        "Prithvi adapter is not yet implemented (Stage 2 roadmap). "
        "See src/models/prithvi_adapter.py for the planned implementation notes."
    )


class PrithviAdapter(nn.Module):
    """
    Placeholder for IBM/NASA Prithvi foundation model adapter.

    Interface: forward(x: Tensor[B, C, H, W]) -> Tensor[B, 2, H, W]
    """

    def __init__(self) -> None:
        super().__init__()

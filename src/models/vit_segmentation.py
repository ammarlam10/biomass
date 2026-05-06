"""
SegFormer-style dense regression model.

Architecture:
  Encoder : Any timm backbone with ``features_only=True`` that returns four
            spatial feature maps at H/4, H/8, H/16, H/32 (same layout as
            SegFormer / MiT).  The timm wheels bundled with PyTorch 2.1.x do
            **not** register ``mit_b0`` … ``mit_b5``; the default is therefore
            **PVTv2-B2** (``pvt_v2_b2``), a hierarchical Pyramid Vision
            Transformer with the same per-stage channel widths
            ``[64, 128, 320, 512]`` as MiT-B2, so the All-MLP decoder is unchanged.
  Decoder : All-MLP head (SegFormer-style)
            Each scale is projected to embed_dim, bilinearly upsampled to H/4,
            concatenated, fused with a single MLP, then upsampled ×4 to H.

Interface (identical to all other registered models):
    forward(x: Tensor[B, C, H, W]) -> Tensor[B, 2, H, W]
where channel 0 = tree_count, channel 1 = mean_height (both in transform space).

Config keys consumed from cfg['model']['vit']:
    encoder_name : timm model name  (default: 'pvt_v2_b2')
    embed_dim    : MLP decoder width (default: 256)
    pretrained   : load ImageNet weights (default: True)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from src.models.factory import register_model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_encoder_channels(model: nn.Module) -> list[int]:
    """Return the output channel counts for each feature-map stage."""
    fi = model.feature_info
    return [f["num_chs"] for f in fi]


class _MLPProjection(nn.Sequential):
    """Linear channel projection + LayerNorm, operating on [B, C, H, W]."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class SegFormerRegressor(nn.Module):
    """
    SegFormer-style dense regression adapted for two continuous targets.

    Encoder produces 4 feature scales; the All-MLP decoder fuses them and
    produces a dense output upsampled to the original spatial resolution.
    """

    def __init__(
        self,
        in_channels: int,
        encoder_name: str = "pvt_v2_b2",
        embed_dim: int = 256,
        pretrained: bool = True,
        num_targets: int = 2,
    ) -> None:
        super().__init__()

        # ── Encoder ───────────────────────────────────────────────────────
        try:
            self.encoder = timm.create_model(
                encoder_name,
                pretrained=pretrained,
                in_chans=in_channels,
                features_only=True,
            )
        except RuntimeError as e:
            if "Unknown model" in str(e):
                raise RuntimeError(
                    f"Unknown timm encoder '{encoder_name}'. "
                    f"SegFormer MiT names (mit_b0 … mit_b5) are not available in "
                    f"this timm build; try 'pvt_v2_b2' (default), 'pvt_v2_b1', "
                    f"or 'pvt_v2_b3'."
                ) from e
            raise
        enc_channels = _get_encoder_channels(self.encoder)  # e.g. [64,128,320,512]

        # ── Per-scale MLP projections ──────────────────────────────────────
        self.proj = nn.ModuleList(
            [_MLPProjection(c, embed_dim) for c in enc_channels]
        )

        # ── Fuse MLP (after concat of all scales at H/4) ──────────────────
        fuse_in = embed_dim * len(enc_channels)
        self.fuse = nn.Sequential(
            nn.Conv2d(fuse_in, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )

        # ── Regression head ───────────────────────────────────────────────
        self.head = nn.Conv2d(embed_dim, num_targets, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
        Returns:
            pred: [B, 2, H, W]  (tree_count, mean_height) in transform space
        """
        H, W = x.shape[2], x.shape[3]

        # Multi-scale features: list of [B, Ci, Hi, Wi]
        features = self.encoder(x)

        # Target spatial size for decoder = H/4 (coarsest useful scale)
        target_h, target_w = features[0].shape[2], features[0].shape[3]

        # Project each scale to embed_dim and upsample to H/4
        projected = []
        for feat, proj_layer in zip(features, self.proj):
            p = proj_layer(feat)
            if p.shape[2:] != (target_h, target_w):
                p = F.interpolate(
                    p, size=(target_h, target_w), mode="bilinear", align_corners=False
                )
            projected.append(p)

        # Fuse all scales
        fused = self.fuse(torch.cat(projected, dim=1))  # [B, embed_dim, H/4, W/4]

        # Head → regression output
        out = self.head(fused)  # [B, 2, H/4, W/4]

        # Upsample to original resolution
        out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)
        return out


# ---------------------------------------------------------------------------
# Registry builder
# ---------------------------------------------------------------------------

@register_model("vit_segmentation")
def build_vit_segmentation(cfg: dict, num_input_channels: int) -> nn.Module:
    vit_cfg = cfg.get("model", {}).get("vit", {})
    targets = cfg.get("data", {}).get("targets", ["tree_count", "mean_height"])
    return SegFormerRegressor(
        in_channels=num_input_channels,
        encoder_name=vit_cfg.get("encoder_name", "pvt_v2_b2"),
        embed_dim=int(vit_cfg.get("embed_dim", 256)),
        pretrained=bool(vit_cfg.get("pretrained", True)),
        num_targets=len(targets),
    )

"""
Clay v1.5 foundation model adapter for biomass regression.

Architecture overview
---------------------
Input  x: [B, 49, 128, 128]  (48 S1+S2 channels + 1 species channel)

1. Split x into:
   - clay_in : [B, 48, 128, 128]  S1 (0:8) + S2 (8:48)
   - species  : [B,  1, 128, 128]

2. Pre-process clay_in (per channel group):
   S1 (ch 0–7):  inverse-dataset-norm → linear → dB → Clay-norm
   S2 (ch 8–47): inverse-dataset-norm → reorder per season → Clay-norm
   (Reorder: S2 stored as [B2,B3,B4,B8,B5,B6,B7,B8A,B11,B12] per season;
    Clay expects [B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12] → permute [0,1,2,4,5,6,3,7,8,9])

3. Clay ViT-Large encoder (frozen Phase 1, unfrozen Phase 2):
   Returns patch tokens [B, 256+1, 1024]; skip CLS → reshape [B, 1024, 16, 16]

4. Species side-branch:
   Conv(1→16) → GELU → Pool(2) →
   Conv(16→32) → GELU → Pool(2) →
   Conv(32→32) → GELU → Pool(2)  →  [B, 32, 16, 16]

5. Fuse + regression head:
   Cat → [B, 1056, 16, 16] → Conv1x1 → [B, 1024, 16, 16]
   ConvTranspose2d × 3 (stride 2): 16→32→64→128
   Conv1x1 → [B, 2, 128, 128]

Config key: model.name = clay_adapter
Model-specific params live under model.clay.*
"""

from __future__ import annotations

import json
import os
import math
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import yaml

from src.models.factory import register_model
from src.models.vendor.clay_encoder import ClayEncoder

# ── Band permutation: dataset S2 order → Clay S2 order ────────────────────────
# Dataset stores per season: B2 B3 B4 B8 B5 B6 B7 B8A B11 B12 (indices 0–9)
# Clay expects per season:   B2 B3 B4 B5 B6 B7 B8  B8A B11 B12
# Permutation (applies within each season's 10-band block):
_S2_TO_CLAY = [0, 1, 2, 4, 5, 6, 3, 7, 8, 9]

# ── Clay ViT-Large hyperparameters ─────────────────────────────────────────────
_CLAY_LARGE_CFG: Dict = dict(
    mask_ratio=0.0,   # 0 = no masking at inference / fine-tune
    patch_size=8,
    shuffle=False,
    dim=1024,
    depth=24,
    heads=16,
    dim_head=64,
    mlp_ratio=4.0,
)

# ── S2 wavelengths in µm (Clay band order) ─────────────────────────────────────
_S2_WAVES_UM = [0.493, 0.560, 0.665, 0.704, 0.740, 0.783, 0.842, 0.865, 1.610, 2.190]
# ── S1 wavelengths in µm (vv, vh) ─────────────────────────────────────────────
_S1_WAVES_UM = [3.5, 4.0]


# ─────────────────────────────────────────────────────────────────────────────


def _build_norm_buffers(
    norm_stats_path: str,
    clay_metadata_path: str,
    n_s1_seasons: int = 4,
    n_s2_seasons: int = 4,
) -> Dict[str, torch.Tensor]:
    """
    Load dataset and Clay normalization parameters into tensors.

    Returns a dict of tensors (to be registered as nn.Buffers):
      ds_mean, ds_std   : [C, 1, 1]  dataset z-score stats for all 49 channels
      clay_mean, clay_std : [48, 1, 1]  Clay stats for S1+S2 channels
      waves             : [48]         wavelengths in µm for Clay encoder
    """
    # ── Dataset norm stats ─────────────────────────────────────────────────────
    with open(norm_stats_path) as f:
        ns = json.load(f)
    ds_mean = torch.tensor(ns["mean"], dtype=torch.float32).unsqueeze(-1).unsqueeze(-1)
    ds_std  = torch.tensor(ns["std"],  dtype=torch.float32).unsqueeze(-1).unsqueeze(-1)

    # ── Clay metadata ──────────────────────────────────────────────────────────
    with open(clay_metadata_path) as f:
        meta = yaml.safe_load(f)

    s2_meta = meta["sentinel-2-l2a"]
    s2_band_order = s2_meta["band_order"]
    s2_clay_mean = [s2_meta["bands"]["mean"][b] for b in s2_band_order]
    s2_clay_std  = [s2_meta["bands"]["std"][b]  for b in s2_band_order]
    s2_waves_um  = [s2_meta["bands"]["wavelength"][b] for b in s2_band_order]

    s1_meta = meta["sentinel-1-rtc"]
    s1_band_order = s1_meta["band_order"]  # [vv, vh]
    s1_clay_mean = [s1_meta["bands"]["mean"][b] for b in s1_band_order]
    s1_clay_std  = [s1_meta["bands"]["std"][b]  for b in s1_band_order]
    s1_waves_um  = [s1_meta["bands"]["wavelength"][b] for b in s1_band_order]

    # ── Tile to full 48-channel Clay tensors ───────────────────────────────────
    # S1: n_s1_seasons × 2 bands
    clay_s1_mean = torch.tensor(s1_clay_mean * n_s1_seasons, dtype=torch.float32)
    clay_s1_std  = torch.tensor(s1_clay_std  * n_s1_seasons, dtype=torch.float32)

    # S2: n_s2_seasons × 10 bands (Clay order; reordering happens in forward())
    clay_s2_mean = torch.tensor(s2_clay_mean * n_s2_seasons, dtype=torch.float32)
    clay_s2_std  = torch.tensor(s2_clay_std  * n_s2_seasons, dtype=torch.float32)

    clay_mean = torch.cat([clay_s1_mean, clay_s2_mean]).unsqueeze(-1).unsqueeze(-1)
    clay_std  = torch.cat([clay_s1_std,  clay_s2_std ]).unsqueeze(-1).unsqueeze(-1)

    # ── Wavelength tensor passed to Clay encoder ───────────────────────────────
    waves = torch.tensor(
        s1_waves_um * n_s1_seasons + s2_waves_um * n_s2_seasons,
        dtype=torch.float32,
    )

    return dict(
        ds_mean=ds_mean,
        ds_std=ds_std,
        clay_mean=clay_mean,
        clay_std=clay_std,
        waves=waves,
    )


def _build_s2_permutation(n_seasons: int = 4) -> list[int]:
    """
    Full channel index permutation for all S2 seasons.

    The dataset stacks all S2 seasons in a flat tensor of 40 channels.
    Channels 8–47 in the full 49-channel input.  Within each season's
    10-band block we need to apply _S2_TO_CLAY.

    Returns a list of 40 relative indices (into the S2-only sub-tensor)
    that, when used as x[:, indices], gives Clay-ordered S2 channels.
    """
    perm = []
    for s in range(n_seasons):
        perm += [s * 10 + i for i in _S2_TO_CLAY]
    return perm


class _SpeciesBranch(nn.Module):
    """Downsample the species map from 128×128 to 16×16 and embed it."""

    def __init__(self, out_channels: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(2, 2),          # 128 → 64
            nn.Conv2d(16, out_channels, 3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(2, 2),          # 64 → 32
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(2, 2),          # 32 → 16
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _RegressionHead(nn.Module):
    """
    Upsample [B, D, 16, 16] → [B, 2, 128, 128] via 3× transposed convolutions.
    """

    def __init__(self, in_dim: int = 1024) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_dim, 256, 3, stride=2, padding=1, output_padding=1),  # 16→32
            nn.GELU(),
            nn.ConvTranspose2d(256, 64, 3, stride=2, padding=1, output_padding=1),       # 32→64
            nn.GELU(),
            nn.ConvTranspose2d(64, 16, 3, stride=2, padding=1, output_padding=1),        # 64→128
            nn.GELU(),
            nn.Conv2d(16, 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────


class ClayAdapter(nn.Module):
    """
    Wraps the Clay ViT-Large encoder and attaches a lightweight regression
    head to produce [B, 2, H, W] biomass predictions.

    Parameters
    ----------
    encoder         : pre-built ClayEncoder (weights loaded by build fn)
    norm_buffers    : dict from _build_norm_buffers()
    s2_perm         : list[int] – full 40-channel S2 permutation
    species_feat_dim: channels for the species side-branch (default 32)
    freeze_encoder  : if True, encoder weights are frozen
    """

    def __init__(
        self,
        encoder: ClayEncoder,
        norm_buffers: Dict[str, torch.Tensor],
        s2_perm: list,
        species_feat_dim: int = 32,
        freeze_encoder: bool = True,
    ) -> None:
        super().__init__()

        # ── Clay encoder ───────────────────────────────────────────────────────
        self.encoder = encoder
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad_(False)
            self.encoder.eval()

        # ── Norm buffers ───────────────────────────────────────────────────────
        self.register_buffer("ds_mean",   norm_buffers["ds_mean"])    # [49, 1, 1]
        self.register_buffer("ds_std",    norm_buffers["ds_std"])     # [49, 1, 1]
        self.register_buffer("clay_mean", norm_buffers["clay_mean"])  # [48, 1, 1]
        self.register_buffer("clay_std",  norm_buffers["clay_std"])   # [48, 1, 1]
        self.register_buffer("waves",     norm_buffers["waves"])      # [48]

        # S2 band permutation (relative indices within S2 sub-tensor)
        self.register_buffer(
            "s2_perm",
            torch.tensor(s2_perm, dtype=torch.long),
        )

        # ── Species side-branch ────────────────────────────────────────────────
        self.species_branch = _SpeciesBranch(out_channels=species_feat_dim)

        # ── Fusion 1×1 conv ────────────────────────────────────────────────────
        fuse_in = self.encoder.dim + species_feat_dim
        self.fuse_conv = nn.Conv2d(fuse_in, self.encoder.dim, 1)

        # ── Regression head ────────────────────────────────────────────────────
        self.head = _RegressionHead(in_dim=self.encoder.dim)

    # ── Normalization helpers ──────────────────────────────────────────────────

    def _clay_preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert the dataset-normalized 48-channel tensor (S1+S2) to the
        normalization expected by the Clay encoder.

        Steps:
          1. Un-normalize with dataset mean/std (first 48 channels).
          2. S1 (ch 0:8): convert linear gamma-0 → dB  (10 × log10).
          3. Reorder S2 (ch 8:48) from dataset band order → Clay band order.
          4. Re-normalize with Clay mean/std.
        """
        # Step 1 – restore raw values
        raw = x * self.ds_std[:48] + self.ds_mean[:48]   # [B, 48, H, W]

        # Step 2 – S1 linear → dB
        s1 = raw[:, :8, :, :]                             # [B, 8, H, W]
        s1_db = 10.0 * torch.log10(s1.clamp(min=1e-6))

        # Step 3 – reorder S2 bands within each season
        s2 = raw[:, 8:, :, :]                             # [B, 40, H, W]
        s2 = s2[:, self.s2_perm, :, :]                    # Clay band order

        # Rebuild 48-channel tensor [S1_dB | S2_reordered]
        clay_raw = torch.cat([s1_db, s2], dim=1)          # [B, 48, H, W]

        # Step 4 – Clay normalization
        clay_norm = (clay_raw - self.clay_mean) / self.clay_std.clamp(min=1e-8)

        return clay_norm

    # ── Forward ────────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : [B, 49, H, W] (or [B, 48, H, W] if use_species=False)

        Returns
        -------
        [B, 2, H, W]
        """
        has_species = x.shape[1] == 49

        clay_in = x[:, :48, :, :]              # [B, 48, H, W]
        species = x[:, 48:49, :, :] if has_species else None

        B, C, H, W = clay_in.shape

        # ── Prepare Clay input ─────────────────────────────────────────────────
        clay_norm = self._clay_preprocess(clay_in)         # [B, 48, H, W]

        grid_size = H // self.encoder.patch_size           # e.g. 128/8 = 16
        device = clay_in.device

        datacube = {
            "pixels": clay_norm,
            "time":   torch.zeros(B, 4, device=device),   # placeholder
            "latlon": torch.zeros(B, 4, device=device),   # placeholder
            "gsd":    torch.tensor(10.0, device=device),
            "waves":  self.waves,                          # [48]
        }

        # ── Encoder forward (mask_ratio=0 → all tokens returned) ──────────────
        if self.encoder.training or not torch.is_grad_enabled():
            encoded, _, _, _ = self.encoder(datacube)      # [B, N+1, D]
        else:
            with torch.no_grad():
                encoded, _, _, _ = self.encoder(datacube)

        # Drop CLS token, reshape to spatial feature map
        tokens = encoded[:, 1:, :]                         # [B, N, D]
        feat = tokens.permute(0, 2, 1).reshape(
            B, self.encoder.dim, grid_size, grid_size
        )                                                   # [B, 1024, 16, 16]

        # ── Species side-branch ────────────────────────────────────────────────
        if species is not None:
            sp_feat = self.species_branch(species)         # [B, 32, 16, 16]
            feat = torch.cat([feat, sp_feat], dim=1)       # [B, 1056, 16, 16]
            feat = self.fuse_conv(feat)                    # [B, 1024, 16, 16]

        # ── Regression head ────────────────────────────────────────────────────
        out = self.head(feat)                               # [B, 2, 128, 128]
        return out

    def unfreeze_encoder(self) -> None:
        """Call to transition from Phase 1 (head only) to Phase 2 (full finetune)."""
        for p in self.encoder.parameters():
            p.requires_grad_(True)
        self.encoder.train()


# ─────────────────────────────────────────────────────────────────────────────


def _load_clay_encoder(ckpt_path: str | None) -> ClayEncoder:
    """
    Build a Clay ViT-Large encoder and optionally load pretrained weights
    from a Lightning checkpoint.

    If ckpt_path is None or the file does not exist, the encoder is randomly
    initialised (useful for smoke tests and architecture validation).
    """
    encoder = ClayEncoder(**_CLAY_LARGE_CFG)

    if ckpt_path is None or not Path(ckpt_path).exists():
        if os.environ.get("RANK", "0") == "0":
            print(
                f"[ClayAdapter] WARNING: checkpoint not found at '{ckpt_path}'. "
                "Encoder randomly initialised — do NOT use for real training."
            )
        return encoder

    if os.environ.get("RANK", "0") == "0":
        print(f"[ClayAdapter] Loading Clay encoder weights from {ckpt_path} …")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt.get("state_dict", ckpt)

    # Lightning checkpoint: keys are "model.encoder.*"
    prefix = "model.encoder."
    encoder_sd = {
        k[len(prefix):]: v
        for k, v in sd.items()
        if k.startswith(prefix)
    }

    if not encoder_sd:
        raise KeyError(
            f"No 'model.encoder.*' keys found in checkpoint {ckpt_path}. "
            f"Available prefixes: {sorted({k.split('.')[0] for k in sd})}"
        )

    missing, unexpected = encoder.load_state_dict(encoder_sd, strict=False)
    if os.environ.get("RANK", "0") == "0":
        if missing:
            print(
                f"[ClayAdapter]   Missing keys  : {missing[:5]}"
                f"{'...' if len(missing) > 5 else ''}"
            )
        if unexpected:
            print(
                f"[ClayAdapter]   Unexpected keys: {unexpected[:5]}"
                f"{'...' if len(unexpected) > 5 else ''}"
            )
        print("[ClayAdapter] Encoder weights loaded successfully.")
    return encoder


@register_model("clay_adapter")
def build_clay_adapter(cfg: dict, num_input_channels: int) -> nn.Module:
    """
    Registry entry point.  Called by src.models.factory.build_model().

    Expected config keys under model.clay:
      ckpt_path       : path to clay-v1.5.ckpt  (None → random init)
      metadata_path   : path to clay_metadata.yaml
      freeze_encoder  : bool (default True)
      species_feat_dim: int  (default 32)

    Expected config key under data:
      norm_stats_path : path to norm_stats.json
    """
    clay_cfg = cfg.get("model", {}).get("clay", {})

    ckpt_path     = clay_cfg.get("ckpt_path", None)
    metadata_path = clay_cfg.get("metadata_path")
    freeze_enc    = clay_cfg.get("freeze_encoder", True)
    sp_dim        = clay_cfg.get("species_feat_dim", 32)

    norm_stats_path = cfg.get("data", {}).get("norm_stats_path")

    if metadata_path is None or not Path(metadata_path).exists():
        raise FileNotFoundError(
            f"Clay metadata YAML not found at '{metadata_path}'. "
            "Set model.clay.metadata_path in the config and ensure the file exists."
        )
    if norm_stats_path is None or not Path(norm_stats_path).exists():
        raise FileNotFoundError(
            f"norm_stats.json not found at '{norm_stats_path}'. "
            "Run scripts/compute_stats.py first."
        )

    norm_buffers = _build_norm_buffers(norm_stats_path, metadata_path)
    s2_perm      = _build_s2_permutation(n_seasons=4)
    encoder      = _load_clay_encoder(ckpt_path)

    return ClayAdapter(
        encoder=encoder,
        norm_buffers=norm_buffers,
        s2_perm=s2_perm,
        species_feat_dim=sp_dim,
        freeze_encoder=freeze_enc,
    )

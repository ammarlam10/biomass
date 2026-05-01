"""
Vendored Clay v1.5 encoder — Apache-2.0 License.
Source: https://github.com/Clay-foundation/model

Combines claymodel/utils.py, claymodel/backbone.py, claymodel/factory.py,
and the Encoder class from claymodel/model.py into a single self-contained
module.  Only the Encoder is retained (Decoder, ClayMAE, teacher etc. are
omitted since we only need the pretrained encoder for fine-tuning).

External dependency added: einops  (lightweight, no conflicts).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn


# ── Positional encodings (from claymodel/utils.py) ────────────────────────────


def posemb_sincos_2d_with_gsd(
    h: int,
    w: int,
    dim: int,
    gsd: torch.Tensor,
    temperature: int = 10000,
    dtype=torch.float32,
) -> torch.Tensor:
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"

    gsd = gsd.to(x.device)
    omega = torch.arange(dim // 4, device=x.device) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** (2 * omega / dim)) * (gsd / 1.0)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)


def posemb_sincos_1d(
    waves: torch.Tensor | int,
    dim: int,
    temperature: int = 10000,
    dtype=torch.float32,
) -> torch.Tensor:
    assert dim % 2 == 0, "Feature dimension must be a multiple of 2 for sincos embedding"
    waves = torch.arange(waves) if isinstance(waves, int) else waves

    omega = torch.arange(dim // 2, device=waves.device) / (dim // 2 - 1)
    omega = 1.0 / (temperature**omega)

    scaled_waves = waves[:, None] * omega[None, :]
    pe = torch.cat((scaled_waves.sin(), scaled_waves.cos()), dim=1)
    return pe.type(dtype)


# ── Transformer backbone (from claymodel/backbone.py) ─────────────────────────


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64, fused_attn: bool = True) -> None:
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5
        self.norm = nn.LayerNorm(dim)
        self.fused_attn = fused_attn

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        else:
            attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
            attn = attn.softmax(dim=-1)
            x = torch.matmul(attn, v)

        x = rearrange(x, "b h n d -> b n (h d)")
        return self.to_out(x)


class Transformer(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        fused_attn: bool,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        Attention(dim, heads=heads, dim_head=dim_head, fused_attn=fused_attn),
                        FeedForward(dim, mlp_dim),
                    ]
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


# ── Dynamic patch embedding (from claymodel/factory.py) ───────────────────────


class FCBlock(nn.Module):
    def __init__(self, size: int) -> None:
        super().__init__()
        self.l1 = nn.Linear(size, size)
        self.l2 = nn.Linear(size, size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.gelu(self.l1(x))
        y = F.gelu(self.l2(y))
        return x + y


class WavesTransformer(nn.Module):
    def __init__(
        self,
        wave_dim: int,
        output_dim: int,
        num_latent_tokens: int,
        embed_dim: int,
        is_decoder: bool,
        num_heads: int = 4,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.num_latent_tokens = num_latent_tokens
        self.is_decoder = is_decoder
        layer = nn.TransformerEncoderLayer(
            d_model=wave_dim,
            nhead=num_heads,
            activation="gelu",
            dropout=0,
            norm_first=False,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers)

        self.fc_weight = nn.Linear(wave_dim, output_dim)
        self.fc_bias = None if self.is_decoder else nn.Linear(wave_dim, embed_dim)

        self.weight_tokens = nn.Parameter(torch.randn(self.num_latent_tokens, wave_dim) * 0.02)
        self.bias_token = nn.Parameter(torch.randn(1, wave_dim) * 0.02)

    def forward(self, x: torch.Tensor):
        x = torch.cat([self.weight_tokens, x, self.bias_token], dim=0)
        out = self.encoder(x)
        weights = self.fc_weight(
            out[self.num_latent_tokens : -1] + x[self.num_latent_tokens : -1]
        )
        bias = None if self.is_decoder else self.fc_bias(out[-1])
        return weights, bias


class DynamicEmbedding(nn.Module):
    def __init__(
        self,
        wave_dim: int,
        num_latent_tokens: int,
        patch_size: int,
        embed_dim: int,
        is_decoder: bool = False,
    ) -> None:
        super().__init__()
        self.wave_dim = wave_dim
        self.num_latent_tokens = num_latent_tokens
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.is_decoder = is_decoder
        self.output_dim = (patch_size**2) * embed_dim

        self.weight_generator = WavesTransformer(
            wave_dim,
            self.output_dim,
            self.num_latent_tokens,
            self.embed_dim,
            is_decoder,
        )
        self.fclayer = FCBlock(self.wave_dim)
        self._initialize_weights()

    def forward(self, batch: torch.Tensor, waves: torch.Tensor):
        waves_enc = posemb_sincos_1d(waves, self.wave_dim)
        waves_enc = waves_enc.to(batch.device)
        waves_enc = self.fclayer(waves_enc)
        weight, bias = self.weight_generator(waves_enc)

        dynamic_weight = rearrange(
            weight,
            "cin (cout k1 k2) -> cout cin k1 k2",
            k1=self.patch_size,
            k2=self.patch_size,
        )
        if bias is not None:
            bias = rearrange(bias, "b -> (b)")
        dynamic_out = F.conv2d(
            batch, dynamic_weight * 0.02, bias=bias, stride=self.patch_size
        )
        x = rearrange(dynamic_out, "b c h w -> b (h w) c")
        return x, waves_enc

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


# ── Clay Encoder (from claymodel/model.py — Encoder class only) ───────────────


class ClayEncoder(nn.Module):
    """
    Clay ViT encoder with dynamic wavelength-conditioned patch embedding.

    For fine-tuning we always use mask_ratio=0.0 and shuffle=False so that
    the full spatial grid of patch tokens is returned in a deterministic order.
    """

    def __init__(
        self,
        mask_ratio: float,
        patch_size: int,
        shuffle: bool,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_ratio: float,
    ) -> None:
        super().__init__()
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        self.shuffle = shuffle
        self.dim = dim
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)

        self.patch_embedding = DynamicEmbedding(
            wave_dim=128,
            num_latent_tokens=128,
            patch_size=patch_size,
            embed_dim=dim,
            is_decoder=False,
        )

        self.transformer = Transformer(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=int(dim * mlp_ratio),
            fused_attn=True,
        )

    def _to_patch_embed(self, cube: torch.Tensor, waves: torch.Tensor):
        patches, waves_encoded = self.patch_embedding(cube, waves)
        return patches, waves_encoded

    def _add_encodings(
        self,
        patches: torch.Tensor,
        time: torch.Tensor,
        latlon: torch.Tensor,
        gsd: torch.Tensor,
    ) -> torch.Tensor:
        B, L, D = patches.shape
        grid_size = int(math.sqrt(L))
        self.num_patches = grid_size**2

        pos_encoding = (
            posemb_sincos_2d_with_gsd(
                h=grid_size,
                w=grid_size,
                dim=(self.dim - 8),
                gsd=gsd,
            )
            .to(patches.device)
            .detach()
        )

        time_latlon = torch.hstack((time, latlon)).to(patches.device).detach()

        pos_encoding = repeat(pos_encoding, "L D -> B L D", B=B)
        time_latlon = repeat(time_latlon, "B D -> B L D", L=L)
        pos_metadata_encoding = torch.cat((pos_encoding, time_latlon), dim=-1)

        return patches + pos_metadata_encoding

    def _mask_out(self, patches: torch.Tensor):
        B, L, D = patches.shape

        if self.shuffle:
            noise = torch.randn((B, L), device=patches.device)
        else:
            noise = rearrange(
                torch.arange(B * L, device=patches.device), "(B L) -> B L", B=B, L=L
            )

        random_indices = torch.argsort(noise, dim=-1)
        reverse_indices = torch.argsort(random_indices, dim=-1)

        num_masked_patches = int(self.mask_ratio * self.num_patches)
        masked_indices, unmasked_indices = (
            random_indices[:, :num_masked_patches],
            random_indices[:, num_masked_patches:],
        )

        masked_matrix = torch.zeros((B, L), device=patches.device)
        masked_matrix[:, :num_masked_patches] = 1
        masked_matrix = torch.gather(masked_matrix, dim=1, index=reverse_indices)

        batch_indices = rearrange(torch.arange(B, device=patches.device), "B -> B 1")
        unmasked_patches = patches[batch_indices, unmasked_indices, :]
        _ = patches[batch_indices, masked_indices, :]

        return unmasked_patches, unmasked_indices, masked_indices, masked_matrix

    def forward(self, datacube: dict) -> tuple:
        cube = datacube["pixels"]    # [B C H W]
        time = datacube["time"]      # [B 4]
        latlon = datacube["latlon"]  # [B 4]
        gsd = datacube["gsd"]        # scalar tensor
        waves = datacube["waves"]    # [C]

        B, C, H, W = cube.shape

        patches, _ = self._to_patch_embed(cube, waves)
        patches = self._add_encodings(patches, time, latlon, gsd)

        (
            unmasked_patches,
            unmasked_indices,
            masked_indices,
            masked_matrix,
        ) = self._mask_out(patches)

        cls_tokens = repeat(self.cls_token, "1 1 D -> B 1 D", B=B)
        unmasked_patches = torch.cat((cls_tokens, unmasked_patches), dim=1)
        encoded = self.transformer(unmasked_patches)

        return encoded, unmasked_indices, masked_indices, masked_matrix

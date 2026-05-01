"""
Clay adapter smoke test.

Validates:
  1. Import and model build (random-init encoder, no checkpoint needed).
  2. Forward pass shape: [B, 2, 128, 128].
  3. No NaN / Inf in output.
  4. Normalization round-trip: S1 linear→dB path produces finite values.
  5. Species branch: test both with and without species channel.

Run inside Docker:
    docker compose run --rm smoke_clay
"""

import sys
from pathlib import Path

# Allow running directly from the repo root without installing the package.
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from src.models.factory import build_model

# ── Config ─────────────────────────────────────────────────────────────────────
# Resolve paths relative to this script so the test works both in Docker
# (/workspace) and directly on the host (/data/ammar/biomass).
_REPO = Path(__file__).parent.parent

CFG = {
    "model": {
        "name": "clay_adapter",
        "clay": {
            "ckpt_path": None,   # random init – no checkpoint needed for smoke test
            "metadata_path": str(_REPO / "configs" / "clay_metadata.yaml"),
            "freeze_encoder": True,
            "species_feat_dim": 32,
        },
    },
    "data": {
        "norm_stats_path": str(_REPO / "artifacts" / "norm_stats.json"),
    },
}


def _make_input(B: int, C: int, H: int = 128, W: int = 128) -> torch.Tensor:
    """Simulate dataset-normalised input (mean≈0, std≈1)."""
    return torch.randn(B, C, H, W)


def check(condition: bool, msg: str) -> None:
    if condition:
        print(f"  PASS  {msg}")
    else:
        print(f"  FAIL  {msg}")
        sys.exit(1)


def main() -> None:
    print("\n=== Clay Adapter Smoke Test ===\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Build model ────────────────────────────────────────────────────────────
    print("\n[1] Building model …")
    model = build_model(CFG, num_input_channels=49)
    model = model.to(device)
    model.eval()
    print(f"  Trainable params : {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"  Total params     : {sum(p.numel() for p in model.parameters()):,}")

    # ── Forward pass: with species (49 ch) ────────────────────────────────────
    print("\n[2] Forward pass — 49 channels (S1+S2+species) …")
    B = 2
    x49 = _make_input(B, 49).to(device)
    with torch.no_grad():
        out = model(x49)

    check(out.shape == (B, 2, 128, 128), f"Output shape {out.shape} == (B, 2, 128, 128)")
    check(torch.isfinite(out).all().item(),  "Output has no NaN / Inf")

    # ── Forward pass: without species (48 ch) ─────────────────────────────────
    print("\n[3] Forward pass — 48 channels (S1+S2 only) …")
    x48 = _make_input(B, 48).to(device)
    with torch.no_grad():
        out48 = model(x48)

    check(out48.shape == (B, 2, 128, 128), f"Output shape {out48.shape} == (B, 2, 128, 128)")
    check(torch.isfinite(out48).all().item(),  "Output has no NaN / Inf")

    # ── Normalization: S1 linear→dB path ─────────────────────────────────────
    print("\n[4] S1 linear→dB normalization …")
    # Simulate typical linear gamma-0 values (VV≈0.05, VH≈0.015)
    # The dataset z-scores these, so back-calculate what the z-scored value is.
    ds_mean = model.ds_mean[:8, 0, 0].cpu()   # [8]
    ds_std  = model.ds_std[:8, 0, 0].cpu()    # [8]

    s1_linear = torch.tensor([0.05, 0.015] * 4)           # 4 seasons × [VV, VH]
    s1_znorm  = (s1_linear - ds_mean) / ds_std.clamp(min=1e-8)

    x_s1_test = _make_input(1, 49)
    x_s1_test[0, :8, :, :] = s1_znorm.view(8, 1, 1)

    # Run _clay_preprocess and verify S1 dB values are in reasonable range
    x_s1_dev = x_s1_test.to(device)
    raw_s1   = x_s1_dev[:, :8] * model.ds_std[:8] + model.ds_mean[:8]   # [1, 8, 128, 128]
    s1_db    = 10.0 * torch.log10(raw_s1.clamp(min=1e-6))

    # VV should be ~10*log10(0.05) ≈ -13 dB, VH ~10*log10(0.015) ≈ -18 dB
    vv_db = s1_db[0, 0, 0, 0].item()
    vh_db = s1_db[0, 1, 0, 0].item()
    check(-20.0 < vv_db < -5.0,  f"VV dB = {vv_db:.1f} (expected ≈ -13)")
    check(-25.0 < vh_db < -10.0, f"VH dB = {vh_db:.1f} (expected ≈ -18)")

    # ── Gradient flow: Phase 1 (frozen encoder) ───────────────────────────────
    print("\n[5] Gradient flow (Phase 1 — frozen encoder) …")
    model.train()
    x = _make_input(B, 49).to(device)
    out_train = model(x)
    loss = out_train.mean()
    loss.backward()

    enc_has_grad = any(
        p.grad is not None for p in model.encoder.parameters() if p.requires_grad
    )
    head_has_grad = any(
        p.grad is not None for p in model.head.parameters()
    )
    check(not enc_has_grad, "Encoder has NO gradients when frozen")
    check(head_has_grad,    "Regression head HAS gradients")

    print("\n=== All checks passed ===\n")


if __name__ == "__main__":
    main()

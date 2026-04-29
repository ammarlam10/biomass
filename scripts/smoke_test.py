"""
Smoke test: verifies the full pipeline (dataset → model → loss → backward)
with a tiny synthetic batch, without needing the real dataset on disk.

Run inside the container:
    python scripts/smoke_test.py

Exit 0 = everything wired up correctly.
Exit 1 = something is broken (see traceback).
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.models  # noqa: F401 – triggers @register_model decorators
from src.losses.masked_regression import build_loss
from src.models.factory import build_model, list_models
from src.training.metrics import RunningMetrics, get_inverse_transforms


def run_smoke_test() -> None:
    print("=" * 55)
    print("Biomass pipeline smoke test")
    print("=" * 55)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    print(f"Registered models: {list_models()}")

    # ── synthetic batch ───────────────────────────────────────────────────────
    B, C, H, W = 2, 49, 128, 128
    x = torch.randn(B, C, H, W, device=device)
    y = torch.rand(B, 2, H, W, device=device)

    # sparse valid mask: ~30% of pixels are "tree pixels"
    mask = torch.rand(B, H, W, device=device) < 0.30

    print(f"\nBatch shape : x={tuple(x.shape)}  y={tuple(y.shape)}  mask={tuple(mask.shape)}")
    print(f"Valid pixels: {mask.float().mean().item()*100:.1f}%")

    # ── build model ───────────────────────────────────────────────────────────
    cfg = {
        "model": {
            "name": "unet_resnet50",
            "unet": {
                "encoder_name": "resnet50",
                "encoder_weights": None,   # no imagenet download needed for smoke test
                "decoder_channels": [256, 128, 64, 32, 16],
            },
        },
        "training": {"loss": "masked_mse", "loss_weights": {"tree_count": 1.0, "mean_height": 1.0}},
        "data": {"target_transform": {"tree_count": "log1p", "mean_height": "none"}},
    }

    print("\n[1/4] Building model ... ", end="", flush=True)
    model = build_model(cfg, num_input_channels=C).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"OK  ({n_params:,} params)")

    # ── forward pass ──────────────────────────────────────────────────────────
    print("[2/4] Forward pass ... ", end="", flush=True)
    pred = model(x)
    assert pred.shape == (B, 2, H, W), f"Unexpected output shape: {pred.shape}"
    print(f"OK  output={tuple(pred.shape)}")

    # ── loss ──────────────────────────────────────────────────────────────────
    print("[3/4] Loss + backward ... ", end="", flush=True)
    criterion = build_loss(cfg)
    loss = criterion(pred, y, mask)
    loss.backward()
    assert loss.item() >= 0, "Loss is negative"
    print(f"OK  loss={loss.item():.6f}")

    # ── metrics (via RunningMetrics streaming accumulator) ────────────────────
    print("[4/4] Metrics (RunningMetrics) ... ", end="", flush=True)
    from src.training.metrics import RunningMetrics
    inv_transforms = get_inverse_transforms(cfg)
    rm = RunningMetrics(inv_transforms)
    rm.update(pred.detach().cpu(), y.cpu(), mask.cpu())
    metrics = rm.compute()
    for k, v in metrics.items():
        print(f"\n        {k}: {v:.4f}" if v == v else f"\n        {k}: NaN", end="")
    print("\n")

    # ── zero-mask edge case (fresh forward pass) ─────────────────────────────
    print("Edge case (all-zero mask) ... ", end="", flush=True)
    with torch.no_grad():
        pred_zero = model(x)
    zero_mask = torch.zeros(B, H, W, dtype=torch.bool, device=device)
    loss_zero = criterion(pred_zero, y, zero_mask)
    assert loss_zero.item() == 0.0
    print("OK")

    print("\n" + "=" * 55)
    print("All checks passed.")
    print("=" * 55)


if __name__ == "__main__":
    try:
        run_smoke_test()
    except Exception:
        traceback.print_exc()
        sys.exit(1)

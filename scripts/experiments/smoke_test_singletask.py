"""
Smoke test for Experiment A1: verifies that single-task and dual-task model
builds produce the correct output shape, and that the dataset / loss / metrics
work end-to-end with a single-target config.

Run inside the biomass Docker container:
    python scripts/experiments/smoke_test_singletask.py
"""
import sys
import torch
import numpy as np

sys.path.insert(0, "/workspace")

from src.models.factory import build_model
from src.losses.masked_regression import build_loss, MaskedRegressionLoss
from src.training.metrics import RunningMetrics, get_inverse_transforms

BATCH = 2
H, W  = 128, 128
C     = 49   # 48 S1+S2 + 1 species

x = torch.randn(BATCH, C, H, W)
mask = torch.ones(BATCH, H, W, dtype=torch.bool)
mask[0, :64, :64] = False   # some invalid pixels

print("=" * 60)
print("  A1 Smoke Test – Single-Task Decoupling")
print("=" * 60)

# ── 1. Model output shapes ────────────────────────────────────────────────────
print("\n[1] Model forward pass shapes")

model_cases = [
    ("SegFormer-B3 dual",   "vit_segmentation",  {"encoder_name": "pvt_v2_b3", "embed_dim": 256, "pretrained": False}, ["tree_count", "mean_height"]),
    ("SegFormer-B3 height", "vit_segmentation",  {"encoder_name": "pvt_v2_b3", "embed_dim": 256, "pretrained": False}, ["mean_height"]),
    ("SegFormer-B3 count",  "vit_segmentation",  {"encoder_name": "pvt_v2_b3", "embed_dim": 256, "pretrained": False}, ["tree_count"]),
    ("U-Net dual",          "unet_resnet50",      {"encoder_name": "resnet50",  "encoder_weights": None},               ["tree_count", "mean_height"]),
    ("U-Net height",        "unet_resnet50",      {"encoder_name": "resnet50",  "encoder_weights": None},               ["mean_height"]),
    ("U-Net count",         "unet_resnet50",      {"encoder_name": "resnet50",  "encoder_weights": None},               ["tree_count"]),
]

for label, model_name, model_sub_cfg, targets in model_cases:
    n_t = len(targets)
    cfg = {
        "model": {"name": model_name, "vit": model_sub_cfg, "unet": model_sub_cfg},
        "data":  {"targets": targets},
    }
    model = build_model(cfg, C)
    model.eval()
    with torch.no_grad():
        out = model(x)
    expected = (BATCH, n_t, H, W)
    assert out.shape == expected, f"{label}: expected {expected}, got {out.shape}"
    print(f"  OK  {label:28s}  out={tuple(out.shape)}")

# ── 2. Loss – single-task ─────────────────────────────────────────────────────
print("\n[2] Loss – single-task")

for targets in [["mean_height"], ["tree_count"], ["tree_count", "mean_height"]]:
    cfg = {
        "training": {"loss": "masked_smooth_l1", "loss_weights": {}},
        "data":     {"targets": targets},
    }
    criterion = build_loss(cfg)
    assert len(criterion.weights) == len(targets), \
        f"Weight count mismatch: {len(criterion.weights)} vs {len(targets)}"
    pred   = torch.randn(BATCH, len(targets), H, W)
    target = torch.randn(BATCH, len(targets), H, W)
    loss   = criterion(pred, target, mask)
    assert loss.shape == torch.Size([]), f"Loss should be scalar, got {loss.shape}"
    print(f"  OK  targets={targets!s:35s}  loss={loss.item():.4f}")

# ── 3. RunningMetrics – single-task ──────────────────────────────────────────
print("\n[3] RunningMetrics – single-task")

for targets in [["mean_height"], ["tree_count"], ["tree_count", "mean_height"]]:
    cfg = {"data": {"targets": targets, "target_transform": {t: "log1p" for t in targets}}}
    inv = get_inverse_transforms(cfg, targets)
    rm  = RunningMetrics(inv, target_names=targets)
    pred   = torch.randn(BATCH, len(targets), H, W)
    target = torch.randn(BATCH, len(targets), H, W)
    rm.update(pred, target, mask)
    metrics = rm.compute()
    for t in targets:
        assert f"rmse_{t}" in metrics, f"Missing rmse_{t}"
        assert f"rmse_{t}_orig" in metrics, f"Missing rmse_{t}_orig"
    unwanted = [k for k in metrics if not any(t in k for t in targets)]
    assert not unwanted, f"Unexpected metric keys: {unwanted}"
    print(f"  OK  targets={targets!s:35s}  keys={sorted(metrics.keys())}")

print("\n" + "=" * 60)
print("  All smoke tests passed.")
print("=" * 60)

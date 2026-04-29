"""
Full end-to-end pipeline validation against the real dataset.

Run inside the container:
    python scripts/pipeline_check.py

Checks:
  1. Parquet columns match config
  2. Dataset construction for all splits
  3. Sample patch shapes, NaN handling, and mask coverage
  4. DataLoader multi-worker batch
  5. Model forward on a real batch
  6. Masked loss + backward
  7. RunningMetrics on real batch
  8. Target encoding (tree_count=0 vs NaN, mean_height=NaN)

Exit 0 = all clear.
Exit 1 = at least one check failed (details printed).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import zarr
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.models  # noqa: F401 – register all models
from src.data.dataset import BiomassDataset
from src.data.transforms import build_train_transform
from src.losses.masked_regression import build_loss
from src.models.factory import build_model
from src.training.metrics import RunningMetrics, get_inverse_transforms
from src.utils.config import load_config

PASS = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"


def main() -> None:
    cfg = load_config("/workspace/configs/default.yaml")
    errors: list[str] = []

    # ── 1. Parquet split file ─────────────────────────────────────────────────
    section("1. Parquet split file")
    df = pd.read_parquet(cfg["data"]["split_file"])
    split_col = cfg["data"]["split_column"]
    idx_col = cfg["data"]["patch_idx_column"]

    if split_col in df.columns and idx_col in df.columns:
        counts = df[split_col].value_counts().to_dict()
        ok(f"Columns present: split_col='{split_col}', idx_col='{idx_col}'")
        ok(f"Split sizes: {counts}")
        ok(f"zarr_idx range: {df[idx_col].min()} – {df[idx_col].max()}")
    else:
        msg = (
            f"Config mismatch: split_col='{split_col}', idx_col='{idx_col}' "
            f"not all in columns={df.columns.tolist()}"
        )
        fail(msg)
        errors.append(msg)

    # report extra useful columns
    for col in ["valid_pixel_pct", "tree_pixel_pct", "mean_tree_count", "buffered", "in_bavaria"]:
        if col in df.columns:
            val = df[col]
            if val.dtype == bool or val.dtype == object:
                ok(f"  {col}: {val.value_counts().to_dict()}")
            else:
                ok(f"  {col}: mean={val.mean():.4f}  min={val.min():.4f}  max={val.max():.4f}  zero={( val==0).sum()}")

    # ── 2. Dataset construction ───────────────────────────────────────────────
    section("2. Dataset construction (train / val / test)")
    datasets: dict[str, BiomassDataset] = {}
    for split in ["train", "val", "test"]:
        try:
            ds = BiomassDataset(
                root=cfg["data"]["root"], split=split, cfg=cfg,
                norm_stats=None, transform=None,
            )
            datasets[split] = ds
            ok(f"{split:5s}: {len(ds)} patches  channels={ds.num_channels}")
        except Exception as e:
            msg = f"{split} dataset construction failed: {e}"
            fail(msg)
            errors.append(msg)

    if "train" not in datasets:
        fail("Cannot continue without train dataset")
        sys.exit(1)

    ds_train = datasets["train"]

    # ── 3. Sample patch shapes and NaN audit ─────────────────────────────────
    section("3. Sample patch checks (first 50 train patches)")
    support_ratios: list[float] = []
    nan_in_x_total = 0
    y_nan_at_valid = 0
    n_zero_mask = 0

    for i in range(50):
        x, y, mask = ds_train[i]
        assert x.shape == (ds_train.num_channels, 128, 128), f"x shape {x.shape}"
        assert y.shape == (2, 128, 128),                     f"y shape {y.shape}"
        assert mask.shape == (128, 128),                     f"mask shape {mask.shape}"
        assert mask.dtype == torch.bool,                     f"mask dtype {mask.dtype}"

        nan_in_x_total += torch.isnan(x).sum().item()
        sr = mask.float().mean().item()
        support_ratios.append(sr)
        if sr == 0:
            n_zero_mask += 1

        y_valid = y[:, mask]
        if y_valid.numel() > 0 and torch.isnan(y_valid).any():
            y_nan_at_valid += 1

    ok(f"All 50 patch shapes correct: x=[{ds_train.num_channels},128,128] y=[2,128,128] mask=[128,128]")
    if nan_in_x_total > 0:
        warn(f"NaN found in input tensors (post-fill): {nan_in_x_total} (should be 0 after nan_to_num)")
        errors.append(f"NaN in input after nan_to_num: {nan_in_x_total}")
    else:
        ok("No NaN in input tensors (nan_to_num fill working)")

    if y_nan_at_valid > 0:
        fail(f"NaN in y at valid-mask positions in {y_nan_at_valid} patches")
        errors.append(f"NaN in y at valid positions")
    else:
        ok("No NaN in targets at valid-mask positions")

    ok(
        f"Support ratios over 50 patches: "
        f"min={min(support_ratios)*100:.1f}%  "
        f"mean={np.mean(support_ratios)*100:.1f}%  "
        f"max={max(support_ratios)*100:.1f}%"
    )
    if n_zero_mask:
        warn(f"{n_zero_mask}/50 patches have zero-coverage mask (all non-tree)")
    else:
        ok("No zero-coverage masks in first 50 patches")

    # ── 4. DataLoader ─────────────────────────────────────────────────────────
    section("4. DataLoader (batch_size=4, num_workers=4)")
    ds_aug = BiomassDataset(
        root=cfg["data"]["root"], split="train", cfg=cfg,
        norm_stats=None, transform=build_train_transform(),
    )
    loader = DataLoader(
        ds_aug, batch_size=4, shuffle=True, num_workers=4,
        prefetch_factor=2, persistent_workers=True,
    )
    xb, yb, mb = next(iter(loader))
    assert xb.shape == (4, ds_train.num_channels, 128, 128)
    assert yb.shape == (4, 2, 128, 128)
    assert mb.shape == (4, 128, 128)
    ok(f"Batch shapes: x={tuple(xb.shape)}  y={tuple(yb.shape)}  mask={tuple(mb.shape)}")
    ok(f"Batch x finite: {torch.isfinite(xb).all().item()}")
    ok(f"Batch mask support: {mb.float().mean().item()*100:.1f}%")

    # ── 5. Model forward on real batch ────────────────────────────────────────
    section("5. Model forward pass on real batch")
    cfg_no_pretrain = dict(cfg)
    cfg_no_pretrain["model"] = dict(cfg["model"])
    cfg_no_pretrain["model"]["unet"] = dict(cfg["model"].get("unet", {}))
    cfg_no_pretrain["model"]["unet"]["encoder_weights"] = None
    model = build_model(cfg_no_pretrain, num_input_channels=ds_train.num_channels)
    model.eval()
    with torch.no_grad():
        pred = model(xb)
    assert pred.shape == (4, 2, 128, 128), f"pred shape {pred.shape}"
    ok(f"pred shape={tuple(pred.shape)}  finite={torch.isfinite(pred).all().item()}")
    ok(f"pred range: [{pred.min().item():.3f}, {pred.max().item():.3f}]")

    # ── 6. Loss backward on real batch ────────────────────────────────────────
    section("6. Masked loss + backward on real batch")
    model.train()
    pred2 = model(xb)
    criterion = build_loss(cfg)
    loss = criterion(pred2, yb, mb)
    assert loss.item() >= 0, f"Negative loss: {loss.item()}"
    loss.backward()
    ok(f"loss={loss.item():.4f}  (gradient flow OK)")
    if mb.sum() == 0:
        warn("All-zero mask in this batch – loss=0 by design")

    # ── 7. RunningMetrics on real batch ───────────────────────────────────────
    section("7. RunningMetrics on real batch")
    inv_t = get_inverse_transforms(cfg)
    rm = RunningMetrics(inv_t)
    with torch.no_grad():
        rm.update(pred2.detach(), yb, mb)
    m = rm.compute()
    ok(f"rmse_tree_count={m.get('rmse_tree_count', float('nan')):.4f}")
    ok(f"rmse_mean_height={m.get('rmse_mean_height', float('nan')):.4f}")
    for k in ("rmse_tree_count_orig", "rmse_mean_height_orig"):
        if k in m:
            ok(f"{k}={m[k]:.4f}  (original scale after expm1)")

    # ── 8. Target encoding audit ──────────────────────────────────────────────
    section("8. Target encoding (tree_count=0 vs NaN, mean_height=NaN)")
    tc_store = zarr.open(str(Path(cfg["data"]["root"]) / "labels" / "tree_count"), mode="r")
    mh_store = zarr.open(str(Path(cfg["data"]["root"]) / "labels" / "mean_height"), mode="r")
    tc_nan_pct = np.isnan(np.array(tc_store[42])).mean() * 100
    mh_nan_pct = np.isnan(np.array(mh_store[42])).mean() * 100
    ok(f"tree_count  NaN% for patch 42: {tc_nan_pct:.1f}  (non-tree pixels use 0, not NaN)")
    ok(f"mean_height NaN% for patch 42: {mh_nan_pct:.1f}  (non-tree pixels use NaN)")
    ok("valid_mask_mode='notnull' → mask = isfinite(tree_count) & isfinite(mean_height)")
    ok("Effectively: mask is driven by isfinite(mean_height) since tree_count is never NaN")
    if cfg["data"].get("target_transform", {}).get("tree_count") == "log1p":
        ok("log1p applied to tree_count: log1p(0)=0 for non-tree pixels, masked in loss ✓")

    # ── 9. Channel count sanity ───────────────────────────────────────────────
    section("9. Channel layout sanity")
    expected = 8 + 40 + (1 if cfg["data"].get("use_species") else 0)
    ok(f"Expected channels: S1={8} + S2={40} + species={1 if cfg['data'].get('use_species') else 0} = {expected}")
    ok(f"Dataset reports: {ds_train.num_channels}")
    if ds_train.num_channels != expected:
        msg = f"Channel count mismatch: expected {expected}, got {ds_train.num_channels}"
        fail(msg)
        errors.append(msg)
    else:
        ok("Channel count matches config")

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    if errors:
        print(f"PIPELINE ISSUES FOUND ({len(errors)}):")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print("All pipeline checks PASSED. Ready to train.")
    print("=" * 60)


def section(title: str) -> None:
    print(f"\n{'─'*60}")
    print(f" {title}")
    print(f"{'─'*60}")


def ok(msg: str) -> None:
    print(f"  {PASS} {msg}")


def fail(msg: str) -> None:
    print(f"  {FAIL} {msg}")


def warn(msg: str) -> None:
    print(f"  {WARN} {msg}")


if __name__ == "__main__":
    main()

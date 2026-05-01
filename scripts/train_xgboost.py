"""
XGBoost baseline – pixel-wise regression for tree_count and mean_height.

Two separate XGBRegressor models are trained (one per target) using features
extracted from the same BiomassDataset used by the UNet pipeline.  Normalisation,
masking, and log1p target transforms are applied identically so that metrics are
directly comparable.

Feature extraction modes (configured via xgboost.feature_mode in configs/xgboost.yaml):
  pixel – each valid pixel is one sample; feature vector = all C input channels.
          Patches are subsampled via xgboost.subsample_pixels to control memory.
  patch – each patch is one sample; features = per-channel statistics of valid
          pixels (mean, std, min, max, median, p25, p75) → C × 7 features.
          Targets are the patch-mean of tree_count / mean_height over valid pixels.

Usage:
    python scripts/train_xgboost.py --config configs/xgboost.yaml
    python scripts/train_xgboost.py --config configs/xgboost.yaml \\
        --feature_mode patch --no_test
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import xgboost as xgb
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import BiomassDataset
from src.training.metrics import (
    TARGET_NAMES,
    _inverse_log1p_stable,
    _LOG1P_INV_MAX,
    _single_target_metrics,
    get_inverse_transforms,
)
from src.utils.config import load_config, load_norm_stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── stats used in patch-mode feature extraction ───────────────────────────────
_PATCH_STATS: List[str] = ["mean", "std", "min", "max", "median", "p25", "p75"]


# ─── feature extraction ───────────────────────────────────────────────────────

def extract_features(
    dataset: BiomassDataset,
    feature_mode: str,
    subsample_pixels: float = 0.01,
    rng: Optional[np.random.Generator] = None,
    num_workers: int = 4,
    batch_size: int = 64,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Iterate over a BiomassDataset and extract tabular features for XGBoost.

    Args:
        dataset         : BiomassDataset instance (no augmentation).
        feature_mode    : "pixel" or "patch".
        subsample_pixels: fraction of valid pixels to keep per patch (pixel mode).
        rng             : numpy random generator for reproducible subsampling.
        num_workers     : DataLoader workers.
        batch_size      : patches per batch (controls RAM peak).

    Returns:
        X      : float32 array [N, n_features]
        y_tc   : float32 array [N]  – tree_count (log1p-transformed if configured)
        y_mh   : float32 array [N]  – mean_height (log1p-transformed if configured)
    """
    if rng is None:
        rng = np.random.default_rng(0)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
    )

    X_list: List[np.ndarray] = []
    y_tc_list: List[np.ndarray] = []
    y_mh_list: List[np.ndarray] = []

    n_total = len(dataset)
    n_processed = 0

    for x_batch, y_batch, mask_batch in loader:
        # x_batch:    [B, C, H, W]  float32
        # y_batch:    [B, 2, H, W]  float32
        # mask_batch: [B, H, W]     bool
        x_np = x_batch.numpy()       # [B, C, H, W]
        y_np = y_batch.numpy()        # [B, 2, H, W]
        m_np = mask_batch.numpy()     # [B, H, W]

        for b in range(x_np.shape[0]):
            x_patch = x_np[b]         # [C, H, W]
            y_patch = y_np[b]         # [2, H, W]
            mask    = m_np[b]         # [H, W] bool

            if mask.sum() == 0:
                continue

            if feature_mode == "pixel":
                _extract_pixel(
                    x_patch, y_patch, mask,
                    subsample_pixels, rng,
                    X_list, y_tc_list, y_mh_list,
                )
            else:
                _extract_patch(
                    x_patch, y_patch, mask,
                    X_list, y_tc_list, y_mh_list,
                )

        n_processed += x_np.shape[0]
        if n_processed % max(1, (n_total // 10)) < x_np.shape[0]:
            log.info(
                "  feature extraction: %d / %d patches (%.0f%%)",
                n_processed, n_total, 100 * n_processed / n_total,
            )

    if not X_list:
        raise RuntimeError("No valid pixels/patches found – check valid_mask_mode config.")

    X    = np.concatenate(X_list, axis=0).astype(np.float32)
    y_tc = np.concatenate(y_tc_list, axis=0).astype(np.float32)
    y_mh = np.concatenate(y_mh_list, axis=0).astype(np.float32)
    return X, y_tc, y_mh


def _extract_pixel(
    x_patch: np.ndarray,   # [C, H, W]
    y_patch: np.ndarray,   # [2, H, W]
    mask:    np.ndarray,   # [H, W] bool
    subsample: float,
    rng: np.random.Generator,
    X_list: list,
    y_tc_list: list,
    y_mh_list: list,
) -> None:
    # x_valid: [N_valid, C],  y_valid: [N_valid, 2]
    x_valid = x_patch[:, mask].T          # [N_valid, C]
    y_valid = y_patch[:, mask].T          # [N_valid, 2]

    n = x_valid.shape[0]
    if subsample < 1.0:
        k = max(1, int(n * subsample))
        idx = rng.choice(n, size=k, replace=False)
        x_valid = x_valid[idx]
        y_valid = y_valid[idx]

    X_list.append(x_valid)
    y_tc_list.append(y_valid[:, 0])
    y_mh_list.append(y_valid[:, 1])


def _extract_patch(
    x_patch: np.ndarray,   # [C, H, W]
    y_patch: np.ndarray,   # [2, H, W]
    mask:    np.ndarray,   # [H, W] bool
    X_list: list,
    y_tc_list: list,
    y_mh_list: list,
) -> None:
    x_valid = x_patch[:, mask].T   # [N_valid, C]
    C = x_valid.shape[1]
    feat = np.empty(C * len(_PATCH_STATS), dtype=np.float32)
    pcts = np.percentile(x_valid, [25, 50, 75], axis=0)  # [3, C]

    for c in range(C):
        col = x_valid[:, c]
        base = c * len(_PATCH_STATS)
        feat[base + 0] = col.mean()
        feat[base + 1] = col.std()
        feat[base + 2] = col.min()
        feat[base + 3] = col.max()
        feat[base + 4] = pcts[1, c]   # median
        feat[base + 5] = pcts[0, c]   # p25
        feat[base + 6] = pcts[2, c]   # p75

    y_valid = y_patch[:, mask]   # [2, N_valid]
    X_list.append(feat[np.newaxis])
    y_tc_list.append(np.array([y_valid[0].mean()], dtype=np.float32))
    y_mh_list.append(np.array([y_valid[1].mean()], dtype=np.float32))


# ─── training ─────────────────────────────────────────────────────────────────

def train_one_target(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    xgb_cfg: dict,
    target_name: str,
) -> xgb.XGBRegressor:
    """Train one XGBRegressor and return the fitted model."""
    params = dict(
        n_estimators       = xgb_cfg.get("n_estimators", 500),
        max_depth          = xgb_cfg.get("max_depth", 6),
        learning_rate      = xgb_cfg.get("learning_rate", 0.05),
        subsample          = xgb_cfg.get("subsample", 0.8),
        colsample_bytree   = xgb_cfg.get("colsample_bytree", 0.8),
        min_child_weight   = xgb_cfg.get("min_child_weight", 5),
        tree_method        = xgb_cfg.get("tree_method", "hist"),
        device             = xgb_cfg.get("device", "cuda"),
        objective          = "reg:squarederror",
        eval_metric        = "rmse",
        random_state       = xgb_cfg.get("seed", 42),
        early_stopping_rounds = xgb_cfg.get("early_stopping_rounds", 20),
        verbosity          = 1,
    )
    log.info("Training XGBoost for %s  |  n_train=%d  n_val=%d", target_name, len(y_tr), len(y_val))
    t0 = time.time()
    model = xgb.XGBRegressor(**params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )
    best = model.best_iteration + 1 if hasattr(model, "best_iteration") else params["n_estimators"]
    log.info(
        "  %s done in %.1fs  best_iteration=%d",
        target_name, time.time() - t0, best,
    )
    return model


# ─── metrics ──────────────────────────────────────────────────────────────────

def compute_xgb_metrics(
    model_tc: xgb.XGBRegressor,
    model_mh: xgb.XGBRegressor,
    X: np.ndarray,
    y_tc: np.ndarray,
    y_mh: np.ndarray,
    inv_transforms,
) -> Dict[str, float]:
    """
    Predict with both models and compute RMSE / MAE / R² in log-space and
    original scale, matching the metric keys produced by RunningMetrics.
    """
    pred_tc = torch.from_numpy(model_tc.predict(X)).float()
    pred_mh = torch.from_numpy(model_mh.predict(X)).float()
    true_tc = torch.from_numpy(y_tc).float()
    true_mh = torch.from_numpy(y_mh).float()

    metrics: Dict[str, float] = {}
    for i, (pred, true, name) in enumerate(
        [(pred_tc, true_tc, "tree_count"), (pred_mh, true_mh, "mean_height")]
    ):
        metrics.update(_single_target_metrics(pred, true, name))
        inv = inv_transforms[i]
        if inv is not None:
            metrics.update(
                _single_target_metrics(inv(pred), inv(true), f"{name}_orig")
            )

    return metrics


def _print_metrics(metrics: Dict[str, float], phase: str) -> None:
    log.info("── %s metrics ──────────────────────────────", phase)
    for name in TARGET_NAMES:
        rmse = metrics.get(f"rmse_{name}", float("nan"))
        mae  = metrics.get(f"mae_{name}", float("nan"))
        r2   = metrics.get(f"r2_{name}", float("nan"))
        rmse_o = metrics.get(f"rmse_{name}_orig", float("nan"))
        mae_o  = metrics.get(f"mae_{name}_orig", float("nan"))
        r2_o   = metrics.get(f"r2_{name}_orig", float("nan"))
        log.info(
            "  %-14s  RMSE=%.4f  MAE=%.4f  R²=%.4f  "
            "(orig: RMSE=%.4f  MAE=%.4f  R²=%.4f)",
            name, rmse, mae, r2, rmse_o, mae_o, r2_o,
        )


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train XGBoost baseline for biomass estimation")
    p.add_argument("--config", default="configs/xgboost.yaml",
                   help="Path to YAML config (default: configs/xgboost.yaml)")
    p.add_argument("--feature_mode", choices=["pixel", "patch"], default=None,
                   help="Override xgboost.feature_mode from config")
    p.add_argument("--subsample_pixels", type=float, default=None,
                   help="Override xgboost.subsample_pixels from config")
    p.add_argument("--seed", type=int, default=None,
                   help="Override training.seed from config")
    p.add_argument("--num_workers", type=int, default=4,
                   help="DataLoader workers for feature extraction (default: 4)")
    p.add_argument("--no_test", action="store_true",
                   help="Skip test-set evaluation after training")
    p.add_argument("--save-dir", dest="save_dir", default=None, metavar="PATH",
                   help="Override xgboost.save_dir (use for per-seed artifact isolation).")
    return p.parse_args()


# ─── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    cfg = load_config(args.config)
    xgb_cfg = cfg.get("xgboost", {})

    # ── apply CLI overrides ───────────────────────────────────────────────────
    if args.feature_mode is not None:
        xgb_cfg["feature_mode"] = args.feature_mode
    if args.subsample_pixels is not None:
        xgb_cfg["subsample_pixels"] = args.subsample_pixels

    seed = args.seed if args.seed is not None else cfg.get("training", {}).get("seed", 42)
    xgb_cfg["seed"] = seed
    rng = np.random.default_rng(seed)

    feature_mode     = xgb_cfg.get("feature_mode", "pixel")
    subsample_pixels = float(xgb_cfg.get("subsample_pixels", 0.01))
    if args.save_dir is not None:
        xgb_cfg["save_dir"] = args.save_dir
    save_dir         = Path(xgb_cfg.get("save_dir", "/workspace/artifacts"))
    save_dir.mkdir(parents=True, exist_ok=True)

    log.info("Config:  %s", args.config)
    log.info("Feature mode:  %s  (subsample_pixels=%.4f)", feature_mode, subsample_pixels)
    log.info("Save dir:      %s", save_dir)

    # ── norm stats ───────────────────────────────────────────────────────────
    norm_stats_path = cfg.get("data", {}).get("norm_stats_path", "/workspace/artifacts/norm_stats.json")
    norm_stats = load_norm_stats(norm_stats_path)

    # ── inverse transforms (for original-scale metrics) ───────────────────────
    inv_transforms = get_inverse_transforms(cfg)

    # ── datasets (no augmentation) ────────────────────────────────────────────
    data_root = cfg["data"]["root"]
    log.info("Building train / val datasets …")
    ds_train = BiomassDataset(data_root, "train", cfg, norm_stats=norm_stats, transform=None)
    ds_val   = BiomassDataset(data_root, "val",   cfg, norm_stats=norm_stats, transform=None)
    log.info("  train: %d patches  |  val: %d patches", len(ds_train), len(ds_val))
    log.info("  input channels: %d  |  channel_names[0]: %s", ds_train.num_channels, ds_train.channel_names[0])

    # ── feature extraction ────────────────────────────────────────────────────
    log.info("Extracting train features (%s mode) …", feature_mode)
    X_tr, y_tc_tr, y_mh_tr = extract_features(
        ds_train, feature_mode, subsample_pixels, rng=rng, num_workers=args.num_workers
    )
    log.info("  train: X=%s  y_tc=%s", X_tr.shape, y_tc_tr.shape)

    log.info("Extracting val features …")
    X_val, y_tc_val, y_mh_val = extract_features(
        ds_val, feature_mode, subsample_pixels, rng=rng, num_workers=args.num_workers
    )
    log.info("  val: X=%s  y_tc=%s", X_val.shape, y_tc_val.shape)

    # ── feature names for XGBoost ─────────────────────────────────────────────
    if feature_mode == "pixel":
        feature_names = ds_train.channel_names
    else:
        feature_names = [
            f"{ch}_{stat}"
            for ch in ds_train.channel_names
            for stat in _PATCH_STATS
        ]

    # ── train ─────────────────────────────────────────────────────────────────
    log.info("=" * 60)
    model_tc = train_one_target(X_tr, y_tc_tr, X_val, y_tc_val, xgb_cfg, "tree_count")
    log.info("=" * 60)
    model_mh = train_one_target(X_tr, y_mh_tr, X_val, y_mh_val, xgb_cfg, "mean_height")
    log.info("=" * 60)

    # ── save models ───────────────────────────────────────────────────────────
    tc_path = save_dir / "xgb_tree_count.json"
    mh_path = save_dir / "xgb_mean_height.json"
    model_tc.save_model(str(tc_path))
    model_mh.save_model(str(mh_path))
    log.info("Saved models → %s  |  %s", tc_path, mh_path)

    # ── save run metadata (used by evaluate_xgboost.py) ───────────────────────
    run_info = {
        "feature_mode": feature_mode,
        "subsample_pixels": subsample_pixels,
        "feature_names": feature_names,
        "channel_names": ds_train.channel_names,
        "n_features": X_tr.shape[1],
        "xgb_cfg": xgb_cfg,
        "config_path": str(args.config),
    }
    run_info_path = save_dir / "xgb_run_info.json"
    with open(run_info_path, "w") as fh:
        json.dump(run_info, fh, indent=2)
    log.info("Saved run info → %s", run_info_path)

    # ── validation metrics ────────────────────────────────────────────────────
    log.info("Computing val metrics …")
    val_metrics = compute_xgb_metrics(
        model_tc, model_mh, X_val, y_tc_val, y_mh_val, inv_transforms
    )
    _print_metrics(val_metrics, "val")

    # ── test evaluation ───────────────────────────────────────────────────────
    if not args.no_test:
        log.info("Building test dataset …")
        ds_test = BiomassDataset(data_root, "test", cfg, norm_stats=norm_stats, transform=None)
        log.info("  test: %d patches", len(ds_test))

        log.info("Extracting test features …")
        X_test, y_tc_test, y_mh_test = extract_features(
            ds_test, feature_mode, subsample_pixels=1.0,  # use all pixels for test
            rng=rng, num_workers=args.num_workers,
        )
        log.info("  test: X=%s  y_tc=%s", X_test.shape, y_tc_test.shape)

        log.info("Computing test metrics …")
        test_metrics = compute_xgb_metrics(
            model_tc, model_mh, X_test, y_tc_test, y_mh_test, inv_transforms
        )
        _print_metrics(test_metrics, "test")

        test_out = {
            "n_patches": len(ds_test),
            "n_pixels":  int(X_test.shape[0]),
            "feature_mode": feature_mode,
            "model_tc": str(tc_path),
            "model_mh": str(mh_path),
            "metrics": test_metrics,
        }
        test_path = save_dir / "test_metrics_xgboost.json"
        with open(test_path, "w") as fh:
            json.dump(test_out, fh, indent=2)
        log.info("Saved test metrics → %s", test_path)

    log.info("Done.")


if __name__ == "__main__":
    main()

"""
XGBoost baseline – Stage 3 future extension stub.

This script is intentionally separate from the deep-learning training loop
(XGBoost is not a differentiable model and cannot share the Trainer class).
However, it uses the SAME:
  - BiomassDataset for data loading
  - norm_stats.json for normalisation
  - masked pixel filtering (only valid pixels contribute)
  - metric functions for fair comparison

Feature extraction strategy (to implement):
    Option A – pixel-level:
        Flatten each patch to (H*W) pixels; stack all valid pixels across patches.
        Feature vector per pixel: all C input channels (e.g. 49-dim).
        Very large matrix but conceptually simplest.

    Option B – patch-level aggregate:
        Reduce each patch to per-channel statistics (mean, std, min, max, etc.)
        → dense feature vector per patch → predict patch-level mean statistics.
        Loses spatial resolution but fast.

    Recommended: Option A with aggressive subsampling (e.g. 1% of valid pixels).

Usage (once implemented):
    python scripts/train_xgboost.py --config configs/default.yaml
    python scripts/evaluate_xgboost.py --config configs/default.yaml --model artifacts/xgb_model.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train XGBoost baseline (stub)")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--subsample_pixels", type=float, default=0.01,
                   help="Fraction of valid pixels to use per patch (memory control)")
    return p.parse_args()


def extract_pixel_features(cfg: dict, split: str, subsample: float):
    """
    TODO Stage 3:
    1. Open BiomassDataset for `split`
    2. For each patch: load x [C, H, W], y [2, H, W], mask [H, W]
    3. Select valid pixels: x_valid = x[:, mask].T  →  [N_valid, C]
    4. Optionally subsample rows for memory
    5. Return X (features), y_tc (tree_count), y_mh (mean_height)
    """
    raise NotImplementedError("Feature extraction not yet implemented (Stage 3 roadmap)")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    print("[XGBoost stub] This script is a placeholder for Stage 3.")
    print("See docstring in this file for the implementation plan.")
    print("Required packages: xgboost, scikit-learn (already in requirements.txt)")

    # Example skeleton (uncomment when implementing):
    #
    # import xgboost as xgb
    # from sklearn.metrics import mean_squared_error
    #
    # X_train, y_tc_train, y_mh_train = extract_pixel_features(cfg, "train", args.subsample_pixels)
    # X_val,   y_tc_val,   y_mh_val   = extract_pixel_features(cfg, "val",   args.subsample_pixels)
    #
    # reg_tc = xgb.XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.05, tree_method="gpu_hist")
    # reg_tc.fit(X_train, y_tc_train, eval_set=[(X_val, y_tc_val)], early_stopping_rounds=20, verbose=50)
    #
    # reg_mh = xgb.XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.05, tree_method="gpu_hist")
    # reg_mh.fit(X_train, y_mh_train, eval_set=[(X_val, y_mh_val)], early_stopping_rounds=20, verbose=50)
    #
    # reg_tc.save_model("/workspace/artifacts/xgb_tree_count.json")
    # reg_mh.save_model("/workspace/artifacts/xgb_mean_height.json")


if __name__ == "__main__":
    main()

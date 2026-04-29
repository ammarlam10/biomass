"""
XGBoost evaluation – load saved models and evaluate on any split.

Analogous to scripts/evaluate.py for the UNet pipeline.  Loads the two
XGBoost models produced by train_xgboost.py, extracts features from the
requested split, and reports the same RMSE / MAE / R² metrics in both
log1p-transformed and original scale.

Usage:
    python scripts/evaluate_xgboost.py --config configs/xgboost.yaml
    python scripts/evaluate_xgboost.py --config configs/xgboost.yaml \\
        --model_dir /workspace/artifacts --split val
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict

import xgboost as xgb

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

# Import feature extraction + metrics utilities from train_xgboost
from train_xgboost import (  # noqa: E402  (scripts/ on sys.path)
    _PATCH_STATS,
    compute_xgb_metrics,
    extract_features,
    _print_metrics,
)
from src.data.dataset import BiomassDataset
from src.training.metrics import get_inverse_transforms
from src.utils.config import load_config, load_norm_stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate XGBoost baseline on a dataset split"
    )
    p.add_argument("--config", default="configs/xgboost.yaml",
                   help="Path to YAML config (default: configs/xgboost.yaml)")
    p.add_argument("--model_dir", default=None,
                   help="Directory containing xgb_tree_count.json, xgb_mean_height.json,"
                        " and xgb_run_info.json. Defaults to xgboost.save_dir in config.")
    p.add_argument("--split", default="test", choices=["train", "val", "test"],
                   help="Dataset split to evaluate (default: test)")
    p.add_argument("--num_workers", type=int, default=4,
                   help="DataLoader workers for feature extraction (default: 4)")
    p.add_argument("--output", default=None,
                   help="Path to write output JSON. Defaults to "
                        "<model_dir>/eval_<split>_xgboost.json")
    return p.parse_args()


# ─── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    cfg = load_config(args.config)
    xgb_cfg = cfg.get("xgboost", {})

    model_dir = Path(
        args.model_dir if args.model_dir is not None
        else xgb_cfg.get("save_dir", "/workspace/artifacts")
    )

    # ── load run metadata ─────────────────────────────────────────────────────
    run_info_path = model_dir / "xgb_run_info.json"
    if not run_info_path.exists():
        raise FileNotFoundError(
            f"Run info not found at {run_info_path}. "
            "Run train_xgboost.py first to generate the models."
        )
    with open(run_info_path) as fh:
        run_info = json.load(fh)

    feature_mode = run_info["feature_mode"]
    log.info("Config:       %s", args.config)
    log.info("Model dir:    %s", model_dir)
    log.info("Split:        %s", args.split)
    log.info("Feature mode: %s", feature_mode)

    # ── load models ───────────────────────────────────────────────────────────
    tc_path = model_dir / "xgb_tree_count.json"
    mh_path = model_dir / "xgb_mean_height.json"
    for p in (tc_path, mh_path):
        if not p.exists():
            raise FileNotFoundError(f"Model not found: {p}")

    model_tc = xgb.XGBRegressor()
    model_mh = xgb.XGBRegressor()
    model_tc.load_model(str(tc_path))
    model_mh.load_model(str(mh_path))
    log.info("Loaded models from %s", model_dir)

    # ── norm stats + inverse transforms ──────────────────────────────────────
    norm_stats_path = cfg.get("data", {}).get("norm_stats_path", "/workspace/artifacts/norm_stats.json")
    norm_stats = load_norm_stats(norm_stats_path)
    inv_transforms = get_inverse_transforms(cfg)

    # ── dataset ───────────────────────────────────────────────────────────────
    data_root = cfg["data"]["root"]
    log.info("Building %s dataset …", args.split)
    dataset = BiomassDataset(data_root, args.split, cfg, norm_stats=norm_stats, transform=None)
    log.info("  %s: %d patches  |  %d input channels", args.split, len(dataset), dataset.num_channels)

    # ── feature extraction ────────────────────────────────────────────────────
    # Use subsample_pixels=1.0 for val/test (full evaluation); honour config for train
    if args.split == "train":
        subsample = float(run_info.get("subsample_pixels", xgb_cfg.get("subsample_pixels", 0.01)))
    else:
        subsample = 1.0

    log.info("Extracting %s features (subsample_pixels=%.4f) …", args.split, subsample)
    X, y_tc, y_mh = extract_features(
        dataset,
        feature_mode=feature_mode,
        subsample_pixels=subsample,
        num_workers=args.num_workers,
    )
    log.info("  X=%s  y_tc=%s", X.shape, y_tc.shape)

    # ── compute metrics ───────────────────────────────────────────────────────
    log.info("Computing metrics …")
    metrics = compute_xgb_metrics(model_tc, model_mh, X, y_tc, y_mh, inv_transforms)
    _print_metrics(metrics, args.split)

    # ── save results ──────────────────────────────────────────────────────────
    out_path = Path(args.output) if args.output else model_dir / f"eval_{args.split}_xgboost.json"
    result: Dict = {
        "split": args.split,
        "n_patches": len(dataset),
        "n_pixels": int(X.shape[0]),
        "feature_mode": feature_mode,
        "model_tc": str(tc_path),
        "model_mh": str(mh_path),
        "metrics": metrics,
    }
    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=2)
    log.info("Saved evaluation results → %s", out_path)


if __name__ == "__main__":
    main()

# XGBoost baseline — architecture and data preprocessing

This document describes the **XGBoost** tabular baseline for the same pixel-wise targets (`tree_count`, `mean_height`) as the UNet pipeline. Implementation: `scripts/train_xgboost.py`, `scripts/evaluate_xgboost.py`, config: `configs/xgboost.yaml`.

---

## Model architecture

### High-level design

Instead of one spatial neural network, the baseline trains **two independent [XGBoost](https://xgboost.readthedocs.io/) regressors**:

| Model | Target | Objective (default) |
|--------|--------|----------------------|
| `XGBRegressor` #1 | `tree_count` (in configured transform space, e.g. `log1p`) | `reg:squarederror` |
| `XGBRegressor` #2 | `mean_height` (same) | `reg:squarederror` |

Both models share the **same feature matrix `X`**; only the label vector differs. This is **not** a multi-output single learner: tree count and mean height do not share gradient updates inside one booster.

### Typical hyperparameters (from config)

Configured under the `xgboost:` key in YAML (see `configs/xgboost.yaml` for current values), including:

- `n_estimators`, `max_depth`, `learning_rate`
- `subsample`, `colsample_bytree`, `min_child_weight`
- `tree_method: hist` (often with `device: cuda` for GPU-accelerated histogram builds)
- `early_stopping_rounds` using a **validation** feature matrix
- `random_state` / seed from `training.seed`

### Saved artifacts

After training:

- `xgb_tree_count.json` / `xgb_mean_height.json` — serialized models
- `xgb_run_info.json` — `feature_mode`, `subsample_pixels`, feature names, channel list, hyperparameters (used by `evaluate_xgboost.py`)

---

## Data preprocessing

XGBoost does **not** define a separate raw-data path. It uses **`BiomassDataset`** (`src/data/dataset.py`) with **`transform=None`** (no spatial augmentation). Therefore **input normalisation, NaN handling, target transforms, and validity masks are identical to the UNet pipeline** when `configs/xgboost.yaml` mirrors the `data` block of the neural config.

### 1. Shared dataset steps (same as UNet)

- Parquet-driven **train / val / test** patch lists.
- Zarr-backed **S1, S2, optional species** stacked to `[C, H, W]`.
- **Z-score** using precomputed `norm_stats.json` (`data.norm_stats_path`); species channel uses **identity** norm from stats computation.
- **NaN → 0** on inputs after normalisation.
- **Targets**: finite masking per `valid_mask_mode`; optional **`log1p`** per `data.target_transform`.
- Output tensors per patch: `x [C,H,W]`, `y [2,H,W]`, `mask [H,W]`.

### 2. Tabular feature extraction (`extract_features`)

The neural tensors are **flattened or summarised** into rows of a design matrix `X` and label vectors `y_tc`, `y_mh`. Mode is `xgboost.feature_mode`:

#### Pixel mode (`feature_mode: pixel`)

- Each **valid pixel** (where `mask` is true) is one row.
- **Features**: the `C`-dimensional input vector at that pixel (all channels at `(h, w)`).
- **Labels**: the two target channels at that pixel (already transform-space from the dataset).

**Subsampling:** `xgboost.subsample_pixels` ∈ `(0, 1]` controls what fraction of valid pixels per patch are kept when building **train** and **val** matrices (random choice per patch, seeded). This caps memory and training time.

- **`train_xgboost.py`**: test-set feature extraction uses **`subsample_pixels=1.0`** (all valid test pixels).
- **`evaluate_xgboost.py`**: **val** and **test** use `subsample_pixels=1.0`; **train** evaluation can use the stored training subsample rate to stay tractable.

#### Patch mode (`feature_mode: patch`)

- Each **patch** is one row (if it has any valid pixels).
- **Features**: for each channel, seven statistics computed over **valid** pixels only: mean, std, min, max, median, p25, p75 → **7×C** features.
- **Labels**: **mean** of `tree_count` and **mean** of `mean_height` over valid pixels in that patch (still in dataset transform space).

Patch mode answers a **different question** (patch-level summary) than dense UNet pixel metrics; compare to UNet only if you define a matching patch-level aggregation for the neural model.

### 3. Training procedure

1. Build `BiomassDataset` for `train` and `val` with `norm_stats` and `transform=None`.
2. Run `extract_features` on train and val (respecting `feature_mode` and `subsample_pixels`).
3. Fit **tree_count** booster with early stopping on **val** RMSE.
4. Fit **mean_height** booster the same way (same `X`, different `y`).

### 4. Evaluation metrics

`compute_xgb_metrics` in `train_xgboost.py` uses the same **`_single_target_metrics`** and **`get_inverse_transforms`** as the rest of the repo (`src/training/metrics.py`), so **RMSE / MAE / R²** keys match the UNet evaluation script semantics (transform space + optional `*_orig` in original units with clamped inverse `log1p`).

---

## Comparison notes (UNet vs XGBoost)

| Aspect | Notes |
|--------|--------|
| **Data** | Same dataset class and config fields when YAML `data` blocks match. |
| **Pixel mode + full eval** | Use `evaluate_xgboost.py` on val/test (`subsample_pixels=1.0`) for the same **per-pixel** population as `scripts/evaluate.py` for UNet. |
| **Val metrics printed during `train_xgboost`** | Built on **subsampled** val pixels if `subsample_pixels < 1`; not equivalent to “full val” UNet validation unless subsample is 1. |
| **Patch mode** | Patch-level targets vs UNet’s pixel-wise head — **not** directly comparable without extra aggregation on the UNet side. |
| **Augmentation** | XGBoost path uses **no** flips/rotations; UNet training does. Held-out metrics remain comparable if both evals disable augmentation (they do). |
| **Objective** | UNet training may use **Smooth L1** / MAE / MSE on masks; XGBoost uses **squared error** in boosting. Reported test metrics are still the same metric definitions. |

---

## Quick reference

| Topic | Location |
|--------|-----------|
| Train + feature extraction | `scripts/train_xgboost.py` |
| Standalone eval | `scripts/evaluate_xgboost.py` |
| Shared preprocessing | `src/data/dataset.py` |
| Metric helpers | `src/training/metrics.py` |
| Config | `configs/xgboost.yaml` |

For the convolutional baseline’s layout and preprocessing details, see `docs/unet_architecture.md`.

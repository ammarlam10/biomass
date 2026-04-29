# UNet (ResNet50 encoder) — architecture and data preprocessing

This document describes the **UNet + ResNet50** baseline used for pixel-wise regression of `tree_count` and `mean_height`. Implementation: `src/models/unet_resnet50.py`, training: `scripts/train.py` / `src/training/trainer.py`, config: `configs/unet_resnet50.yaml`.

---

## Model architecture

### High-level design

The model is a **U-Net** from [segmentation-models-pytorch](https://github.com/qubvel/segmentation_models.pytorch): an **encoder–decoder** with skip connections. The encoder is a **ResNet50** backbone; the decoder upsamples feature maps and fuses them with encoder features at matching resolutions.

| Stage | Role |
|--------|------|
| Encoder | Extracts hierarchical spatial features from the multispectral patch. |
| Decoder | Reconstructs full spatial resolution with per-pixel predictions. |
| Skip links | Preserve fine detail from shallow encoder stages. |

### Input and output tensors

- **Input** `x`: `[B, C, H, W]` where `C` is determined by the dataset config (default: 4 Sentinel-1 seasons × 2 bands + 4 Sentinel-2 seasons × 10 bands + optional 1 species channel → **49 channels** when species is enabled).
- **Output** `pred`: `[B, 2, H, W]` with **no final activation** (`activation=None`): raw regression in the **same transformed space as the targets** (e.g. `log1p` if configured).
  - Channel 0: `tree_count`
  - Channel 1: `mean_height`

### Encoder and transfer learning

- `encoder_name` defaults to `resnet50`; `encoder_weights` may be `imagenet` (RGB pretraining) or `None`.
- When `in_channels ≠ 3`, the library **adapts the first convolution** so extra spectral channels can be learned from scratch while deeper layers may still use ImageNet weights where applicable.

### Decoder

- `decoder_channels` default: `[256, 128, 64, 32, 16]` (configurable in YAML under `model.unet.decoder_channels`).

### Registration

The builder is registered as `unet_resnet50` in `src/models/factory.py` and selected with `model.name: unet_resnet50` in the config.

---

## Data preprocessing

All steps below are implemented in `src/data/dataset.py` (`BiomassDataset`) unless noted.

### 1. Patch index and splits

- A **parquet** file (`data.split_file`) lists which Zarr patch indices belong to `train`, `val`, or `test` (`data.split_column`, `data.patch_idx_column`).
- Each sample loads one patch by index from Zarr stores under `data.root`.

### 2. Input assembly (`_load_input`)

Channels are stacked in a **fixed order** (see `build_channel_names` in `src/data/dataset.py`):

1. For each configured S1 season: VV and VH → `[2, H, W]` per season.
2. For each configured S2 season: ten optical bands → `[10, H, W]` per season.
3. If `use_species: true`: `tree_species` → `[1, H, W]` (categorical class index per pixel).

The result is `x` as **float32** `[C, H, W]` (patch height/width come from the Zarr data, typically 128×128).

### 3. Per-channel normalisation

- **Before training**, run `scripts/compute_stats.py` with the same `data` settings. It scans **training-split** patches only, accumulates per-channel sum and sum of squares on **finite** pixels, and writes `norm_stats.json` to `data.norm_stats_path`.
- Formula: `x_ch = (x_ch - mean_ch) / std_ch`, with `std` floored to avoid division by zero.
- **Special case — `tree_species`:** values are **class IDs**, not reflectance. `compute_stats.py` forces **identity** normalisation (`mean=0`, `std=1`) so the network receives raw codes, not z-scored ordinals.
- **Missing inputs:** after normalisation, `np.nan_to_num(..., nan=0.0)` replaces NaNs (e.g. missing acquisitions) with zero.

### 4. Targets and validity mask (`_load_targets`)

- `tree_count` and `mean_height` are read from Zarr label stores.
- **`valid_mask_mode`**:
  - `notnull` (default): valid where both targets are finite (`tree_count` is typically finite everywhere; `mean_height` is NaN off trees, so the mask selects tree-bearing pixels).
  - `positive`: additionally requires `tree_count > 0`.

### 5. Target transforms (training / eval space)

Configured under `data.target_transform` (per target: `none` or `log1p`).

- For `log1p`: after sanitising non-finite values to 0, apply `log1p(max(value, 0))` so the network is trained in **log1p space**.
- Stacked into `y`: `[2, H, W]`.

### 6. Training-only: spatial augmentation

`scripts/train.py` attaches `build_train_transform()` from `src/data/transforms.py` **only to the train** `DataLoader`:

- Random horizontal flip (p=0.5)
- Random vertical flip (p=0.5)
- Random 90° rotation × {1,2,3} (p=0.75)

Augmentations apply **jointly** to `x`, `y`, and `mask` so geometry stays aligned. Validation and test use **no** augmentation (`build_val_transform()` returns `None`). `scripts/evaluate.py` uses `transform=None`.

### 7. Loss (not preprocessing, but same mask)

Training uses a **masked** regression loss (`masked_mse`, `masked_smooth_l1`, or `masked_mae` from `src/losses/masked_regression.py`) with optional per-target weights. Only **valid-masked** pixels contribute.

### 8. Metrics and inverse transforms

Evaluation (`scripts/evaluate.py`, trainer validation) computes RMSE / MAE / R² in **transform space** and, when `log1p` is used, in **original units** via clamped `expm1` inverses (`src/training/metrics.py` — avoids overflow on extreme predictions).

---

## Quick reference

| Topic | Location |
|--------|-----------|
| Model wrapper | `src/models/unet_resnet50.py` |
| Dataset + preprocessing | `src/data/dataset.py` |
| Augmentations | `src/data/transforms.py` |
| Norm stats | `scripts/compute_stats.py` → `norm_stats.json` |
| Training loop | `src/training/trainer.py` |
| Config | `configs/unet_resnet50.yaml` |

---

## Alignment with the XGBoost baseline

The UNet pipeline shares the **same** `BiomassDataset` contract: same Zarr layout, same `norm_stats`, same `target_transform` and `valid_mask_mode` when configs are kept in sync. That allows **fair comparison** of held-out metrics with the XGBoost path documented in `docs/xgboost_architecture.md` (especially in **pixel** feature mode with full-pixel evaluation).

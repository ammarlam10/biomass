# Dataset Documentation

> Dataset root (host): `/work/ammar/sslrp/data/biomass`
> Dataset root (container): `/data` (mounted read-only)
> All measurements confirmed by `scripts/pipeline_check.py` against the live data.

---

## Overview

The dataset covers forest stands in **Bavaria, Germany**. It consists of
13,626 geo-referenced image patches of 128×128 pixels extracted from a
pre-defined spatial grid. Each patch has four seasonal acquisitions of
Sentinel-1 SAR and Sentinel-2 multispectral imagery, a tree species
classification map, and per-pixel regression targets derived from airborne
lidar data.

---

## File Structure

```
/data/
├── inputs/
│   ├── s1_summer/         Sentinel-1 SAR – summer
│   ├── s1_autumn/         Sentinel-1 SAR – autumn
│   ├── s1_spring/         Sentinel-1 SAR – spring
│   ├── s1_winter/         Sentinel-1 SAR – winter
│   ├── s2_summer/         Sentinel-2 MSI – summer
│   ├── s2_autumn/         Sentinel-2 MSI – autumn
│   ├── s2_spring/         Sentinel-2 MSI – spring
│   ├── s2_winter/         Sentinel-2 MSI – winter
│   └── tree_species/      Tree species classification map
├── labels/
│   ├── tree_count/        ← primary regression target
│   ├── mean_height/       ← primary regression target
│   ├── tree_density/      ancillary
│   ├── median_height/     ancillary
│   ├── height_variance/   ancillary
│   └── tree_count_variance/ ancillary
├── metadata/
│   ├── grid_shape_arr     Spatial grid shape
│   └── transform_coeffs   Affine transform coefficients
└── patch_index_subset.parquet   Patch index and train/val/test split
```

All input and label stores are **Zarr v2 directory stores** (one chunk per
patch: `chunks=(1, 128, 128, bands)`).

---

## Patch Index File

**File:** `patch_index_subset.parquet`

| Column | Type | Description |
|--------|------|-------------|
| `patch_id` | int64 | Original grid cell ID |
| `zarr_idx` | int64 | **Index into zarr arrays** (0–13625) — use as `patch_idx_column` in config |
| `split` | str | `train` / `val` / `test` |
| `center_x`, `center_y` | float64 | Patch centroid in EPSG:25832 (easting / northing) |
| `row_start`, `row_end`, `col_start`, `col_end` | int64 | Pixel extent in full raster |
| `valid_pixel_pct` | float64 | Fraction of S2 pixels with valid (non-NaN) reflectance |
| `tree_pixel_pct` | float64 | Fraction of pixels with lidar-confirmed tree cover |
| `mean_tree_count` | float64 | Patch-level mean tree count (tree pixels only) |
| `mean_tree_count_variance` | float64 | Variance of tree count |
| `block_col`, `block_row`, `block_id` | int64 | Spatial block assignment (used for split) |
| `distance_to_nearest_test_km` | float64 | Spatial buffer metric |
| `buffered` | bool | Whether patch is in the spatial buffer zone |
| `in_bavaria` | bool | All patches are in Bavaria (`True` for all) |

> **Config mapping:** `data.split_column: split` / `data.patch_idx_column: zarr_idx`

---

## Train / Val / Test Split

| Split | Patches | Tree-pixel mean | Zero-tree patches |
|-------|---------|-----------------|-------------------|
| **train** | 9,327 | 32.4 % | 288 |
| **val** | 2,446 | 31.2 % | 48 |
| **test** | 1,853 | 26.7 % | 111 |
| **total** | **13,626** | **31.4 %** | **447** |

The split is **spatially blocked** — patches are grouped into spatial blocks
before assignment, preventing spatial autocorrelation between splits.
`distance_to_nearest_test_km` can be used to inspect the spatial buffer
around the test set.

---

## Input Modalities

### Sentinel-1 SAR (4 seasons × 2 bands = 8 channels)

| Property | Value |
|----------|-------|
| Sensor | Sentinel-1 GRD |
| Polarimetry | Linear gamma-0 (VV, VH) |
| Orbit | Ascending |
| Seasons | summer, autumn, spring, winter |
| Zarr shape | `(13626, 128, 128, 2)` |
| Dtype | float32 |
| Chunk | `(1, 128, 128, 2)` |
| Missing data | NaN (rare; filled to 0 by dataset loader) |

**Channel order in model input (indices 0–7):**

| Index | Name |
|-------|------|
| 0 | s1_summer_VV |
| 1 | s1_summer_VH |
| 2 | s1_autumn_VV |
| 3 | s1_autumn_VH |
| 4 | s1_spring_VV |
| 5 | s1_spring_VH |
| 6 | s1_winter_VV |
| 7 | s1_winter_VH |

### Sentinel-2 MSI (4 seasons × 10 bands = 40 channels)

| Property | Value |
|----------|-------|
| Sensor | Sentinel-2 L2A |
| Bands | B2 B3 B4 B8 B5 B6 B7 B8A B11 B12 |
| Seasons | summer, autumn, spring, winter |
| Zarr shape | `(13626, 128, 128, 10)` |
| Dtype | float32 |
| Chunk | `(1, 128, 128, 10)` |
| Missing data | NaN (very rare; ~99.8% valid per patch) |
| Value range | Raw surface reflectance (e.g. 1000–8000 DN) |

**Channel order in model input (indices 8–47):**
Repeated for each season in `[summer, autumn, spring, winter]`:

| Offset | Band | Description |
|--------|------|-------------|
| +0 | B2 | Blue (490 nm) |
| +1 | B3 | Green (560 nm) |
| +2 | B4 | Red (665 nm) |
| +3 | B8 | NIR broad (842 nm) |
| +4 | B5 | Red-edge 1 (705 nm) |
| +5 | B6 | Red-edge 2 (740 nm) |
| +6 | B7 | Red-edge 3 (783 nm) |
| +7 | B8A | NIR narrow (865 nm) |
| +8 | B11 | SWIR 1 (1610 nm) |
| +9 | B12 | SWIR 2 (2190 nm) |

### Tree Species (1 channel)

| Property | Value |
|----------|-------|
| Zarr shape | `(13626, 128, 128)` |
| Dtype | uint8 |
| Chunk | `(1, 128, 128)` |
| Description | Categorical class IDs per pixel |
| Known class IDs | 0, 1, 2, 3, 4, 5, 6, 7, 11 |
| Missing / non-tree | 0 |

Species channel is cast to float32 and appended as model input channel 48.
Toggle via `data.use_species: false` in config to exclude it (model will then have 48 channels).

### Full channel layout (49 channels total)

```
[  0:  8 ]  S1 – 4 seasons × 2 bands (VV, VH)
[  8: 48 ]  S2 – 4 seasons × 10 bands
[    48  ]  tree_species (optional, controlled by data.use_species)
```

---

## Target Labels (Primary)

Both targets are stored as float32 Zarr arrays of shape `(13626, 128, 128)`.

### `tree_count` — number of trees per pixel

| Property | Value |
|----------|-------|
| Units | count (integer-valued, stored as float32) |
| Non-tree pixels | **0** (not NaN) |
| Tree pixels | 1–8 (confirmed from 976k valid pixel sample) |
| Median (tree pixels) | 1.0 |
| Mean (tree pixels) | 1.48 |
| 95th percentile | 3.0 |
| Max observed | 8.0 |
| Training transform | `log1p` (config: `data.target_transform.tree_count: log1p`) |

> Because `tree_count` uses **0** for non-tree pixels and never NaN, the
> validity mask is **not** derived from `tree_count`.

### `mean_height` — mean tree height per pixel (metres)

| Property | Value |
|----------|-------|
| Units | metres |
| Non-tree pixels | **NaN** |
| Tree pixels | 5.5 – 53.7 m |
| Median (tree pixels) | 23.1 m |
| Mean (tree pixels) | 21.9 m |
| 95th percentile | 33.6 m |
| Max observed | 53.7 m |
| Training transform | `log1p` (config: `data.target_transform.mean_height: log1p`) |

> `mean_height` NaN pattern drives the **valid pixel mask**.

---

## Target Labels (Ancillary — available but not trained on by default)

| Name | Description |
|------|-------------|
| `tree_density` | Tree density (stems / unit area) |
| `median_height` | Median tree height per pixel (m) |
| `height_variance` | Variance of tree height per pixel |
| `tree_count_variance` | Variance of tree count per pixel |

All have shape `(13626, 128, 128)` float32 and can be added as additional
regression heads by extending `model.classes` and `loss_weights` in config.

---

## Sparse Tree Pixel Handling

Tree pixels represent on average **31.4%** of each patch, with high variance:

| Percentile | tree_pixel_pct |
|------------|---------------|
| p0 | 0.0 % |
| p5 | 1.0 % |
| p25 | 10.8 % |
| p50 | 25.9 % |
| p75 | 48.7 % |
| p90 | 70.5 % |
| p95 | 78.1 % |
| p100 | 90.9 % |

447 patches (3.3%) have **zero tree pixels**.

### How the pipeline handles sparsity

1. **Valid pixel mask** (`valid_mask_mode: notnull`):
   `valid = isfinite(tree_count) & isfinite(mean_height)`
   Effectively `valid = isfinite(mean_height)` since `tree_count` is never NaN.

2. **Masked loss** (`src/losses/masked_regression.py`):
   Only the `~31%` valid pixels contribute to the MSE loss each step.
   Zero-mask batches return loss=0 (differentiable, no crash).

3. **`log1p` transform on `tree_count`**:
   Compresses the heavy tail (values 0–8 → 0–2.2) and reduces scale mismatch
   with `mean_height`.

4. **Optional stricter mask** (`valid_mask_mode: positive`):
   Requires `tree_count > 0`, restricting to pixels with confirmed trees and
   excluding background forest-free pixels. Use to focus on regression quality
   rather than forest detection.

---

## Normalisation Statistics

Computed by `scripts/compute_stats.py` on the **train split only**
(to avoid data leakage) and saved to `artifacts/norm_stats.json`.
Applied in `BiomassDataset.__getitem__` via z-score normalisation per channel.

Run:
```bash
docker compose run --rm compute_stats
```

---

## Data Loading

Implemented in [`src/data/dataset.py`](src/data/dataset.py) — `BiomassDataset`:

```python
ds = BiomassDataset(
    root="/data",
    split="train",          # "train" | "val" | "test"
    cfg=cfg,
    norm_stats=norm_stats,  # from artifacts/norm_stats.json
    transform=build_train_transform(),
)
x, y, mask = ds[0]
# x    : float32 [49, 128, 128]  – normalised, NaN→0
# y    : float32 [ 2, 128, 128]  – (tree_count_log1p, mean_height)
# mask : bool    [128, 128]       – True on tree pixels
```

The dataset opens all Zarr stores **lazily** at construction and reads one
patch at a time in `__getitem__`, making it safe to use with PyTorch
`DataLoader(num_workers=12)`.

---

## Quick Data Checks

```bash
# Inspect parquet columns
docker compose run --rm compute_stats \
  python scripts/compute_stats.py --config configs/unet_resnet50.yaml --inspect

# Full pipeline validation (shapes, NaN, model forward, loss)
docker compose run --rm check

# EDA notebook (sparsity plots, sample patches, band histograms)
docker compose up notebook
# → http://localhost:8888
```

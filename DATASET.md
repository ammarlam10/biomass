# Dataset Documentation

This document describes the biomass regression dataset, how it is laid out on disk, how configuration maps to files, and how the training code consumes it end-to-end. It is written to be read sequentially: high-level context first, then on-disk layout, then index and splits, then modalities and targets, then sparsity and losses, then preprocessing and loading, finally operational commands.

---

## Paths and runtime layout

> **Dataset root (`data.root` in configs):** `/data/ammar/4g.zarr` inside the training container.  
> **Docker bind (host → container):** `/data/ammar/4g.zarr:/data/ammar/4g.zarr:ro` (see `docker-compose.yml`).  
> **Code / artifacts:** the repository is mounted at `/workspace` from host `/work/ammar/sslrp/biomass`; normalisation statistics default to `/workspace/artifacts/norm_stats.json`.  
> **Validation:** `scripts/pipeline_check.py` exercises the real parquet, Zarr stores, `BiomassDataset`, `DataLoader`, model forward, masked loss, backward pass, and metrics against `configs/unet_resnet50.yaml`.

**Why these paths matter:** every script that loads data resolves `data.root` and `data.split_file` from YAML. If you run outside Docker, point `root` and `split_file` at the same logical tree on your filesystem. Inside Docker, the Zarr tree must appear at the path given in the config so that `Path(root) / "inputs" / ...` exists.

---

## Overview

The dataset covers forest stands in **Bavaria, Germany**. It consists of **13,626** geo-referenced image patches of **128×128** pixels, each cell aligned to a **pre-defined spatial grid**. For each patch index, the stores provide:

1. **Four seasonal stacks** of Sentinel-1 SAR (VV/VH) and Sentinel-2 L2A reflectance (ten bands per season), so the model can use phenology and multi-temporal context.  
2. A **tree species** raster (categorical class ID per pixel).  
3. **Per-pixel regression targets** derived from airborne lidar: primarily `tree_count` and `mean_height`, with additional ancillary label rasters.

**What “one sample” is in PyTorch terms:** `BiomassDataset[i]` returns one patch: a tensor `x` of shape `[C, 128, 128]` (multi-modal inputs stacked as channels), a tensor `y` of shape `[2, 128, 128]` (two targets), and a boolean mask `[128, 128]` selecting which pixels participate in the loss. The patch index `i` is *not* the Zarr row: it is the *i*-th row of the parquet after filtering by `split`; the actual Zarr index comes from the `patch_idx_column` (typically `zarr_idx`).

---

## File structure

All paths below are relative to **`data.root`** (for example `/data/ammar/4g.zarr` in `configs/unet_resnet50.yaml`).

```
{data.root}/
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

### Zarr layout and chunking

All input and label rasters named above are stored as **Zarr v2 directory stores**. The first dimension of each array is the **patch index** (length 13,626). For a given patch `p`, reading `store[p]` returns a single patch slice.

**Chunking:** stores use **`chunks=(1, 128, 128, bands)`** (or `(1, 128, 128)` for single-band / species rasters). That means:

- Each chunk corresponds to **one patch** in the first dimension.  
- Sequential or random access by patch index maps cleanly to chunk reads, which is efficient for the training loop (one patch per `__getitem__`).  
- `DataLoader` workers each open their own Zarr handles (`scripts/compute_stats.py` does the same in worker processes), avoiding shared mutable state across processes.

### Parquet vs Zarr

- **Parquet** (`patch_index_subset.parquet`) holds **tabular metadata per patch**: split assignment, spatial block IDs, quality fractions, centroids, and crucially the **integer index** used to subscript the Zarr arrays.  
- **Zarr** holds **dense rasters**. The join between them is: `parquet.patch_idx_column` → first dimension of each Zarr array.

### Metadata folder

`metadata/grid_shape_arr` and `metadata/transform_coeffs` describe the **global grid** and georeferencing. The training code in this repository does not read these files directly for sampling; they are useful for **geospatial context**, aligning patches to map coordinates, or building visualisations. Patch extents and centroids in the parquet (`row_*`, `col_*`, `center_x`, `center_y`) are derived from this grid.

---

## Patch index file

**File:** `{data.root}/patch_index_subset.parquet` (YAML key: `data.split_file`).

### How the loader uses this file (step by step)

1. **`BiomassDataset.__init__`** reads the parquet with `pandas.read_parquet`.  
2. It reads **`data.split_column`** (default `split`) and keeps rows where the value equals the constructor argument `split` (`train`, `val`, or `test`).  
3. It extracts **`data.patch_idx_column`** (in repo configs: **`zarr_idx`**) into a NumPy array `patch_indices`.  
4. **`__len__`** returns `len(patch_indices)`.  
5. **`__getitem__(i)`** sets `patch_idx = patch_indices[i]`, then reads `zarr_store[patch_idx]` for every opened array.

So the **dataset index `i`** runs from `0` to *number of patches in that split minus one*; **`patch_idx`** runs in `0 … 13625` and is the **Zarr first dimension**.

### Column reference

| Column | Type | Description |
|--------|------|-------------|
| `patch_id` | int64 | Original grid cell ID (may differ from Zarr row order). |
| `zarr_idx` | int64 | **Index into Zarr arrays** (0–13625). Set `data.patch_idx_column: zarr_idx` in YAML when this column exists. |
| `split` | str | `train` / `val` / `test`. |
| `center_x`, `center_y` | float64 | Patch centroid in **EPSG:25832** (easting / northing, metres). |
| `row_start`, `row_end`, `col_start`, `col_end` | int64 | Pixel extent of the patch in the **full** reference raster. |
| `valid_pixel_pct` | float64 | Fraction of Sentinel-2 pixels with **finite** reflectance (quality / missing-data aware). |
| `tree_pixel_pct` | float64 | Fraction of pixels with **lidar-confirmed tree cover** (used in dataset statistics and sparsity analysis). |
| `mean_tree_count` | float64 | Patch-level mean tree count (typically over tree pixels). |
| `mean_tree_count_variance` | float64 | Variance of tree count (patch-level summary). |
| `block_col`, `block_row`, `block_id` | int64 | **Spatial block** assignment used when constructing the split (patches in the same block tend to be assigned to the same split to reduce leakage). |
| `distance_to_nearest_test_km` | float64 | Distance-style metric to the test region (buffer / spatial separation analysis). |
| `buffered` | bool | Whether the patch lies in a **buffer** zone around test (dataset design for spatial holdout). |
| `in_bavaria` | bool | Geographic filter flag; documented as `True` for all patches in this release. |

### Config mapping and code defaults

- **Recommended in this repo:** `data.split_column: split` and `data.patch_idx_column: zarr_idx` (see `configs/unet_resnet50.yaml`, `configs/xgboost.yaml`, `configs/clay.yaml`, etc.).  
- **Fallback in code:** if `patch_idx_column` is **omitted** from YAML, `BiomassDataset` defaults to **`patch_idx`** (`src/data/dataset.py`). If your parquet only has `zarr_idx`, you **must** set `patch_idx_column` explicitly or construction will raise a `KeyError` from `_validate_parquet_columns`.

---

## Train / validation / test split

| Split | Patches | Tree-pixel mean | Zero-tree patches |
|-------|---------|-----------------|-------------------|
| **train** | 9,327 | 32.4 % | 288 |
| **val** | 2,446 | 31.2 % | 48 |
| **test** | 1,853 | 26.7 % | 111 |
| **total** | **13,626** | **31.4 %** | **447** |

### Why the split is spatially blocked

If patches were assigned **independently at random**, neighbouring patches could land in both train and test. Because forest structure and acquisition artefacts are **spatially correlated**, the model could **indirectly memorise** test locations from training neighbours (optimistic evaluation).

This dataset mitigates that by **grouping patches into spatial blocks** (`block_id`, `block_row`, `block_col`) and assigning **whole blocks** (or block groups) to splits. That enforces a **geographic separation** between train, validation, and test.

### How to use buffer-related columns

- **`distance_to_nearest_test_km`** and **`buffered`** support analysis of how close a patch is to the test region. They are useful for **error analysis** (e.g. whether errors cluster near boundaries) or for **filtering** experiments; the default training code does not automatically exclude buffered patches unless you add that logic.

### “Zero-tree” patches

**447** patches (about **3.3%**) have **no tree pixels** at all (`tree_pixel_pct == 0`). For those patches, the valid mask can be **all False**. The masked loss is designed so that an all-false mask still yields a **differentiable zero loss** (see Sparse section below).

---

## Input modalities

The model consumes a **single stack of channels** built in a **fixed order** in `BiomassDataset._load_input` (`src/data/dataset.py`): for each season in `data.s1_seasons`, append VV then VH; for each season in `data.s2_seasons`, append the ten bands in `S2_BANDS` order; optionally append species as one channel. Default season lists are `["summer", "autumn", "spring", "winter"]` for both S1 and S2.

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
| Missing data | NaN (rare; replaced after normalisation, see Data loading) |

**From disk to tensor (conceptual steps):**

1. Read `s1_{season}[patch_idx]` → array shaped `[128, 128, 2]` (H, W, band).  
2. Transpose to `[2, H, W]` so each band is a channel plane.  
3. Concatenate seasons in config order → **8** channel planes total.

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
| Missing data | NaN (very rare; ~99.8% valid per patch, patch-level) |
| Value range | Raw surface reflectance (typical DN ranges, e.g. thousands) |

**From disk to tensor:** same pattern as S1: each season yields `[10, 128, 128]`, concatenated along the channel axis after all S1 channels, producing indices **8–47**.

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

### Tree species (optional 1 channel)

| Property | Value |
|----------|-------|
| Zarr shape | `(13626, 128, 128)` |
| Dtype | uint8 (stored); read as float32 for the stack |
| Chunk | `(1, 128, 128)` |
| Description | Categorical class IDs per pixel |
| Known class IDs | 0, 1, 2, 3, 4, 5, 6, 7, 11 |
| Missing / non-tree | 0 |

**Behaviour:**

- If `data.use_species` is **true** (default in repo configs), the dataset opens `inputs/tree_species` and appends **one** channel at index **48**.  
- If **false**, the species Zarr is not opened and the model receives **48** input channels; you must recompute **`norm_stats.json`** with the same setting so channel counts align.

Species values are **not spectral reflectance**; `compute_stats.py` forces **identity** normalisation (mean 0, std 1) for the `tree_species` channel so z-scoring does not distort categorical codes.

### Full channel layout (48 or 49 channels)

With `data.use_species: true`: **49** channels. With `data.use_species: false`: **48** channels.

```
[  0 :  8 ]  S1 – 4 seasons × 2 bands (VV, VH)
[  8 : 48 ]  S2 – 4 seasons × 10 bands
[     48   ]  tree_species (only if data.use_species is true)
```

**Consistency rule:** `build_channel_names` in `dataset.py` must match the order used in `compute_stats.py` and in `_load_input`; any new modality requires updating all three.

---

## Target labels (primary)

Both primary targets are float32 Zarr arrays of shape **`(13626, 128, 128)`**.

### `tree_count` — number of trees per pixel

| Property | Value |
|----------|-------|
| Units | Count (integer-valued, stored as float32) |
| Non-tree pixels | **0** (not NaN) |
| Tree pixels | 1–8 (from a large valid-pixel sample) |
| Median (tree pixels) | 1.0 |
| Mean (tree pixels) | 1.48 |
| 95th percentile | 3.0 |
| Max observed | 8.0 |
| Training transform | `log1p` when `data.target_transform.tree_count: log1p` |

**Interpretation:** each pixel answers “how many trees” under the lidar-derived definition. Background is explicit **zero**, not missing data.

**Mask interaction:** because `tree_count` is finite everywhere on valid grid cells, **`isfinite(tree_count)` does not restrict the mask**. The mask is effectively driven by **`mean_height`** (see below).

**Training-time transform:** when `log1p` is enabled, the dataset applies `log1p(max(tc, 0))` after replacing non-finite values with 0 for stacking; on non-tree pixels, `log1p(0)=0`. Metrics at evaluation use the inverse (`expm1`) where configured in `src/training/metrics.py`.

### `mean_height` — mean tree height per pixel (metres)

| Property | Value |
|----------|-------|
| Units | Metres |
| Non-tree pixels | **NaN** |
| Tree pixels | 5.5 – 53.7 m (observed range) |
| Median (tree pixels) | 23.1 m |
| Mean (tree pixels) | 21.9 m |
| 95th percentile | 33.6 m |
| Max observed | 53.7 m |
| Training transform | `log1p` when `data.target_transform.mean_height: log1p` |

**Interpretation:** height is only defined where trees exist; open ground is **missing** in the label raster.

**Mask interaction:** **`mean_height` NaN defines “no label here”`**, which becomes **`valid_mask == False`**. Those pixels are ignored by the masked loss even if inputs exist.

---

## Target labels (ancillary)

These rasters exist under `labels/` but are **not loaded** by the current `BiomassDataset` implementation (only `tree_count` and `mean_height` are opened in `dataset.py`).

| Name | Description |
|------|-------------|
| `tree_density` | Tree density (stems per unit area). |
| `median_height` | Median tree height per pixel (m). |
| `height_variance` | Variance of tree height per pixel. |
| `tree_count_variance` | Variance of tree count per pixel. |

All are shape **`(13626, 128, 128)`** float32 in Zarr.

**If you want to train on them:** you would extend **`BiomassDataset`** to read extra arrays, extend the **model output** beyond two channels (today `classes=2` in `src/models/unet_resnet50.py`), extend **`MaskedRegressionLoss`** / `training.loss_weights`, and extend **metrics** for reporting. There is no single YAML flag for that today.

---

## Sparse tree pixels and the loss pipeline

Tree cover is **sparse**: `tree_pixel_pct` varies strongly across patches.

| Percentile | tree_pixel_pct |
|------------|----------------|
| p0 | 0.0 % |
| p5 | 1.0 % |
| p25 | 10.8 % |
| p50 | 25.9 % |
| p75 | 48.7 % |
| p90 | 70.5 % |
| p95 | 78.1 % |
| p100 | 90.9 % |

**447** patches (~**3.3%**) have **zero** tree pixels.

### End-to-end steps from raw store to loss

1. **Load targets** (`_load_targets`): read `tree_count` and `mean_height` for `patch_idx`.  
2. **Build valid mask** (`valid_mask_mode`):  
   - **`notnull` (default):** `valid = isfinite(tree_count) & isfinite(mean_height)` → effectively **`isfinite(mean_height)`** since `tree_count` has no NaNs for valid cells.  
   - **`positive`:** additionally requires **`tree_count > 0`**, dropping pixels where count is zero even if height were finite (stricter “tree only” supervision).  
3. **Fill targets for tensor stacking:** non-finite values are mapped to **0.0** in NumPy before `log1p`, so `y` has no NaNs; **the mask** tells the loss which positions matter.  
4. **Apply `log1p`** per `data.target_transform` to produce the `y` tensor consumed by the loss.  
5. **Forward:** model outputs `pred` with shape `[B, 2, H, W]` for the baseline UNet.  
6. **Loss** (`src/losses/masked_regression.py`): for each target channel, gather `pred[:, i][mask]` and `target[:, i][mask]` and apply the chosen reduction (`mse`, `smooth_l1`, or `mae`). Weights come from `training.loss_weights` (`tree_count`, `mean_height`).  
7. **Empty mask:** if a batch has **no** `True` mask pixels, the loss returns **`pred.sum() * 0.0`** so the graph stays connected and training does not crash on rare all-background batches.

### Why `log1p` on `tree_count`

Counts have a **heavy tail** (up to 8). `log1p` maps `[0,8]` into a bounded range roughly `[0, 2.2]`, stabilising optimisation next to height targets that are also `log1p`-compressed.

---

## Normalisation statistics (`compute_stats.py`)

**Purpose:** per-channel **mean and standard deviation** on the **training split only**, saved as JSON for z-score normalisation in `BiomassDataset.__getitem__`.

### Step-by-step what the script does

1. Load YAML (`--config`), read `data.root`, `data.split_file`, `split_column`, `patch_idx_column`, seasons, `use_species`.  
2. If **`--inspect`**: print column names and sample rows, then exit (no stats).  
3. Collect all **`zarr_idx`** (or whatever `patch_idx_column` is) for rows with `split == "train"`.  
4. Optionally **`--subsample`** a fraction of train patches (reproducible RNG) for faster approximate stats.  
5. Split the index list into **`--num_workers`** chunks (default in script: half CPU count; Docker `compute_stats` service passes `12`).  
6. Each worker process opens its **own** Zarr readers, iterates its patch list, and accumulates **sum**, **sum of squares**, and **count** of **finite** values per channel (species included as finite counts).  
7. The main process merges partial sums and computes **mean** and **std** via the standard merge formula (`var = E[x²] - E[x]²`).  
8. For **`tree_species`**, overwrite that channel’s mean with **0** and std with **1** (identity transform).  
9. Write JSON to **`data.norm_stats_path`** (default under Docker: `/workspace/artifacts/norm_stats.json`), including `channel_names` aligned with `build_channel_names`.

### When you must re-run it

- Changed **`use_species`**, **season lists**, or **any path** that changes which voxels are read.  
- Swapped to a **different config** with different channel layout (e.g. Clay vs UNet only matters if `data.*` differs).  
- After **updating the dataset** itself.

### Applying stats in the dataset

If `norm_stats` is not `None`, **`x = (x - mean) / std`** channel-wise, then **`np.nan_to_num(x, nan=0)`** so missing SAR/S2 becomes zero **after** scaling. If `norm_stats` is `None` (some tests or `pipeline_check` sections), inputs are raw float32 with NaNs still replaced to zero at the end of `__getitem__`.

**Run full stats (Docker service default command):**

```bash
docker compose run --rm compute_stats
```

---

## Data loading (`BiomassDataset`)

Implementation: [`src/data/dataset.py`](src/data/dataset.py).

### Initialisation (once per dataset instance)

1. Resolve `data` section (`cfg.get("data", cfg)` allows passing a sub-dict).  
2. Read season lists and flags (`use_species`, `valid_mask_mode`, `target_transform`).  
3. Load parquet, validate required columns, filter by split, store **`patch_indices`**.  
4. **`zarr.open(..., mode="r")`** for each S1/S2 season path under `root/inputs/`, optionally `tree_species`, and for `labels/tree_count` and `labels/mean_height`. No large arrays are read yet.  
5. Build **`channel_names`** for logging and stats alignment.  
6. If **`norm_stats`** provided, broadcast mean/std to shape `[C, 1, 1]` for vectorised normalisation.

### Each `__getitem__(i)` call

1. `patch_idx = patch_indices[i]`.  
2. **`_load_input`:** read and concatenate modalities → `[C, 128, 128]` float32.  
3. **`_load_targets`:** read both labels, compute **`valid_mask`**.  
4. Normalise `x`; **`nan_to_num`** on `x`.  
5. Prepare **`tc_safe` / `mh_safe`** (non-finite → 0), apply **`log1p`** per config.  
6. Stack into **`y`** `[2, H, W]`; convert to torch; apply optional **`transform(x, y, mask)`** (augmentations).  
7. Return **`(x_t, y_t, mask_t)`**.

### Why this works with `DataLoader` workers

Each worker process gets a **copy** of the dataset object (including open Zarr stores on many systems this is picklable / re-opened depending on start method). Reading **one patch per step** keeps memory bounded. **`num_workers: 12`** in config matches the comment in `unet_resnet50.yaml` about saturating one NUMA node; reduce if you see worker OOM or diminishing returns.

### Example construction

```python
from src.utils.config import load_config, load_norm_stats
from src.data.dataset import BiomassDataset
from src.data.transforms import build_train_transform

cfg = load_config("configs/unet_resnet50.yaml")
norm_stats = load_norm_stats(cfg["data"]["norm_stats_path"])

ds = BiomassDataset(
    root=cfg["data"]["root"],  # e.g. /data/ammar/4g.zarr
    split="train",
    cfg=cfg,
    norm_stats=norm_stats,
    transform=build_train_transform(),
)
x, y, mask = ds[0]
# x    : float32 [C, 128, 128]  – C = 49 if use_species else 48; normalised; NaN→0
# y    : float32 [ 2, 128, 128]  – (tree_count, mean_height) after target_transform
# mask : bool    [128, 128]      – True = supervise this pixel (see valid_mask_mode)
```

---

## Quick data checks (commands and what they do)

### Inspect parquet schema

Overrides the `compute_stats` service entrypoint so only inspection runs:

```bash
docker compose run --rm compute_stats \
  python scripts/compute_stats.py --config configs/unet_resnet50.yaml --inspect
```

**Expected outcome:** printed column list and head of the dataframe; use this to fix `patch_idx_column` / `split_column` if you see `KeyError`.

### Full pipeline check

```bash
docker compose run --rm check
```

**What `scripts/pipeline_check.py` does (high level):** validates parquet columns and split counts; builds `BiomassDataset` for train/val/test; checks shapes and NaN behaviour on sample patches; runs a **`DataLoader`** batch with augmentations; builds the UNet with **`encoder_weights: None`** to avoid downloading weights; forward pass; **`build_loss`** + backward; **`RunningMetrics`** update; audits Zarr NaN rates on a fixed patch index; verifies channel count **`8 + 40 + (1 if use_species else 0)`**.

### Jupyter notebook

```bash
docker compose up notebook
# → http://localhost:8888
```

Use this for exploratory plots (histograms, patch thumbnails, mask overlays). The compose service mounts the same volumes as training so paths match.

---

## Summary checklist before training

1. **`data.root`** and **`data.split_file`** point at the mounted Zarr tree.  
2. **`patch_idx_column`** matches the parquet (here: **`zarr_idx`**).  
3. **`norm_stats.json`** exists and was built with the same **`use_species`** and season lists as training.  
4. **`docker compose run --rm check`** passes on the machine that sees the data.

This closes the loop between **DATASET.md**, **YAML**, and the **Python** entrypoints that consume the data.

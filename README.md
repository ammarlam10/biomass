# Biomass Regression Pipeline

Pixel-wise regression of **tree count** and **mean height** from multi-seasonal
Sentinel-1, Sentinel-2, and species data, using PyTorch + Docker.

## Architecture

```
UNet (ResNet50 encoder) → [B, 49, 128, 128] → [B, 2, 128, 128]
  ↑
  S1 (4 seasons × 2 bands = 8 ch)
  S2 (4 seasons × 10 bands = 40 ch)
  tree_species (1 ch, optional)
```

Sparse tree pixels are handled by a **masked regression loss** – only valid
(non-NaN) pixels contribute to training.

## Quick Start

### 1. Build the image

```bash
cd /work/ammar/sslrp/biomass
docker compose build
```

### 2. Inspect the parquet split file (if column names are unknown)

```bash
docker compose run --rm compute_stats \
    python scripts/compute_stats.py --config configs/unet_resnet50.yaml --inspect
```

Update `configs/unet_resnet50.yaml` and `configs/xgboost.yaml` (`data.split_column`, `data.patch_idx_column`) if
the defaults (`split`, `zarr_idx`) don't match.

### 3. Compute per-channel normalisation stats (run once)

```bash
docker compose run --rm compute_stats
# Stats saved to artifacts/norm_stats.json
```

### 4. Smoke test (sanity check, ~30 s)

```bash
docker compose run --rm train python scripts/smoke_test.py
```

### 5. Train

```bash
docker compose run --rm train
# TensorBoard logs → artifacts/runs/
# Checkpoints     → artifacts/checkpoints/best.pt, latest.pt
# CSV metrics     → artifacts/metrics.csv
```

### 6. Evaluate

```bash
docker compose run --rm eval        # test split
docker compose run --rm eval_val    # val split
# Metrics saved to artifacts/checkpoints/{test,val}_metrics.json
```

### 7. EDA notebook

```bash
docker compose up notebook
# Open http://localhost:8888  (no token)
```

## Project Structure

```
biomass/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── configs/
│   ├── unet_resnet50.yaml    ← UNet training / eval / compute_stats
│   └── xgboost.yaml          ← XGBoost baseline (train_xgboost, evaluate_xgboost)
├── src/
│   ├── data/
│   │   ├── dataset.py        ← BiomassDataset (zarr + parquet)
│   │   └── transforms.py     ← flip / rotate augmentations
│   ├── models/
│   │   ├── factory.py        ← model registry (@register_model)
│   │   ├── unet_resnet50.py  ← baseline model
│   │   ├── vit_segmentation.py  ← Stage 1 stub
│   │   └── prithvi_adapter.py   ← Stage 2 stub
│   ├── losses/
│   │   └── masked_regression.py  ← MaskedRegressionLoss
│   └── training/
│       ├── metrics.py        ← masked RMSE / MAE / R²
│       └── trainer.py        ← training loop, checkpointing, TB logging
├── scripts/
│   ├── compute_stats.py      ← normalisation stats (run before training)
│   ├── train.py              ← main training entrypoint
│   ├── evaluate.py           ← evaluation on any split
│   ├── smoke_test.py         ← pipeline connectivity check
│   ├── train_xgboost.py      ← XGBoost baseline
│   └── evaluate_xgboost.py   ← XGBoost evaluation
├── notebooks/
│   └── explore_data.ipynb    ← EDA (run inside notebook service)
└── artifacts/                ← created at runtime (gitignored)
    ├── norm_stats.json
    ├── checkpoints/
    │   ├── best.pt
    │   └── latest.pt
    ├── runs/                 ← TensorBoard
    └── metrics.csv
```

## Config Reference

Key settings in `configs/unet_resnet50.yaml` (UNet) / `configs/xgboost.yaml` (XGBoost; shares the same `data` block for fair comparison):

| Key | Default | Description |
|---|---|---|
| `data.use_species` | `true` | Include tree_species channel |
| `data.valid_mask_mode` | `notnull` | `notnull` or `positive` (stricter) |
| `data.target_transform.tree_count` | `log1p` | `log1p` or `none` |
| `model.name` | `unet_resnet50` | Model registry key |
| `training.loss` | `masked_mse` | `masked_mse` \| `masked_smooth_l1` \| `masked_mae` |
| `training.epochs` | `100` | Number of training epochs |
| `training.batch_size` | `16` | Batch size |
| `training.amp` | `true` | Mixed precision (CUDA only) |

## Adding a New Model (e.g. ViT)

1. Create `src/models/my_model.py`
2. Decorate a builder function:
   ```python
   from src.models.factory import register_model

   @register_model("my_model")
   def build_my_model(cfg, num_input_channels):
       return MyModel(in_channels=num_input_channels, ...)
   ```
3. Add `from src.models import my_model` to `src/models/__init__.py`
4. Set `model.name: my_model` in `configs/unet_resnet50.yaml`
5. No changes needed to the trainer, loss, or metrics.

## Roadmap

| Stage | Model | Status |
|---|---|---|
| Baseline | UNet + ResNet50 | Implemented |
| Stage 1 | ViT segmentation | Stub (`vit_segmentation.py`) |
| Stage 2 | Prithvi foundation model adapter | Stub (`prithvi_adapter.py`) |
| Stage 3 | XGBoost pixel baseline | Implemented (`train_xgboost.py`, `evaluate_xgboost.py`) |

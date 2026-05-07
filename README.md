# TreeUQ: Uncertainty-Aware Multimodal Regression for Tree Density and Height

Pixel-wise regression of **tree count** and **mean height** from multi-seasonal Sentinel-1, Sentinel-2, and tree species data over Bavaria, Germany. The pipeline benchmarks three deep learning architectures — **UNet-ResNet50**, **SegFormer-B3**, and **Clay** (a geospatial foundation model) — against an **XGBoost** pixel baseline. A masked regression loss handles sparse ground-truth labels by only supervising valid (non-NaN) pixels.

## Requirements

- [Docker](https://docs.docker.com/get-docker/) with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- NVIDIA GPU (16 GB+ VRAM recommended)

## Setup

### 1. Configure paths

Open `docker-compose.yml` and update the volume block to point at your dataset and working directory:

```yaml
x-volumes: &volumes
  volumes:
    - /path/to/dataset.zarr:/data/ammar/4g.zarr:ro
    - /path/to/this/repo:/workspace
    - /path/to/this/repo/artifacts:/workspace/artifacts
```

### 2. Build the image

```bash
docker compose build
```

### 3. Compute per-channel normalisation statistics (run once)

```bash
docker compose run --rm compute_stats
# Output: artifacts/norm_stats.json
```

## Training

```bash
# UNet-ResNet50
docker compose run --rm train

# SegFormer-B3
docker compose run --rm train_segformer_b3

# Clay (geospatial foundation model)
docker compose run --rm train_clay

# XGBoost pixel baseline
docker compose run --rm train_xgboost
```

Outputs per run:
- Best checkpoint → `artifacts/<model>/checkpoints/best.pt`
- TensorBoard logs → `artifacts/<model>/runs/`
- Per-epoch metrics → `artifacts/<model>/metrics.csv`

## Evaluation

```bash
docker compose run --rm eval               # UNet-ResNet50
docker compose run --rm eval_segformer_b3  # SegFormer-B3
docker compose run --rm eval_clay          # Clay
docker compose run --rm eval_xgboost       # XGBoost
```

Metrics (masked RMSE, MAE, R²) are written to `artifacts/<model>/checkpoints/test_metrics.json`.

## Project Structure

```
biomass/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── configs/               # per-model YAML training configs
├── src/
│   ├── data/              # BiomassDataset (Zarr + Parquet)
│   ├── models/            # UNet, SegFormer, Clay, XGBoost
│   ├── losses/            # masked regression loss
│   └── training/          # training loop and metrics
└── scripts/               # train.py, evaluate.py, train_xgboost.py, evaluate_xgboost.py, compute_stats.py
```

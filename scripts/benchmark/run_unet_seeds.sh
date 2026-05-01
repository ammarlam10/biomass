#!/usr/bin/env bash
# =============================================================================
# run_unet_seeds.sh – UNet + ResNet50, 3 seeds, 3-GPU DDP, 60 epochs each
#
# GPUs (host): 0, 1, 2   (docker --gpus "device=0,1,2")
# Inside container:       CUDA_VISIBLE_DEVICES=0,1,2
# Effective batch size:   21 per GPU × 3 = 63
#
# Usage:
#   bash scripts/benchmark/run_unet_seeds.sh
#
# Environment overrides (export before running):
#   IMAGE          Docker image name  (default: biomass:latest)
#   REPO_ROOT      Host path to repo  (default: /data/ammar/biomass)
#   DATA_ZARR      Host path to zarr  (default: /data/ammar/4g.zarr)
#   ARTIFACTS_HOST Host artifact root (default: /data/ammar/biomass/artifacts)
#   MASTER_PORT    torchrun port      (default: 29510)
#   NUM_WORKERS    DataLoader workers per rank (default: 4)
#   SHM_SIZE       --shm-size for docker      (default: 16g)
# =============================================================================
set -euo pipefail

IMAGE="${IMAGE:-biomass:latest}"
REPO_ROOT="${REPO_ROOT:-/data/ammar/biomass}"
DATA_ZARR="${DATA_ZARR:-/data/ammar/4g.zarr}"
ARTIFACTS_HOST="${ARTIFACTS_HOST:-/data/ammar/biomass/artifacts}"
MASTER_PORT="${MASTER_PORT:-29510}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SHM_SIZE="${SHM_SIZE:-16g}"

SEEDS=(42 123 456)
CONFIG="configs/benchmark/unet_resnet50.yaml"

SEP="──────────────────────────────────────────────────────────────"

echo "$SEP"
echo "  UNet benchmark  |  seeds: ${SEEDS[*]}"
echo "  Host GPUs: 0,1,2  |  nproc=3  |  60 epochs / seed"
echo "  Image:     $IMAGE"
echo "  Artifacts: $ARTIFACTS_HOST/benchmark/unet/"
echo "$SEP"

for SEED in "${SEEDS[@]}"; do
    RUN_DIR="/workspace/artifacts/benchmark/unet/seed_${SEED}"

    echo ""
    echo ">>> seed=${SEED}  run_dir=${RUN_DIR}"

    docker run --rm \
        --gpus '"device=0,1,2"' \
        --shm-size "${SHM_SIZE}" \
        -v "${REPO_ROOT}:/workspace" \
        -v "${DATA_ZARR}:/data/ammar/4g.zarr:ro" \
        -v "${ARTIFACTS_HOST}:/workspace/artifacts" \
        -w /workspace \
        -e CUDA_VISIBLE_DEVICES=0,1,2 \
        -e MASTER_PORT="${MASTER_PORT}" \
        "${IMAGE}" \
        torchrun \
            --standalone \
            --nproc_per_node=3 \
            --master_port="${MASTER_PORT}" \
            scripts/train.py \
            --config "${CONFIG}" \
            --ddp \
            --seed "${SEED}" \
            --run-dir "${RUN_DIR}" \
            --num-workers "${NUM_WORKERS}"

    echo "  seed=${SEED} DONE  →  ${ARTIFACTS_HOST}/benchmark/unet/seed_${SEED}/"
done

echo ""
echo "$SEP"
echo "  All UNet seeds complete."
echo "  Test metrics:"
for SEED in "${SEEDS[@]}"; do
    echo "    seed=${SEED}  →  ${ARTIFACTS_HOST}/benchmark/unet/seed_${SEED}/checkpoints/test_metrics.json"
done
echo "$SEP"

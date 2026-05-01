#!/usr/bin/env bash
# =============================================================================
# run_clay_seeds.sh – Clay v1.5, 3 seeds, 2-GPU DDP, 30e frozen + 30e unfrozen
#
# GPUs (host): 4, 5   (docker --gpus "device=4,5")
# Inside container:   CUDA_VISIBLE_DEVICES=0,1
#
# Per-seed flow:
#   Phase 1 – freeze encoder, lr=1e-4, 30 epochs
#             → artifacts/benchmark/clay/seed_S/p1/checkpoints/best.pt
#   Phase 2 – unfreeze encoder, lr=5e-5, 30 epochs, resume from Phase 1 best
#             → artifacts/benchmark/clay/seed_S/p2/checkpoints/test_metrics.json
#
# Do NOT run this script concurrently with run_xgboost_seeds.sh – both use
# host GPU 4.
#
# Usage:
#   bash scripts/benchmark/run_clay_seeds.sh
#
# Environment overrides (export before running):
#   IMAGE            Docker image name  (default: biomass:latest)
#   REPO_ROOT        Host path to repo  (default: /data/ammar/biomass)
#   DATA_ZARR        Host path to zarr  (default: /data/ammar/4g.zarr)
#   ARTIFACTS_HOST   Host artifact root (default: /data/ammar/biomass/artifacts)
#   MASTER_PORT      torchrun port      (default: 29520)
#   NUM_WORKERS      DataLoader workers per rank (default: 4)
#   SHM_SIZE         --shm-size         (default: 16g)
# =============================================================================
set -euo pipefail

IMAGE="${IMAGE:-biomass:latest}"
REPO_ROOT="${REPO_ROOT:-/data/ammar/biomass}"
DATA_ZARR="${DATA_ZARR:-/data/ammar/4g.zarr}"
ARTIFACTS_HOST="${ARTIFACTS_HOST:-/data/ammar/biomass/artifacts}"
MASTER_PORT="${MASTER_PORT:-29520}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SHM_SIZE="${SHM_SIZE:-16g}"

SEEDS=(42 123 456)
CONFIG="configs/benchmark/clay.yaml"

SEP="──────────────────────────────────────────────────────────────"

echo "$SEP"
echo "  Clay benchmark  |  seeds: ${SEEDS[*]}"
echo "  Host GPUs: 4,5  |  nproc=2  |  30e frozen + 30e unfrozen / seed"
echo "  Image:     $IMAGE"
echo "  Artifacts: $ARTIFACTS_HOST/benchmark/clay/"
echo "  WARNING: do NOT run concurrently with run_xgboost_seeds.sh (both use host GPU 4)"
echo "$SEP"

for SEED in "${SEEDS[@]}"; do
    P1_DIR="/workspace/artifacts/benchmark/clay/seed_${SEED}/p1"
    P2_DIR="/workspace/artifacts/benchmark/clay/seed_${SEED}/p2"
    P1_CKPT="${P1_DIR}/checkpoints/best.pt"

    echo ""
    echo ">>> seed=${SEED}  Phase 1 (freeze encoder, lr=1e-4, 30 epochs)"

    docker run --rm \
        --gpus '"device=4,5"' \
        --shm-size "${SHM_SIZE}" \
        -v "${REPO_ROOT}:/workspace" \
        -v "${DATA_ZARR}:/data/ammar/4g.zarr:ro" \
        -v "${ARTIFACTS_HOST}:/workspace/artifacts" \
        -w /workspace \
        -e CUDA_VISIBLE_DEVICES=0,1 \
        -e MASTER_PORT="${MASTER_PORT}" \
        "${IMAGE}" \
        torchrun \
            --standalone \
            --nproc_per_node=2 \
            --master_port="${MASTER_PORT}" \
            scripts/train.py \
            --config "${CONFIG}" \
            --ddp \
            --seed "${SEED}" \
            --freeze-clay-encoder \
            --epochs 30 \
            --lr 1e-4 \
            --run-dir "${P1_DIR}" \
            --num-workers "${NUM_WORKERS}"

    echo "  seed=${SEED} Phase 1 DONE  →  ${ARTIFACTS_HOST}/benchmark/clay/seed_${SEED}/p1/"
    echo ""
    echo ">>> seed=${SEED}  Phase 2 (unfreeze encoder, lr=5e-5, 30 epochs, resume Phase 1 best)"

    docker run --rm \
        --gpus '"device=4,5"' \
        --shm-size "${SHM_SIZE}" \
        -v "${REPO_ROOT}:/workspace" \
        -v "${DATA_ZARR}:/data/ammar/4g.zarr:ro" \
        -v "${ARTIFACTS_HOST}:/workspace/artifacts" \
        -w /workspace \
        -e CUDA_VISIBLE_DEVICES=0,1 \
        -e MASTER_PORT="${MASTER_PORT}" \
        "${IMAGE}" \
        torchrun \
            --standalone \
            --nproc_per_node=2 \
            --master_port="${MASTER_PORT}" \
            scripts/train.py \
            --config "${CONFIG}" \
            --ddp \
            --seed "${SEED}" \
            --no-freeze-clay-encoder \
            --epochs 30 \
            --lr 5e-5 \
            --resume "${P1_CKPT}" \
            --run-dir "${P2_DIR}" \
            --num-workers "${NUM_WORKERS}"

    echo "  seed=${SEED} Phase 2 DONE  →  ${ARTIFACTS_HOST}/benchmark/clay/seed_${SEED}/p2/"
done

echo ""
echo "$SEP"
echo "  All Clay seeds complete."
echo "  Test metrics (Phase 2 best checkpoint):"
for SEED in "${SEEDS[@]}"; do
    echo "    seed=${SEED}  →  ${ARTIFACTS_HOST}/benchmark/clay/seed_${SEED}/p2/checkpoints/test_metrics.json"
done
echo "$SEP"

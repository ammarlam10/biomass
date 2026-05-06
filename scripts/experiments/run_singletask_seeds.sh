#!/usr/bin/env bash
# =============================================================================
# run_singletask_seeds.sh – Experiment A1: Single-Task Decoupling
#
# Trains single-target variants (mean_height only, tree_count only) for all
# three DL models (SegFormer-B3, U-Net ResNet50, Clay v1.5) over 3 seeds each.
#
# GPU assignments (same as benchmark):
#   SegFormer-B3 : GPU 7  (single process)
#   U-Net        : GPUs 0/1/2 (3-GPU DDP via torchrun --nproc_per_node=3)
#   Clay         : GPUs 2/3   (2-GPU DDP via torchrun --nproc_per_node=2)
#
# Environment overrides (export before running):
#   IMAGE          Docker image name  (default: biomass:latest)
#   REPO_ROOT      Host path to repo  (default: /data/ammar/biomass)
#   DATA_ZARR      Host path to zarr  (default: /data/ammar/4g.zarr)
#   ARTIFACTS_HOST Host artifact root (default: /data/ammar/biomass/artifacts)
#   NUM_WORKERS    DataLoader workers (default: 8)
#   SHM_SIZE       --shm-size         (default: 8g)
#
# Usage:
#   bash scripts/experiments/run_singletask_seeds.sh
#
# To run only a specific model/target combo, comment out the other sections.
# =============================================================================
set -euo pipefail

IMAGE="${IMAGE:-biomass:latest}"
REPO_ROOT="${REPO_ROOT:-/data/ammar/biomass}"
DATA_ZARR="${DATA_ZARR:-/data/ammar/4g.zarr}"
ARTIFACTS_HOST="${ARTIFACTS_HOST:-/data/ammar/biomass/artifacts}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SHM_SIZE="${SHM_SIZE:-8g}"

SEEDS=(42 123 456)
SEP="──────────────────────────────────────────────────────────────"

# Helper: run a single-GPU container
run_single_gpu() {
    local GPU_HOST="$1"
    local CONFIG="$2"
    local SEED="$3"
    local RUN_DIR="$4"

    docker run --rm \
        --gpus "\"device=${GPU_HOST}\"" \
        --shm-size "${SHM_SIZE}" \
        -v "${REPO_ROOT}:/workspace" \
        -v "${DATA_ZARR}:/data/ammar/4g.zarr:ro" \
        -v "${ARTIFACTS_HOST}:/workspace/artifacts" \
        -w /workspace \
        -e CUDA_VISIBLE_DEVICES=0 \
        "${IMAGE}" \
        python scripts/train.py \
            --config "${CONFIG}" \
            --seed "${SEED}" \
            --run-dir "${RUN_DIR}" \
            --num-workers "${NUM_WORKERS}"
}

# Helper: run a DDP container (torchrun)
run_ddp() {
    local GPUS_HOST="$1"     # e.g. "0,1,2"
    local NPROC="$2"
    local CONFIG="$3"
    local SEED="$4"
    local RUN_DIR="$5"
    local EXTRA_ARGS="${6:-}"

    docker run --rm \
        --gpus "\"device=${GPUS_HOST}\"" \
        --shm-size "${SHM_SIZE}" \
        -v "${REPO_ROOT}:/workspace" \
        -v "${DATA_ZARR}:/data/ammar/4g.zarr:ro" \
        -v "${ARTIFACTS_HOST}:/workspace/artifacts" \
        -w /workspace \
        "${IMAGE}" \
        torchrun --nproc_per_node="${NPROC}" scripts/train.py \
            --config "${CONFIG}" \
            --ddp \
            --seed "${SEED}" \
            --run-dir "${RUN_DIR}" \
            --num-workers "${NUM_WORKERS}" \
            ${EXTRA_ARGS}
}

# =============================================================================
# 1. SegFormer-B3 – mean_height only  (GPU 7, single process)
# =============================================================================
echo "$SEP"
echo "  A1 SegFormer-B3 / mean_height  |  seeds: ${SEEDS[*]}"
echo "$SEP"

for SEED in "${SEEDS[@]}"; do
    RUN_DIR="/workspace/artifacts/experiments/singletask_height_segformer_b3/seed_${SEED}"
    echo ">>> SegFormer-B3 height  seed=${SEED}  run_dir=${RUN_DIR}"
    run_single_gpu 7 "configs/experiments/singletask_height_segformer_b3.yaml" "${SEED}" "${RUN_DIR}"
    echo "  seed=${SEED} DONE"
done

# =============================================================================
# 2. SegFormer-B3 – tree_count only  (GPU 7, single process)
# =============================================================================
echo "$SEP"
echo "  A1 SegFormer-B3 / tree_count  |  seeds: ${SEEDS[*]}"
echo "$SEP"

for SEED in "${SEEDS[@]}"; do
    RUN_DIR="/workspace/artifacts/experiments/singletask_count_segformer_b3/seed_${SEED}"
    echo ">>> SegFormer-B3 count  seed=${SEED}  run_dir=${RUN_DIR}"
    run_single_gpu 7 "configs/experiments/singletask_count_segformer_b3.yaml" "${SEED}" "${RUN_DIR}"
    echo "  seed=${SEED} DONE"
done

# =============================================================================
# 3. U-Net ResNet50 – mean_height only  (GPUs 0/1/2, 3-GPU DDP)
# =============================================================================
echo "$SEP"
echo "  A1 U-Net / mean_height  |  seeds: ${SEEDS[*]}"
echo "$SEP"

for SEED in "${SEEDS[@]}"; do
    RUN_DIR="/workspace/artifacts/experiments/singletask_height_unet/seed_${SEED}"
    echo ">>> U-Net height  seed=${SEED}  run_dir=${RUN_DIR}"
    run_ddp "0,1,2" 3 "configs/experiments/singletask_height_unet.yaml" "${SEED}" "${RUN_DIR}"
    echo "  seed=${SEED} DONE"
done

# =============================================================================
# 4. U-Net ResNet50 – tree_count only  (GPUs 0/1/2, 3-GPU DDP)
# =============================================================================
echo "$SEP"
echo "  A1 U-Net / tree_count  |  seeds: ${SEEDS[*]}"
echo "$SEP"

for SEED in "${SEEDS[@]}"; do
    RUN_DIR="/workspace/artifacts/experiments/singletask_count_unet/seed_${SEED}"
    echo ">>> U-Net count  seed=${SEED}  run_dir=${RUN_DIR}"
    run_ddp "0,1,2" 3 "configs/experiments/singletask_count_unet.yaml" "${SEED}" "${RUN_DIR}"
    echo "  seed=${SEED} DONE"
done

# =============================================================================
# 5. Clay v1.5 – mean_height only  (GPUs 2/3, 2-GPU DDP, two-phase)
# =============================================================================
echo "$SEP"
echo "  A1 Clay / mean_height  |  seeds: ${SEEDS[*]}"
echo "$SEP"

for SEED in "${SEEDS[@]}"; do
    P1_DIR="/workspace/artifacts/experiments/singletask_height_clay/seed_${SEED}/p1"
    P2_DIR="/workspace/artifacts/experiments/singletask_height_clay/seed_${SEED}/p2"
    P1_BEST="${ARTIFACTS_HOST}/experiments/singletask_height_clay/seed_${SEED}/p1/checkpoints/best.pt"

    echo ">>> Clay height  seed=${SEED}  Phase 1 → ${P1_DIR}"
    run_ddp "2,3" 2 \
        "configs/experiments/singletask_height_clay.yaml" \
        "${SEED}" "${P1_DIR}" \
        "--freeze-clay-encoder --lr 1e-4 --epochs 30"

    echo ">>> Clay height  seed=${SEED}  Phase 2 → ${P2_DIR}"
    run_ddp "2,3" 2 \
        "configs/experiments/singletask_height_clay.yaml" \
        "${SEED}" "${P2_DIR}" \
        "--no-freeze-clay-encoder --lr 5e-5 --epochs 30 --resume /workspace/artifacts/experiments/singletask_height_clay/seed_${SEED}/p1/checkpoints/best.pt"

    echo "  seed=${SEED} DONE → ${ARTIFACTS_HOST}/experiments/singletask_height_clay/seed_${SEED}/p2/checkpoints/test_metrics.json"
done

# =============================================================================
# 6. Clay v1.5 – tree_count only  (GPUs 2/3, 2-GPU DDP, two-phase)
# =============================================================================
echo "$SEP"
echo "  A1 Clay / tree_count  |  seeds: ${SEEDS[*]}"
echo "$SEP"

for SEED in "${SEEDS[@]}"; do
    P1_DIR="/workspace/artifacts/experiments/singletask_count_clay/seed_${SEED}/p1"
    P2_DIR="/workspace/artifacts/experiments/singletask_count_clay/seed_${SEED}/p2"

    echo ">>> Clay count  seed=${SEED}  Phase 1 → ${P1_DIR}"
    run_ddp "2,3" 2 \
        "configs/experiments/singletask_count_clay.yaml" \
        "${SEED}" "${P1_DIR}" \
        "--freeze-clay-encoder --lr 1e-4 --epochs 30"

    echo ">>> Clay count  seed=${SEED}  Phase 2 → ${P2_DIR}"
    run_ddp "2,3" 2 \
        "configs/experiments/singletask_count_clay.yaml" \
        "${SEED}" "${P2_DIR}" \
        "--no-freeze-clay-encoder --lr 5e-5 --epochs 30 --resume /workspace/artifacts/experiments/singletask_count_clay/seed_${SEED}/p1/checkpoints/best.pt"

    echo "  seed=${SEED} DONE → ${ARTIFACTS_HOST}/experiments/singletask_count_clay/seed_${SEED}/p2/checkpoints/test_metrics.json"
done

# =============================================================================
echo ""
echo "$SEP"
echo "  All A1 single-task runs complete."
echo ""
echo "  Key results (test_metrics.json):"
for MODEL in singletask_height_segformer_b3 singletask_count_segformer_b3 \
             singletask_height_unet singletask_count_unet; do
    for SEED in "${SEEDS[@]}"; do
        echo "    ${MODEL}/seed_${SEED}/checkpoints/test_metrics.json"
    done
done
for MODEL in singletask_height_clay singletask_count_clay; do
    for SEED in "${SEEDS[@]}"; do
        echo "    ${MODEL}/seed_${SEED}/p2/checkpoints/test_metrics.json"
    done
done
echo "  All paths under: ${ARTIFACTS_HOST}/experiments/"
echo "$SEP"

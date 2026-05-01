#!/usr/bin/env bash
# =============================================================================
# run_xgboost_seeds.sh – XGBoost baseline, 3 seeds, single GPU
#
# GPU (host): 4   (docker --gpus "device=4")
# Inside container: CUDA_VISIBLE_DEVICES=0
#
# Each seed writes its own artifact directory:
#   artifacts/benchmark/xgb/seed_S/
#     xgb_tree_count.json
#     xgb_mean_height.json
#     xgb_run_info.json
#     test_metrics_xgboost.json
#
# Do NOT run this script concurrently with run_clay_seeds.sh – both use
# host GPU 4.
#
# Usage:
#   bash scripts/benchmark/run_xgboost_seeds.sh
#
# Environment overrides (export before running):
#   IMAGE          Docker image name  (default: biomass:latest)
#   REPO_ROOT      Host path to repo  (default: /data/ammar/biomass)
#   DATA_ZARR      Host path to zarr  (default: /data/ammar/4g.zarr)
#   ARTIFACTS_HOST Host artifact root (default: /data/ammar/biomass/artifacts)
#   NUM_WORKERS    DataLoader workers (default: 4)
#   SHM_SIZE       --shm-size         (default: 8g)
# =============================================================================
set -euo pipefail

IMAGE="${IMAGE:-biomass:latest}"
REPO_ROOT="${REPO_ROOT:-/data/ammar/biomass}"
DATA_ZARR="${DATA_ZARR:-/data/ammar/4g.zarr}"
ARTIFACTS_HOST="${ARTIFACTS_HOST:-/data/ammar/biomass/artifacts}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SHM_SIZE="${SHM_SIZE:-8g}"

SEEDS=(42 123 456)
CONFIG="configs/benchmark/xgboost.yaml"

SEP="──────────────────────────────────────────────────────────────"

echo "$SEP"
echo "  XGBoost benchmark  |  seeds: ${SEEDS[*]}"
echo "  Host GPU: 4  |  single process"
echo "  Image:     $IMAGE"
echo "  Artifacts: $ARTIFACTS_HOST/benchmark/xgb/"
echo "  WARNING: do NOT run concurrently with run_clay_seeds.sh (both use host GPU 4)"
echo "$SEP"

for SEED in "${SEEDS[@]}"; do
    SAVE_DIR="/workspace/artifacts/benchmark/xgb/seed_${SEED}"

    echo ""
    echo ">>> seed=${SEED}  save_dir=${SAVE_DIR}"

    docker run --rm \
        --gpus '"device=4"' \
        --shm-size "${SHM_SIZE}" \
        -v "${REPO_ROOT}:/workspace" \
        -v "${DATA_ZARR}:/data/ammar/4g.zarr:ro" \
        -v "${ARTIFACTS_HOST}:/workspace/artifacts" \
        -w /workspace \
        -e CUDA_VISIBLE_DEVICES=0 \
        "${IMAGE}" \
        python scripts/train_xgboost.py \
            --config "${CONFIG}" \
            --seed "${SEED}" \
            --save-dir "${SAVE_DIR}" \
            --num_workers "${NUM_WORKERS}"

    echo "  seed=${SEED} DONE  →  ${ARTIFACTS_HOST}/benchmark/xgb/seed_${SEED}/"
done

echo ""
echo "$SEP"
echo "  All XGBoost seeds complete."
echo "  Test metrics:"
for SEED in "${SEEDS[@]}"; do
    echo "    seed=${SEED}  →  ${ARTIFACTS_HOST}/benchmark/xgb/seed_${SEED}/test_metrics_xgboost.json"
done
echo "$SEP"

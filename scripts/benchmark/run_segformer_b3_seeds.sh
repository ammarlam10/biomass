#!/usr/bin/env bash
# =============================================================================
# run_segformer_b3_seeds.sh – SegFormer-B3, 3 seeds, single GPU, 60 epochs each
#
# GPU (host): 7   (docker --gpus "device=7")
# Inside container: CUDA_VISIBLE_DEVICES=0
#
# Usage:
#   bash scripts/benchmark/run_segformer_b3_seeds.sh
#
# Environment overrides (export before running):
#   IMAGE          Docker image name  (default: biomass:latest)
#   REPO_ROOT      Host path to repo  (default: /data/ammar/biomass)
#   DATA_ZARR      Host path to zarr  (default: /data/ammar/4g.zarr)
#   ARTIFACTS_HOST Host artifact root (default: /data/ammar/biomass/artifacts)
#   NUM_WORKERS    DataLoader workers (default: 8)
#   SHM_SIZE       --shm-size         (default: 8g)
# =============================================================================
set -euo pipefail

IMAGE="${IMAGE:-biomass:latest}"
REPO_ROOT="${REPO_ROOT:-/data/ammar/biomass}"
DATA_ZARR="${DATA_ZARR:-/data/ammar/4g.zarr}"
ARTIFACTS_HOST="${ARTIFACTS_HOST:-/data/ammar/biomass/artifacts}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SHM_SIZE="${SHM_SIZE:-8g}"

SEEDS=(42 123 456)
CONFIG="configs/benchmark/segformer_b3.yaml"

SEP="──────────────────────────────────────────────────────────────"

echo "$SEP"
echo "  SegFormer-B3 benchmark  |  seeds: ${SEEDS[*]}"
echo "  Host GPU: 7  |  single process  |  60 epochs / seed"
echo "  Image:     $IMAGE"
echo "  Artifacts: $ARTIFACTS_HOST/benchmark/segformer_b3/"
echo "$SEP"

for SEED in "${SEEDS[@]}"; do
    RUN_DIR="/workspace/artifacts/benchmark/segformer_b3/seed_${SEED}"

    echo ""
    echo ">>> seed=${SEED}  run_dir=${RUN_DIR}"

    docker run --rm \
        --gpus '"device=7"' \
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

    echo "  seed=${SEED} DONE  →  ${ARTIFACTS_HOST}/benchmark/segformer_b3/seed_${SEED}/"
done

echo ""
echo "$SEP"
echo "  All SegFormer-B3 seeds complete."
echo "  Test metrics:"
for SEED in "${SEEDS[@]}"; do
    echo "    seed=${SEED}  →  ${ARTIFACTS_HOST}/benchmark/segformer_b3/seed_${SEED}/checkpoints/test_metrics.json"
done
echo "$SEP"

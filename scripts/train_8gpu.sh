#!/usr/bin/env bash
# Single-node, 8-GPU RoboCasa365 post-training launcher (openpi-omnisf).
#
# Usage:
#   scripts/train_8gpu.sh [config_name] [exp_name] [extra train.py args...]
#
# Examples:
#   scripts/train_8gpu.sh
#       -> robocasa365_target_composite_seen with a timestamped exp name
#   scripts/train_8gpu.sh robocasa365_target_atomic atomic_run0 --overwrite
#   NGPU=8 ROBOCASA365_NUM_TRAIN_STEPS=30000 scripts/train_8gpu.sh robocasa365_target_composite_unseen unseen_run0
#
# Environment overrides live in scripts/env.sh (paths, checkpoint, batch size...).
# Default recipe: micro-batch 256 x gradient accumulation 2 => effective 512,
# 8-way FSDP, omnisf spatial-forcing alignment enabled.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/env.sh
source "${SCRIPT_DIR}/env.sh"

DEFAULT_CONFIG="robocasa365_target_composite_seen"

CONFIG="${1:-${DEFAULT_CONFIG}}"
if [[ $# -ge 1 ]]; then shift; fi

EXP_NAME="${1:-${CONFIG}_$(date +%Y%m%d_%H%M%S)}"
if [[ $# -ge 1 ]]; then shift; fi

EXTRA_ARGS=("$@")

NGPU="${NGPU:-8}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-23456}"

export JAX_NODE_COUNT=1
export JAX_COORDINATOR_ADDR="${JAX_COORDINATOR_ADDR:-${MASTER_ADDR}}"
export JAX_COORDINATOR_PORT="${JAX_COORDINATOR_PORT:-$((MASTER_PORT + 1))}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $((NGPU - 1)))}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-${CODE_ROOT}/.cache/jax/${CONFIG}/1x${NGPU}x1}"
export OPENPI_SHOW_TQDM="${OPENPI_SHOW_TQDM:-0}"

[[ -x "${OPENPI_PYTHON}" ]] || {
    echo "ERROR: openpi python not found/executable: ${OPENPI_PYTHON}" >&2
    exit 1
}
[[ -f "${CODE_ROOT}/train.py" ]] || {
    echo "ERROR: entry point not found: ${CODE_ROOT}/train.py" >&2
    exit 1
}

if [[ "${CONFIG}" == robocasa365_target_* ]]; then
    DATASET_NAME="robocasa365-target-$(echo "${CONFIG#robocasa365_target_}" | tr '_' '-')"
    DATASET_DIR="${ROBOCASA365_DATA_ROOT}/${DATASET_NAME}"
    if [[ ! -f "${DATASET_DIR}/meta/info.json" ]]; then
        echo "ERROR: dataset not found: ${DATASET_DIR}/meta/info.json" >&2
        echo "       Run scripts/convert_v30_to_v21.sh first." >&2
        exit 1
    fi
fi

LOG_DIR="${CODE_ROOT}/logs"
mkdir -p "${LOG_DIR}" "${JAX_COMPILATION_CACHE_DIR}" "${ROBOCASA365_ASSETS_DIR}" "${ROBOCASA365_CHECKPOINT_DIR}"
LOG_STEM="${CONFIG}_$(date +%Y%m%d_%H%M%S)"
LOG_OUT="${LOG_DIR}/${LOG_STEM}.out"
LOG_ERR="${LOG_DIR}/${LOG_STEM}.err"

echo "Job started at $(date)"
echo "CODE_ROOT=${CODE_ROOT}"
echo "OPENPI_ROOT=${OPENPI_ROOT}"
echo "OPENPI_PYTHON=${OPENPI_PYTHON}"
echo "CONFIG=${CONFIG}"
echo "EXP_NAME=${EXP_NAME}"
echo "NGPU=${NGPU}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "ROBOCASA365_DATA_ROOT=${ROBOCASA365_DATA_ROOT}"
echo "ROBOCASA365_ASSETS_DIR=${ROBOCASA365_ASSETS_DIR}"
echo "ROBOCASA365_CHECKPOINT_DIR=${ROBOCASA365_CHECKPOINT_DIR}"
echo "ROBOCASA365_PRETRAIN=${ROBOCASA365_PRETRAIN}"
echo "EXTRA_ARGS=${EXTRA_ARGS[*]:-}"
echo "LOG_OUT=${LOG_OUT}"
echo "LOG_ERR=${LOG_ERR}"

command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L || true

cd "${CODE_ROOT}"

exec > >(tee -a "${LOG_OUT}") 2> >(tee -a "${LOG_ERR}" >&2)

exec "${OPENPI_PYTHON}" -m torch.distributed.run \
    --nnodes=1 \
    --nproc-per-node="${NGPU}" \
    --node-rank=0 \
    --master-addr="${MASTER_ADDR}" \
    --master-port="${MASTER_PORT}" \
    --max-restarts=0 \
    "${CODE_ROOT}/train.py" "${CONFIG}" --exp-name="${EXP_NAME}" "${EXTRA_ARGS[@]}"

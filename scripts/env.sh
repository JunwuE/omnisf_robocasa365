#!/usr/bin/env bash
# Shared environment for RoboCasa365 post-training with openpi-omnisf.
#
# Source this file from the other scripts, or `source` it manually before
# running python entry points directly.

# Resolve the code/ directory regardless of how this file is invoked.
_ROBOCASA365_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CODE_ROOT="$(cd "${_ROBOCASA365_ENV_DIR}/.." && pwd)"
export WORKSPACE_ROOT="$(cd "${CODE_ROOT}/.." && pwd)"

# --- openpi-omnisf checkout (framework + venv + pretrained checkpoints) ------
export OPENPI_ROOT="${OPENPI_ROOT:-/root/nas/lfh/omnisf}"
export OPENPI_PYTHON="${OPENPI_PYTHON:-${OPENPI_ROOT}/.venv/bin/python}"

# --- RoboCasa365 post-training paths ----------------------------------------
# Read-only source: the flat LeRobot v3.0 mirrors. Never modified.
export ROBOCASA365_SOURCE_DATA_ROOT="${ROBOCASA365_SOURCE_DATA_ROOT:-${WORKSPACE_ROOT}/data}"
# Converted (LeRobot v2.1) target datasets used for training.
export ROBOCASA365_DATA_ROOT="${ROBOCASA365_DATA_ROOT:-${WORKSPACE_ROOT}/data_v2.1}"
# Normalization-stats assets.
export ROBOCASA365_ASSETS_DIR="${ROBOCASA365_ASSETS_DIR:-${CODE_ROOT}/assets}"
# Checkpoints produced by training.
export ROBOCASA365_CHECKPOINT_DIR="${ROBOCASA365_CHECKPOINT_DIR:-${CODE_ROOT}/checkpoints}"
# Staging directory used during the v3.0 -> v2.1 conversion (symlinks into the
# source; created and removed automatically).
export ROBOCASA365_STAGING_ROOT="${ROBOCASA365_STAGING_ROOT:-${WORKSPACE_ROOT}/.cache/robocasa365_v21_staging}"

# omnivggt's aggregator loads DINOv2 from the *CWD-relative* path
#   "checkpoints/dinov2/dinov2_vitl14_reg4_pretrain.pth"
# and otherwise falls back to torch.hub (network download; races across ranks ->
# SIGABRT). We run from CODE_ROOT, so expose the omnisf-bundled copy via symlink.
_DINOV2_CACHE="${OPENPI_ROOT}/checkpoints/dinov2"
if [[ -d "${_DINOV2_CACHE}" && ! -e "${CODE_ROOT}/checkpoints/dinov2" ]]; then
    mkdir -p "${CODE_ROOT}/checkpoints"
    ln -s "${_DINOV2_CACHE}" "${CODE_ROOT}/checkpoints/dinov2"
fi

# Pretrained start checkpoint: "omnisf" (default) or "pi05_base", or a literal path.
export ROBOCASA365_PRETRAIN="${ROBOCASA365_PRETRAIN:-omnisf}"

# --- v3.0 -> v2.1 conversion -------------------------------------------------
# The converter needs the newer lerobot API and the `datasets`/`jsonlines`
# packages, which live in the conda `lerobot` environment.
export LEROBOT_CONDA_ENV="${LEROBOT_CONDA_ENV:-lerobot}"
export V30_TO_V21_CONVERTER="${V30_TO_V21_CONVERTER:-/root/nas/lfh/any4lerobot-main/ds_version_convert/v30_to_v21/convert_dataset_v30_to_v21.py}"
export FFMPEG_BIN="${FFMPEG_BIN:-ffmpeg}"

# --- Runtime defaults (mirror the omnisf launcher) ---------------------------
export PYTHONPATH="${CODE_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
# Never write .pyc bytecode caches (keeps /root/nas/lfh/omnisf strictly read-only).
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_ALLOCATOR="${XLA_PYTHON_CLIENT_ALLOCATOR:-platform}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${WORKSPACE_ROOT}/.cache/huggingface/datasets}"
export WANDB_DIR="${WANDB_DIR:-${CODE_ROOT}/wandb}"

mkdir -p "${HF_DATASETS_CACHE}" "${WANDB_DIR}"

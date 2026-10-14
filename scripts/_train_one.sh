#!/usr/bin/env bash
# Internal helper shared by the per-subset training scripts (train_atomic.sh,
# train_composite_seen.sh, train_composite_unseen.sh, train_mixed.sh).
#
# It sets the RoboCasa365 post-training defaults and forwards to train_8gpu.sh:
#   * physical (micro) batch size : 512
#   * gradient accumulation       : 2   (=> effective batch 1024)
#   * train steps                 : 20000
#   * GPUs                        : all 8 local devices (fsdp_devices=8)
#   * start weights               : omnisf_9.0_jax/35999 (ROBOCASA365_PRETRAIN)
#
# Every value can be overridden from the environment, e.g.
#   ROBOCASA365_NUM_TRAIN_STEPS=30000 bash scripts/train_atomic.sh my_exp
#
# Usage: _train_one.sh <config> <default_exp> [exp_name] [extra train.py args...]
# Not meant to be called directly; use the per-subset wrappers.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/env.sh
source "${SCRIPT_DIR}/env.sh"

CONFIG="${1:?usage: _train_one.sh <config> <default_exp> [exp_name] [extra args...]}"
shift
DEFAULT_EXP="${1:?usage: _train_one.sh <config> <default_exp> [exp_name] [extra args...]}"
shift
if [[ $# -gt 0 ]]; then
    EXP="$1"
    shift
else
    EXP="${DEFAULT_EXP}"
fi

# --- RoboCasa365 post-training defaults (env-overridable) --------------------
export ROBOCASA365_BATCH_SIZE="${ROBOCASA365_BATCH_SIZE:-512}"
export ROBOCASA365_GRAD_ACCUM="${ROBOCASA365_GRAD_ACCUM:-2}"
export ROBOCASA365_NUM_TRAIN_STEPS="${ROBOCASA365_NUM_TRAIN_STEPS:-20000}"
# Save the weights only once, at the end of the run.  The trainer also always
# saves the final step (num_train_steps - 1), so the checkpoint dir is ".../19999".
export ROBOCASA365_SAVE_INTERVAL="${ROBOCASA365_SAVE_INTERVAL:-20000}"

# A fresh run needs --overwrite (the exp dir may already exist); pass --resume
# (or any other train.py flag) explicitly to change that.
EXTRA_ARGS=("$@")
if [[ ${#EXTRA_ARGS[@]} -eq 0 ]]; then
    EXTRA_ARGS=("--overwrite")
fi

echo "config      : ${CONFIG}"
echo "exp name    : ${EXP}"
echo "micro batch : ${ROBOCASA365_BATCH_SIZE} x accum ${ROBOCASA365_GRAD_ACCUM}" \
     "= effective $((ROBOCASA365_BATCH_SIZE * ROBOCASA365_GRAD_ACCUM))"
echo "train steps : ${ROBOCASA365_NUM_TRAIN_STEPS}"
echo "save interval: ${ROBOCASA365_SAVE_INTERVAL} (final step always saved)"
echo "pretrain    : ${ROBOCASA365_PRETRAIN}"
echo "data root   : ${ROBOCASA365_DATA_ROOT}"
echo "extra args  : ${EXTRA_ARGS[*]}"
echo

exec bash "${SCRIPT_DIR}/train_8gpu.sh" "${CONFIG}" "${EXP}" "${EXTRA_ARGS[@]}"

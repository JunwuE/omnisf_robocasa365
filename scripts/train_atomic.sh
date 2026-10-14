#!/usr/bin/env bash
# Post-train omnisf pi0.5 on the RoboCasa365 **Atomic-Seen** target split only.
#
# Defaults: micro batch 512 x grad accum 2 (= effective 1024), 20000 steps,
# all 8 local GPUs, starting from the omnisf_9.0_jax/35999 weights.
#
# Usage:
#   bash scripts/train_atomic.sh                       # exp name: atomic_run0
#   bash scripts/train_atomic.sh my_exp                # custom exp name
#   bash scripts/train_atomic.sh my_exp --resume       # extra train.py args
#   ROBOCASA365_NUM_TRAIN_STEPS=30000 bash scripts/train_atomic.sh
#
# Detached (survives SSH disconnect):
#   bash scripts/submit_detached.sh train_atomic bash scripts/train_atomic.sh

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/_train_one.sh" robocasa365_target_atomic atomic_run0 "$@"

#!/usr/bin/env bash
# Post-train omnisf pi0.5 on the RoboCasa365 **Composite-Unseen** target split only.
#
# Defaults: micro batch 512 x grad accum 2 (= effective 1024), 20000 steps,
# all 8 local GPUs, starting from the omnisf_9.0_jax/35999 weights.
#
# Usage:
#   bash scripts/train_composite_unseen.sh                # exp: composite_unseen_run0
#   bash scripts/train_composite_unseen.sh my_exp
#   bash scripts/train_composite_unseen.sh my_exp --resume
#   ROBOCASA365_NUM_TRAIN_STEPS=30000 bash scripts/train_composite_unseen.sh
#
# Detached (survives SSH disconnect):
#   bash scripts/submit_detached.sh train_comp_unseen bash scripts/train_composite_unseen.sh

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/_train_one.sh" robocasa365_target_composite_unseen composite_unseen_run0 "$@"

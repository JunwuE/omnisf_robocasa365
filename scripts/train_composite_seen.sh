#!/usr/bin/env bash
# Post-train omnisf pi0.5 on the RoboCasa365 **Composite-Seen** target split only.
#
# Defaults: micro batch 512 x grad accum 2 (= effective 1024), 20000 steps,
# all 8 local GPUs, starting from the omnisf_9.0_jax/35999 weights.
#
# Usage:
#   bash scripts/train_composite_seen.sh                  # exp: composite_seen_run0
#   bash scripts/train_composite_seen.sh my_exp
#   bash scripts/train_composite_seen.sh my_exp --resume
#   ROBOCASA365_NUM_TRAIN_STEPS=30000 bash scripts/train_composite_seen.sh
#
# Detached (survives SSH disconnect):
#   bash scripts/submit_detached.sh train_comp_seen bash scripts/train_composite_seen.sh

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/_train_one.sh" robocasa365_target_composite_seen composite_seen_run0 "$@"

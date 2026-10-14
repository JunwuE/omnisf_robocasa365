#!/usr/bin/env bash
# Post-train omnisf pi0.5 on **all three RoboCasa365 target subsets at once**
# (atomic + composite-seen + composite-unseen mixed in a single model/run).
#
# The three subsets are sampled with weights ROBOCASA365_MIXED_WEIGHTS
# (default "1.0,1.0,1.0" => proportional to dataset size, i.e. a plain union),
# and share the merged normalisation statistics from the asset
# `robocasa365_target_mixed`.
#
# Defaults: micro batch 512 x grad accum 2 (= effective 1024), 20000 steps,
# all 8 local GPUs, starting from the omnisf_9.0_jax/35999 weights.
#
# Usage:
#   bash scripts/train_mixed.sh                        # exp: mixed_run0
#   bash scripts/train_mixed.sh my_exp
#   ROBOCASA365_MIXED_WEIGHTS=2,1,1 bash scripts/train_mixed.sh   # 2x atomic
#   ROBOCASA365_NUM_TRAIN_STEPS=30000 bash scripts/train_mixed.sh
#
# Detached (survives SSH disconnect):
#   bash scripts/submit_detached.sh train_mixed bash scripts/train_mixed.sh

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/_train_one.sh" robocasa365_target_mixed mixed_run0 "$@"

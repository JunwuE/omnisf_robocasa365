#!/usr/bin/env bash
# Compute normalization statistics for the RoboCasa365 post-training configs.
#
# This uses compute_norm_stats_fast.py, which reads ONLY the
# `observation.state` / `action` Parquet columns directly from the source data.
# It is CPU-only, never decodes video, and does not require the v3.0 -> v2.1
# conversion to have run (the rows are identical).
#
# Usage:
#   scripts/compute_norm_stats.sh                                    # all three splits
#   scripts/compute_norm_stats.sh robocasa365_target_composite_seen  # one split
#   scripts/compute_norm_stats.sh --max-frames 1000000               # extra flags
#
# Stats are written to $ROBOCASA365_ASSETS_DIR/<config_name>/norm_stats.json.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/env.sh
source "${SCRIPT_DIR}/env.sh"

ALL_CONFIGS=(
    "robocasa365_target_atomic"
    "robocasa365_target_composite_seen"
    "robocasa365_target_composite_unseen"
    "robocasa365_target_mixed"
)

CONFIGS=()
EXTRA_ARGS=()
for arg in "$@"; do
    if [[ "${arg}" == robocasa365_target_* ]]; then
        CONFIGS+=("${arg}")
    else
        EXTRA_ARGS+=("${arg}")
    fi
done
if [[ ${#CONFIGS[@]} -eq 0 ]]; then
    CONFIGS=("${ALL_CONFIGS[@]}")
fi

[[ -x "${OPENPI_PYTHON}" ]] || {
    echo "ERROR: openpi python not found/executable: ${OPENPI_PYTHON}" >&2
    exit 1
}

cd "${CODE_ROOT}"

for config in "${CONFIGS[@]}"; do
    echo "==========================================================="
    echo "Computing norm stats for: ${config}"
    echo "==========================================================="
    "${OPENPI_PYTHON}" "${CODE_ROOT}/compute_norm_stats_fast.py" \
        --config-name "${config}" "${EXTRA_ARGS[@]}"
done

echo "Done. Norm stats written under: ${ROBOCASA365_ASSETS_DIR}"

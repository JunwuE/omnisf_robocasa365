#!/usr/bin/env bash
# Run RoboCasa365 post-training jobs one after another on the local 8 GPUs.
#
# Each job uses all 8 GPUs, so they are queued sequentially.  A failing job is
# logged and the queue continues with the next one.
#
# Jobs are delegated to the per-subset scripts:
#   atomic           -> scripts/train_atomic.sh
#   composite_seen   -> scripts/train_composite_seen.sh
#   composite_unseen -> scripts/train_composite_unseen.sh
#   mixed            -> scripts/train_mixed.sh
#
# Usage:
#   scripts/train_all_splits.sh                       # 3 separate splits
#   scripts/train_all_splits.sh --with-mixed          # 3 splits + mixed run
#   scripts/train_all_splits.sh mixed atomic          # only the named jobs
#
# Batch/steps defaults come from scripts/_train_one.sh (micro 512 x accum 2,
# 20000 steps) and can be overridden through the environment, e.g.
#   ROBOCASA365_NUM_TRAIN_STEPS=30000 bash scripts/train_all_splits.sh
#
# Detached launch (survives SSH disconnect):
#   bash scripts/submit_detached.sh train_all bash scripts/train_all_splits.sh --with-mixed

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/env.sh
source "${SCRIPT_DIR}/env.sh"

DEFAULT_JOBS=(atomic composite_seen composite_unseen)

declare -A JOB_SCRIPT=(
    [atomic]="train_atomic.sh"
    [composite_seen]="train_composite_seen.sh"
    [composite_unseen]="train_composite_unseen.sh"
    [mixed]="train_mixed.sh"
)

JOBS=()
WITH_MIXED=0
for arg in "$@"; do
    case "${arg}" in
        --with-mixed) WITH_MIXED=1 ;;
        -h|--help) sed -n '2,24p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) JOBS+=("${arg}") ;;
    esac
done
if [[ ${#JOBS[@]} -eq 0 ]]; then
    JOBS=("${DEFAULT_JOBS[@]}")
    [[ "${WITH_MIXED}" -eq 1 ]] && JOBS+=(mixed)
fi

for job in "${JOBS[@]}"; do
    if [[ -z "${JOB_SCRIPT[${job}]:-}" ]]; then
        echo "Unknown job '${job}'. Known jobs: ${!JOB_SCRIPT[*]}" >&2
        exit 1
    fi
done

FAILED=()
echo "RoboCasa365 post-training queue"
echo "started    : $(date '+%F %T')"
echo "jobs       : ${JOBS[*]}"
echo "steps      : ${ROBOCASA365_NUM_TRAIN_STEPS:-20000 (default)}"
echo "checkpoints: ${ROBOCASA365_CHECKPOINT_DIR}"
echo

for job in "${JOBS[@]}"; do
    echo "==================================================================="
    echo "[$(date '+%F %T')] START ${job} ($(basename "${JOB_SCRIPT[${job}]}"))"
    echo "==================================================================="
    if bash "${SCRIPT_DIR}/${JOB_SCRIPT[${job}]}"; then
        echo "[$(date '+%F %T')] DONE  ${job}"
    else
        echo "[$(date '+%F %T')] FAILED ${job}"
        FAILED+=("${job}")
    fi
    echo
done

echo "==================================================================="
echo "[$(date '+%F %T')] queue finished"
if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "failed jobs: ${FAILED[*]}"
    exit 1
fi
echo "all jobs completed"

#!/usr/bin/env bash
# Submit a long-running command fully detached from the current terminal.
#
# Uses `setsid` + `nohup` so the job becomes its own session leader and ignores
# SIGHUP: it keeps running after you close/loose the SSH connection.  The job's
# stdout/stderr go to a log file, and its PID is written next to the log.
#
# Usage:
#   scripts/submit_detached.sh <log_label> <command> [args...]
#
# Examples:
#   # Convert v3.0 -> v2.1 (writes data_v2.1, source stays read-only)
#   bash scripts/submit_detached.sh convert bash scripts/convert_v30_to_v21.sh
#
#   # Train on 8 GPUs in the background
#   bash scripts/submit_detached.sh train_atomic bash scripts/train_8gpu.sh \
#       robocasa365_target_atomic atomic_run0
#
# Inspect:
#   tail -f code/logs/<log_label>_<timestamp>.log
#   ps -fp "$(cat code/logs/<log_label>.pid)"

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/env.sh
source "${SCRIPT_DIR}/env.sh"

if [[ $# -lt 2 ]]; then
    echo "usage: $0 <log_label> <command> [args...]" >&2
    exit 2
fi

LABEL="$1"
shift

# A predictable PATH for detached (non-login) shells.
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/opt/conda/bin:${PATH}"

mkdir -p "${CODE_ROOT}/logs"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${CODE_ROOT}/logs/${LABEL}_${STAMP}.log"
PID_FILE="${CODE_ROOT}/logs/${LABEL}.pid"

# setsid detaches into a new session; nohup ignores SIGHUP. stdin is /dev/null
# so the job can never block on the terminal.
setsid nohup bash -c 'cd "$CODE_ROOT"; exec "$@"' _ "$@" \
    >"${LOG_FILE}" 2>&1 </dev/null &
JOB_PID=$!

echo "${JOB_PID}" >"${PID_FILE}"

echo "submitted : ${LABEL}"
echo "pid       : ${JOB_PID}  (${PID_FILE})"
echo "log       : ${LOG_FILE}"
echo "command   : $*"
echo
echo "This job is detached from the terminal and will keep running after you disconnect."
echo "Follow it with: tail -f ${LOG_FILE}"

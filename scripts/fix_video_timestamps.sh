#!/usr/bin/env bash
# Fix the per-episode video timestamps of the converted v2.1 datasets.
#
# The any4lerobot v3.0 -> v2.1 converter splits each concatenated camera video
# with `ffmpeg -ss <start> -c copy` but keeps the segment's original PTS, so the
# produced per-episode mp4 files start at t = <episode from_timestamp> (0.1s for
# the first episode) instead of 0.  LeRobot's timestamp check then rejects every
# sample (tolerance 1e-4s) when decoding video.
#
# A plain remux (`ffmpeg -i in -c copy out`) rebases the timestamps to zero
# without re-encoding and without dropping frames, which restores compatibility.
#
# Properties:
#   * parallel (xargs -P), stream copy only, no re-encode;
#   * idempotent: files already starting at 0 are skipped, so it is resumable;
#   * safe: each file is written to a sibling temp file and verified (start_time
#     == 0 and nb_frames unchanged) before atomically replacing the original.
#
# Usage:
#   scripts/fix_video_timestamps.sh                    # all three target splits
#   scripts/fix_video_timestamps.sh --workers 32
#   scripts/fix_video_timestamps.sh --dry-run
#   scripts/fix_video_timestamps.sh robocasa365-target-atomic
#
# This only touches $ROBOCASA365_DATA_ROOT (the converted copies); the v3.0
# source under $ROBOCASA365_SOURCE_DATA_ROOT is never modified.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/env.sh
source "${SCRIPT_DIR}/env.sh"

DEFAULT_DATASETS=(
    "robocasa365-target-atomic"
    "robocasa365-target-composite-seen"
    "robocasa365-target-composite-unseen"
)

WORKERS="${FIX_WORKERS:-16}"
DRY_RUN=0
DATASETS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --workers) WORKERS="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) sed -n '2,32p' "${BASH_SOURCE[0]}"; exit 0 ;;
        --*) echo "Unknown option: $1" >&2; exit 1 ;;
        *) DATASETS+=("$1"); shift ;;
    esac
done
[[ ${#DATASETS[@]} -eq 0 ]] && DATASETS=("${DEFAULT_DATASETS[@]}")

export FFMPEG_BIN FFPROBE_BIN="${FFPROBE_BIN:-ffprobe}"

probe_video() {
    # prints "<start_time> <nb_frames>" (either may be empty)
    "${FFPROBE_BIN}" -v error -select_streams v:0 \
        -show_entries stream=start_time,nb_frames -of default=noprint_wrappers=1:nokey=1 "$1" 2>/dev/null \
        | paste -sd' ' -
}

fix_one() {
    local f="$1"
    [[ -f "$f" ]] || return 0

    local before start nb
    before="$(probe_video "$f")"
    start="$(echo "$before" | awk '{print $1}')"
    nb="$(echo "$before" | awk '{print $2}')"

    if [[ -z "$start" ]]; then
        echo "SKIP-BADPROBE $f"
        return 0
    fi
    # Already zero-based -> nothing to do (idempotent / resumable).
    if [[ "$start" == "0.000000" || "$start" == "0" ]]; then
        echo "SKIP $f"
        return 0
    fi

    local tmp="${f}.tsfix.mp4"
    rm -f "$tmp"
    if ! "${FFMPEG_BIN}" -hide_banner -loglevel error -i "$f" -c copy -y "$tmp" 2>/dev/null; then
        echo "FAIL-FFMPEG $f"
        rm -f "$tmp"
        return 0
    fi

    local after new_start new_nb
    after="$(probe_video "$tmp")"
    new_start="$(echo "$after" | awk '{print $1}')"
    new_nb="$(echo "$after" | awk '{print $2}')"

    if [[ "$new_start" != "0.000000" && "$new_start" != "0" ]]; then
        echo "FAIL-VERIFY-START $f (start_time=$new_start)"
        rm -f "$tmp"
        return 0
    fi
    if [[ -n "$nb" && -n "$new_nb" && "$nb" != "$new_nb" ]]; then
        echo "FAIL-VERIFY-FRAMES $f ($nb -> $new_nb)"
        rm -f "$tmp"
        return 0
    fi

    mv -f "$tmp" "$f"
    echo "FIXED $f ($start -> $new_start, frames=$new_nb)"
}
export -f fix_one probe_video

command -v "${FFMPEG_BIN}" >/dev/null 2>&1 || { echo "ERROR: ffmpeg not found" >&2; exit 1; }
command -v "${FFPROBE_BIN}" >/dev/null 2>&1 || { echo "ERROR: ffprobe not found" >&2; exit 1; }

echo "Root   : ${ROBOCASA365_DATA_ROOT}"
echo "Workers: ${WORKERS}"
echo "Dry-run: ${DRY_RUN}"

for name in "${DATASETS[@]}"; do
    root="${ROBOCASA365_DATA_ROOT}/${name}"
    [[ -d "${root}/videos" ]] || { echo "[${name}] SKIP: no videos dir at ${root}"; continue; }

    total=$(find "${root}/videos" -name 'episode_*.mp4' | wc -l)
    echo "==================================================================="
    echo "[${name}] videos=${total}"
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        echo "  DRY-RUN: would remux non-zero-start videos with ${WORKERS} workers."
        continue
    fi

    # Remove stale temp files from a previously interrupted run.
    find "${root}/videos" -name 'episode_*.mp4.tsfix.mp4' -delete 2>/dev/null || true

    start_ts=$(date +%s)
    find "${root}/videos" -name 'episode_*.mp4' -print0 \
        | xargs -0 -P "${WORKERS}" -n 1 bash -c 'fix_one "$1"' _ \
        > "${CODE_ROOT}/logs/tsfix_${name}_$(date +%Y%m%d_%H%M%S).log" 2>&1
    end_ts=$(date +%s)

    log_file=$(ls -t "${CODE_ROOT}/logs"/tsfix_${name}_*.log 2>/dev/null | head -1)
    fixed=$(grep -c '^FIXED' "$log_file" 2>/dev/null || true)
    skipped=$(grep -c '^SKIP ' "$log_file" 2>/dev/null || true)
    failed=$(grep -c '^FAIL' "$log_file" 2>/dev/null || true)
    fixed=${fixed:-0}; skipped=${skipped:-0}; failed=${failed:-0}
    echo "  done in $((end_ts - start_ts))s | fixed=${fixed} skipped=${skipped} failed=${failed}"
    echo "  log: ${log_file}"
    if [[ "${failed}" -gt 0 ]]; then
        echo "  !! ${failed} file(s) failed; inspect the log above." >&2
    fi
done

echo "==================================================================="
echo "Done. Verify with: $OPENPI_PYTHON ${CODE_ROOT}/preflight.py"

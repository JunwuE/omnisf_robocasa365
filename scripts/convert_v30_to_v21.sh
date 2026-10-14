#!/usr/bin/env bash
# Convert the RoboCasa365 target datasets from LeRobot v3.0 to v2.1.
#
# The openpi-omnisf checkout pins LeRobot v2.1 (CODEBASE_VERSION="v2.1"), so the
# flat v3.0 mirrors must be converted before training.
#
# IMPORTANT: the v3.0 source under $ROBOCASA365_SOURCE_DATA_ROOT is treated as
# READ-ONLY and is never modified or deleted.  Conversion happens on a staging
# directory that only contains symlinks into the source, and the resulting v2.1
# tree is written to $ROBOCASA365_DATA_ROOT (default: <workspace>/data_v21).
#
# Usage:
#   scripts/convert_v30_to_v21.sh                    # all three target splits
#   scripts/convert_v30_to_v21.sh --dry-run          # show the plan only
#   scripts/convert_v30_to_v21.sh --overwrite        # re-convert existing outputs
#   scripts/convert_v30_to_v21.sh robocasa365-target-atomic
#
# Requirements: conda env "$LEROBOT_CONDA_ENV" (default: lerobot) providing the
# new LeRobot dataset API, plus ffmpeg on PATH.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/env.sh
source "${SCRIPT_DIR}/env.sh"

DEFAULT_DATASETS=(
    "robocasa365-target-atomic"
    "robocasa365-target-composite-seen"
    "robocasa365-target-composite-unseen"
)

DRY_RUN=0
OVERWRITE=0
DATASETS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        --overwrite) OVERWRITE=1; shift ;;
        -h|--help)
            sed -n '2,28p' "${BASH_SOURCE[0]}"
            exit 0
            ;;
        --*)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
        *)
            DATASETS+=("$1")
            shift
            ;;
    esac
done

if [[ ${#DATASETS[@]} -eq 0 ]]; then
    DATASETS=("${DEFAULT_DATASETS[@]}")
fi

json_version() {
    python3 - "$1" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    print(json.load(stream)["codebase_version"])
PY
}

validate_dataset() {
    local root="$1"
    python3 - "$root" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])

with (root / "meta/info.json").open(encoding="utf-8") as stream:
    info = json.load(stream)

assert info["codebase_version"] == "v2.1", f"expected v2.1, got {info['codebase_version']}"
episodes = int(info["total_episodes"])
videos_per_episode = len([k for k, f in info["features"].items() if f.get("dtype") == "video"])

parquet = list(root.glob("data/chunk-*/episode_*.parquet"))
video_files = list(root.glob("videos/chunk-*/*/episode_*.mp4"))

for filename in ("episodes.jsonl", "episodes_stats.jsonl"):
    path = root / "meta" / filename
    assert path.is_file(), f"missing {path}"
    with path.open("rb") as stream:
        lines = sum(1 for line in stream if line.strip())
    assert lines == episodes, f"{filename}: {lines} lines != {episodes} episodes"

assert (root / "meta/tasks.jsonl").is_file(), "missing meta/tasks.jsonl"
assert len(parquet) == episodes, f"{len(parquet)} episode parquets != {episodes}"
assert len(video_files) == episodes * videos_per_episode, (
    f"{len(video_files)} videos != {episodes} x {videos_per_episode}"
)
assert all(p.stat().st_size > 0 for p in parquet), "empty parquet file(s) found"
assert all(p.stat().st_size > 0 for p in video_files), "empty video file(s) found"

print(
    f"  OK: {root.name} v2.1 | episodes={episodes} | "
    f"frames={info['total_frames']} | tasks={info['total_tasks']} | videos={len(video_files)}"
)
PY
}

assert_source_unchanged() {
    local src="$1" expected_version="$2"
    [[ -f "${src}/meta/info.json" ]] || {
        echo "  ERROR: source disappeared: ${src}/meta/info.json" >&2
        exit 1
    }
    local now
    now="$(json_version "${src}/meta/info.json")"
    [[ "${now}" == "${expected_version}" ]] || {
        echo "  ERROR: source version changed from ${expected_version} to ${now}: ${src}" >&2
        exit 1
    }
}

activate_lerobot_env() {
    if [[ -n "${CONDA_PREFIX:-}" && "$(basename "${CONDA_PREFIX}")" == "${LEROBOT_CONDA_ENV}" ]]; then
        return 0
    fi
    if ! command -v conda >/dev/null 2>&1; then
        export PATH="/opt/conda/bin:${PATH}"
    fi
    if ! command -v conda >/dev/null 2>&1; then
        echo "ERROR: conda not found; cannot activate env '${LEROBOT_CONDA_ENV}'." >&2
        exit 1
    fi
    eval "$(conda shell.bash hook)"
    conda activate "${LEROBOT_CONDA_ENV}"
    if [[ "$(basename "${CONDA_PREFIX:-}")" != "${LEROBOT_CONDA_ENV}" ]]; then
        echo "ERROR: failed to activate conda env '${LEROBOT_CONDA_ENV}'." >&2
        exit 1
    fi
}

command -v "${FFMPEG_BIN}" >/dev/null 2>&1 || {
    echo "ERROR: ffmpeg ('${FFMPEG_BIN}') not found on PATH." >&2
    exit 1
}
[[ -f "${V30_TO_V21_CONVERTER}" ]] || {
    echo "ERROR: converter not found: ${V30_TO_V21_CONVERTER}" >&2
    exit 1
}

echo "Source (read-only): ${ROBOCASA365_SOURCE_DATA_ROOT}"
echo "Output (v2.1)     : ${ROBOCASA365_DATA_ROOT}"
echo "Staging           : ${ROBOCASA365_STAGING_ROOT}"

for name in "${DATASETS[@]}"; do
    src="${ROBOCASA365_SOURCE_DATA_ROOT}/${name}"
    dst="${ROBOCASA365_DATA_ROOT}/${name}"
    echo "==================================================================="
    echo "[${name}]"
    echo "  src=${src}"
    echo "  dst=${dst}"

    if [[ ! -f "${src}/meta/info.json" ]]; then
        echo "  SKIP: source meta/info.json not found." >&2
        continue
    fi
    src_version="$(json_version "${src}/meta/info.json")"
    echo "  source version: ${src_version}"

    if [[ -f "${dst}/meta/info.json" ]] && [[ "$(json_version "${dst}/meta/info.json")" == "v2.1" ]]; then
        if [[ "${OVERWRITE}" -eq 0 ]]; then
            echo "  output already exists; validating and skipping (use --overwrite to redo)."
            validate_dataset "${dst}"
            continue
        fi
        echo "  --overwrite: removing existing output: ${dst}"
        [[ "${DRY_RUN}" -eq 0 ]] && rm -rf -- "${dst}"
    fi

    if [[ "${src_version}" != "v3.0" ]]; then
        echo "  ERROR: expected a v3.0 source, got '${src_version}'." >&2
        exit 1
    fi

    if [[ "${DRY_RUN}" -eq 1 ]]; then
        echo "  DRY-RUN: would convert v3.0 -> v2.1 into ${dst} (source untouched)."
        continue
    fi

    staging="${ROBOCASA365_STAGING_ROOT}/${name}"
    # Guard: only ever touch real directories directly under the staging root.
    for path in "${staging}" "${staging}_v3.0" "${staging}_v2.1"; do
        if [[ -L "${path}" ]]; then
            echo "  ERROR: refusing to remove symlink: ${path}" >&2
            exit 1
        fi
    done
    rm -rf -- "${staging}" "${staging}_v3.0" "${staging}_v2.1"
    mkdir -p "${staging}"

    # Expose the read-only source to the converter through symlinks.
    ln -s "${src}/meta" "${staging}/meta"
    ln -s "${src}/data" "${staging}/data"
    ln -s "${src}/videos" "${staging}/videos"

    activate_lerobot_env
    echo "  converting v3.0 -> v2.1 (ffmpeg stream-copies per-episode videos)..."
    python "${V30_TO_V21_CONVERTER}" --repo-id "local/${name}" --root "${staging}"

    # Safety: the original source must be intact after conversion.
    assert_source_unchanged "${src}" "${src_version}"

    # Publish the converted tree next to the other outputs.
    mkdir -p "${ROBOCASA365_DATA_ROOT}"
    if [[ -e "${dst}" || -L "${dst}" ]]; then
        echo "  ERROR: refusing to overwrite: ${dst}" >&2
        exit 1
    fi
    mv "${staging}" "${dst}"

    # Clean up the staging leftovers (symlinks only; source is untouched).
    rm -rf -- "${staging}_v3.0" "${staging}_v2.1"

    validate_dataset "${dst}"
done

echo "==================================================================="
echo "Done. Source kept intact at : ${ROBOCASA365_SOURCE_DATA_ROOT}"
echo "Converted v2.1 datasets at  : ${ROBOCASA365_DATA_ROOT}"

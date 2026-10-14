#!/usr/bin/env python
"""Fast normalization-stats computation for RoboCasa365 target configs.

Instead of iterating the LeRobot dataset frame-by-frame (which also touches the
videos), this reads **only two Parquet columns** -- ``observation.state`` and
``action`` -- straight from the dataset, computes ``mean / std / q01 / q99`` and
writes the result in the openpi ``norm_stats.json`` format.

It works on the original **v3.0** source datasets as well as on converted v2.1
ones (the rows are identical), so it can run before, after, or independently of
``scripts/convert_v30_to_v21.sh``.  It is CPU-only and never decodes video.

Usage::

    $OPENPI_PYTHON compute_norm_stats_fast.py --config-name robocasa365_target_atomic
    $OPENPI_PYTHON compute_norm_stats_fast.py --config-name robocasa365_target_composite_seen \\
        --max-frames 2000000

Output: ``$ROBOCASA365_ASSETS_DIR/<config_name>/norm_stats.json``.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys

# Keep the external openpi-omnisf checkout strictly read-only (no .pyc writes).
sys.dont_write_bytecode = True

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import tyro  # noqa: E402

from robocasa365.configs import CONFIGS  # noqa: E402
from robocasa365.configs import SOURCE_DATASET_DIRS  # noqa: E402
from robocasa365.configs import SPLIT_DATASET_NAMES  # noqa: E402
from robocasa365.registry import register_configs  # noqa: E402

STATE_COLUMN = "observation.state"
ACTION_COLUMN = "action"


def _list_column_to_2d(column: pa.ChunkedArray, width_hint: int | None) -> np.ndarray:
    """Convert a Parquet list/fixed-size-list column into an ``(N, D)`` array."""
    try:
        flat = column.combine_chunks().flatten()
        values = np.asarray(flat.to_numpy(zero_copy_only=False), dtype=np.float32)
        if width_hint:
            return values.reshape(-1, width_hint)
        return values.reshape(len(column), -1)
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError, ValueError):
        arr = np.asarray(column.to_pylist(), dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        return arr


def _dataset_dirs(dataset_dir: pathlib.Path) -> list[pathlib.Path]:
    """Return the Parquet shards holding frames for a v3.0 or v2.1 dataset."""
    shards = sorted((dataset_dir / "data").glob("**/*.parquet"))
    if not shards:
        raise FileNotFoundError(f"No Parquet shards found under {dataset_dir / 'data'}")
    return shards


def _feature_widths(dataset_dir: pathlib.Path) -> dict[str, int]:
    info_path = dataset_dir / "meta/info.json"
    if not info_path.is_file():
        return {}
    with info_path.open(encoding="utf-8") as stream:
        info = json.load(stream)
    widths = {}
    for key in (STATE_COLUMN, ACTION_COLUMN):
        feature = info.get("features", {}).get(key)
        shape = feature.get("shape") if feature else None
        if shape:
            widths[key] = int(np.prod(shape))
    return widths


def _collect_column(
    shards: list[pathlib.Path],
    column: str,
    width_hint: int | None,
    max_frames: int | None,
    label: str,
):
    chunks: list[np.ndarray] = []
    total = 0
    for shard in shards:
        table = pq.read_table(shard, columns=[column])
        values = _list_column_to_2d(table.column(column), width_hint)
        chunks.append(values)
        total += values.shape[0]
        print(f"    [{label}] read {shard.name}: +{values.shape[0]} rows (total {total})", flush=True)
        if max_frames is not None and total >= max_frames:
            break

    values = np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 0), dtype=np.float32)
    if max_frames is not None and values.shape[0] > max_frames:
        values = values[:max_frames]
    return values


def _pad_to_dim(values: np.ndarray, dim: int) -> np.ndarray:
    """Right-pad the last axis of ``values`` with zeros up to ``dim``."""
    values = np.asarray(values, dtype=np.float32)
    width = values.shape[-1]
    if width >= dim:
        return values[..., :dim]
    pad = np.zeros((*values.shape[:-1], dim - width), dtype=np.float32)
    return np.concatenate([values, pad], axis=-1)


def compute_stats(values: np.ndarray) -> dict[str, np.ndarray]:
    if values.shape[0] < 2:
        raise ValueError(f"Need at least 2 rows to compute statistics, got {values.shape[0]}")
    mean = values.mean(axis=0).astype(np.float32)
    std = values.std(axis=0).astype(np.float32)
    q01 = np.quantile(values, 0.01, axis=0).astype(np.float32)
    q99 = np.quantile(values, 0.99, axis=0).astype(np.float32)
    return {"mean": mean, "std": std, "q01": q01, "q99": q99}



def _assets_dir_for(config) -> pathlib.Path:
    data_cfg = config.data
    asset_id = data_cfg.assets.asset_id or config.name
    if data_cfg.assets.assets_dir is not None:
        return pathlib.Path(data_cfg.assets.assets_dir).expanduser() / asset_id
    return config.assets_dirs / asset_id


def main(
    config_name: str,
    dataset_dir: str | None = None,
    max_frames: int | None = None,
) -> None:
    """Compute and save norm stats for one RoboCasa365 config.

    Args:
        config_name: e.g. ``robocasa365_target_atomic``.
        dataset_dir: dataset to read from (default: the v3.0 source directory).
        max_frames: optionally cap the number of frames used (for quick checks).
    """
    register_configs(CONFIGS)

    from openpi.shared import normalize as _normalize  # noqa: PLC0415

    from openpi.training import config as _config  # noqa: PLC0415

    config = _config.get_config(config_name)

    mixed_sources = getattr(config.data, "sources", None)
    if dataset_dir is not None:
        roots = [pathlib.Path(dataset_dir)]
    elif mixed_sources:
        # Mixed config: pool the v3.0 source datasets so the merged statistics
        # are computed exactly like the per-split ones.
        roots = [pathlib.Path(SOURCE_DATASET_DIRS[split]) for split in SPLIT_DATASET_NAMES]
    else:
        split = config_name.removeprefix("robocasa365_target_")
        if split not in SPLIT_DATASET_NAMES:
            raise ValueError(
                f"Cannot infer a RoboCasa365 dataset for config '{config_name}'. "
                f"Pass --dataset-dir explicitly."
            )
        candidates = [SOURCE_DATASET_DIRS[split], config.data.repo_id and pathlib.Path(config.data.repo_id)]
        root = next(
            (pathlib.Path(c) for c in candidates if c and (pathlib.Path(c) / "meta/info.json").is_file()),
            None,
        )
        if root is None:
            raise FileNotFoundError(
                "No dataset with meta/info.json found. Tried: "
                + ", ".join(str(c) for c in candidates if c)
            )
        roots = [root]

    missing = [str(root) for root in roots if not (root / "meta/info.json").is_file()]
    if missing:
        raise FileNotFoundError("Missing meta/info.json in: " + ", ".join(missing))

    action_dim = config.model.action_dim
    print(f"== {config_name} ==")
    print(f"  dataset(s)   : {', '.join(str(root) for root in roots)}")
    print(f"  action_dim   : {action_dim}")

    state_parts: list[np.ndarray] = []
    action_parts: list[np.ndarray] = []
    for root in roots:
        widths = _feature_widths(root)
        shards = _dataset_dirs(root)
        print(f"  [{root.name}] parquet shards={len(shards)}")
        state_parts.append(
            _collect_column(shards, STATE_COLUMN, widths.get(STATE_COLUMN), max_frames, f"{root.name}:state")
        )
        action_parts.append(
            _collect_column(shards, ACTION_COLUMN, widths.get(ACTION_COLUMN), max_frames, f"{root.name}:action")
        )

    state = np.concatenate(state_parts, axis=0) if len(state_parts) > 1 else state_parts[0]
    actions = np.concatenate(action_parts, axis=0) if len(action_parts) > 1 else action_parts[0]

    print(f"  state values : {state.shape}")
    print(f"  action values: {actions.shape}")

    # Compute on the native dims, then zero-pad the stats to the model action
    # dim (constant padded dims have mean=std=q01=q99=0, matching openpi).
    native_state_stats = compute_stats(state)
    native_action_stats = compute_stats(actions)
    state_stats = {key: _pad_to_dim(value, action_dim) for key, value in native_state_stats.items()}
    action_stats = {key: _pad_to_dim(value, action_dim) for key, value in native_action_stats.items()}

    norm_stats = {
        "state": _normalize.NormStats(**state_stats),
        "actions": _normalize.NormStats(**action_stats),
    }

    out_dir = _assets_dir_for(config)
    _normalize.save(out_dir, norm_stats)

    native_state = state.shape[-1]
    native_action = actions.shape[-1]
    print(f"  native state/action dims: {native_state}/{native_action} (padded to {action_dim})")
    print(f"  actions q01[:{native_action}]: {np.round(native_action_stats['q01'], 4).tolist()}")
    print(f"  actions q99[:{native_action}]: {np.round(native_action_stats['q99'], 4).tolist()}")
    print(f"  wrote: {out_dir / 'norm_stats.json'}")


if __name__ == "__main__":
    tyro.cli(main)

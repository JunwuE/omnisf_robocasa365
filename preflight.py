#!/usr/bin/env python
"""Pre-flight readiness check for RoboCasa365 post-training (no GPU needed).

Validates everything the trainer will depend on:

* openpi-omnisf checkout, venv python and ``scripts/train_align.py``;
* source v3.0 datasets (read-only) and converted v2.1 datasets
  (version, episode/frame counts, per-episode parquet + video files, no symlinks);
* norm-stats assets for every config (loadable, 32-dim state/actions);
* pretrained checkpoint and omnivggt weights;
* the actual data pipeline (repack + policy transforms) on one real sample.

Usage::

    OPENPI_ROOT=/root/nas/lfh/omnisf $OPENPI_PYTHON preflight.py
    $OPENPI_PYTHON preflight.py --no-load-sample     # skip the sample decode
"""

from __future__ import annotations

import dataclasses
import json
import os
import pathlib
import sys

# Keep the external openpi-omnisf checkout strictly read-only (no .pyc writes).
sys.dont_write_bytecode = True

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import tyro  # noqa: E402

from robocasa365.configs import CONFIGS  # noqa: E402
from robocasa365.configs import DATASET_DIRS  # noqa: E402
from robocasa365.configs import SOURCE_DATASET_DIRS  # noqa: E402
from robocasa365.configs import SPLIT_DATASET_NAMES  # noqa: E402
from robocasa365.configs import VGGT_PATH  # noqa: E402
from robocasa365.registry import OPENPI_ROOT  # noqa: E402

PASS = "PASS"
FAIL = "FAIL"
WARN = "WARN"


@dataclasses.dataclass
class Check:
    name: str
    status: str
    detail: str = ""


def _read_info(path: pathlib.Path) -> dict:
    with (path / "meta/info.json").open(encoding="utf-8") as stream:
        return json.load(stream)


def check_dataset(split: str, source: bool) -> Check:
    root = (SOURCE_DATASET_DIRS if source else DATASET_DIRS)[split]
    label = "source v3.0" if source else "converted v2.1"
    if not (root / "meta/info.json").is_file():
        return Check(f"{split}:{label}", FAIL, f"missing {root}/meta/info.json")

    info = _read_info(root)
    version = info["codebase_version"]
    expected = "v3.0" if source else "v2.1"
    if version != expected:
        return Check(f"{split}:{label}", FAIL, f"version {version} != {expected}")

    if source:
        return Check(f"{split}:{label}", PASS, f"{version} | episodes={info['total_episodes']} | frames={info['total_frames']}")

    episodes = int(info["total_episodes"])
    parquet = list(root.glob("data/chunk-*/episode_*.parquet"))
    videos = list(root.glob("videos/chunk-*/*/episode_*.mp4"))
    cameras = len([k for k, f in info["features"].items() if f.get("dtype") == "video"])

    # Targeted symlink check (a full rglob over ~25k NAS files is too slow):
    # if the staging symlinks had leaked, they would be at these levels.
    candidates = [root, *root.glob("data"), *root.glob("data/chunk-*"),
                  *root.glob("videos"), *root.glob("videos/*"),
                  *root.glob("videos/*/chunk-*"), *root.glob("videos/*/chunk-*/*.mp4"),
                  *root.glob("data/chunk-*/episode_*.parquet")]
    symlinks = [p for p in candidates if p.is_symlink()]

    problems = []
    if len(parquet) != episodes:
        problems.append(f"parquet {len(parquet)} != {episodes}")
    if len(videos) != episodes * cameras:
        problems.append(f"videos {len(videos)} != {episodes}x{cameras}")
    # Sample the size check: the conversion script already asserted every file
    # is non-empty when it published the dataset.
    sample = parquet[:5] + parquet[-5:] + videos[:5] + videos[-5:]
    empty = [p for p in sample if p.stat().st_size == 0]
    if empty:
        problems.append(f"{len(empty)} empty files (sampled)")
    if symlinks:
        problems.append(f"{len(symlinks)} symlinks (should be real files)")

    if problems:
        return Check(f"{split}:{label}", FAIL, "; ".join(problems))
    return Check(
        f"{split}:{label}",
        PASS,
        f"{version} | episodes={episodes} | frames={info['total_frames']} | "
        f"tasks={info['total_tasks']} | videos={len(videos)}",
    )


def check_norm_stats(config_name: str, action_dim: int) -> Check:
    from openpi.shared import normalize as _normalize  # noqa: PLC0415

    cfg = next(c for c in CONFIGS if c.name == config_name)
    assets_dir = pathlib.Path(cfg.data.assets.assets_dir).expanduser() / (cfg.data.assets.asset_id or cfg.name)
    try:
        stats = _normalize.load(assets_dir)
    except FileNotFoundError as exc:
        return Check(f"{config_name}:norm_stats", FAIL, str(exc))

    for key in ("state", "actions"):
        if key not in stats:
            return Check(f"{config_name}:norm_stats", FAIL, f"missing key '{key}'")
        for field in ("mean", "std", "q01", "q99"):
            value = getattr(stats[key], field, None)
            if value is None or np.asarray(value).shape != (action_dim,):
                return Check(
                    f"{config_name}:norm_stats",
                    FAIL,
                    f"{key}.{field} shape {None if value is None else np.asarray(value).shape} != ({action_dim},)",
                )
    return Check(f"{config_name}:norm_stats", PASS, f"{assets_dir}/norm_stats.json (32-dim)")


def check_path(label: str, path: pathlib.Path) -> Check:
    if path.is_dir():
        n = sum(1 for _ in path.iterdir())
        return Check(label, PASS, f"{path} ({n} entries)")
    if path.is_file():
        return Check(label, PASS, str(path))
    return Check(label, FAIL, f"missing: {path}")


def check_openpi_python() -> Check:
    python = pathlib.Path(os.environ.get("OPENPI_PYTHON", OPENPI_ROOT / ".venv/bin/python"))
    if not python.is_file():
        return Check("openpi python", FAIL, f"missing: {python}")
    return Check("openpi python", PASS, str(python))


def check_train_entry() -> Check:
    entry = OPENPI_ROOT / "scripts/train_align.py"
    if not entry.is_file():
        return Check("train entry", FAIL, f"missing: {entry}")
    return Check("train entry", PASS, str(entry))


def check_sample_load(config_name: str) -> Check:
    """Load one real sample through the full training transforms (CPU only)."""
    import openpi.training.data_loader as _data_loader  # noqa: PLC0415
    import openpi.transforms as _transforms  # noqa: PLC0415
    from openpi.training import config as _config  # noqa: PLC0415

    from robocasa365.registry import register_configs  # noqa: PLC0415

    register_configs(CONFIGS)
    cfg = _config.get_config(config_name)
    data_config = cfg.data.create(cfg.assets_dirs, cfg.model)

    try:
        dataset = _data_loader.create_torch_dataset(data_config, cfg.model.action_horizon, cfg.model)
        raw = dataset[0]
        xform = _transforms.compose([*data_config.repack_transforms.inputs, *data_config.data_transforms.inputs])
        out = xform(raw)
        state_shape = tuple(out["state"].shape)
        action_shape = tuple(out["actions"].shape)
        assert state_shape == (cfg.model.action_dim,), state_shape
        assert action_shape == (cfg.model.action_horizon, cfg.model.action_dim), action_shape
        for cam in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"):
            assert out["image"][cam].shape[-1] == 3, out["image"][cam].shape
        assert bool(out["image_mask"]["right_wrist_0_rgb"]) is True
        prompt = str(out["prompt"])[:60]
    except Exception as exc:  # noqa: BLE001
        return Check(f"{config_name}:sample", FAIL, f"{type(exc).__name__}: {exc}")
    return Check(
        f"{config_name}:sample",
        PASS,
        f"state{state_shape} actions{action_shape} prompt='{prompt}...'",
    )


def main(load_sample: bool = True, sample_config: str = "robocasa365_target_atomic") -> None:
    checks: list[Check] = []

    def add(check: Check) -> None:
        checks.append(check)
        print(f"[{check.status}] {check.name}  {check.detail}", flush=True)

    print("RoboCasa365 preflight (no GPU needed)", flush=True)

    add(check_openpi_python())
    add(check_train_entry())
    add(check_path("omnisf pretrain params", pathlib.Path(CONFIGS[0].weight_loader.params_path)))
    add(check_path("omnivggt weights", VGGT_PATH))

    for split in SPLIT_DATASET_NAMES:
        add(check_dataset(split, source=True))
        add(check_dataset(split, source=False))

    for cfg in CONFIGS:
        add(check_norm_stats(cfg.name, cfg.model.action_dim))

    if load_sample:
        add(check_sample_load(sample_config))

    failures = [c for c in checks if c.status == FAIL]
    print("=" * 60, flush=True)
    if failures:
        print(f"RESULT: FAIL ({len(failures)} check(s) failed)", flush=True)
        raise SystemExit(1)
    print(f"RESULT: READY ({len(checks)} checks passed)", flush=True)


if __name__ == "__main__":
    tyro.cli(main)

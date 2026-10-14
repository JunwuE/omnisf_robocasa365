#!/usr/bin/env python
"""Verify that the RoboCasa365 configs/policy can read all three target subsets.

Two modes:

* ``structure`` (default, fast, no video decode): for each config, read real rows
  from the converted v2.1 parquet shards + ``meta/tasks.jsonl`` and push them
  through the *actual* repack + policy transforms, asserting key mapping, prompt,
  state/action dims and image masks.  This does not depend on the video PTS fix.
* ``full``: build the real dataset through the openpi loader and decode a few
  samples (needs the per-episode video timestamps to be fixed, i.e.
  ``fix_video_timestamps.sh`` finished).

Usage::

    $OPENPI_PYTHON check_datasets.py                 # structure, all 3 configs
    $OPENPI_PYTHON check_datasets.py --mode full     # real decode, all 3 configs
    $OPENPI_PYTHON check_datasets.py --mode structure --config robocasa365_target_atomic
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
import pyarrow.parquet as pq  # noqa: E402
import tyro  # noqa: E402

from robocasa365.configs import CONFIGS  # noqa: E402
from robocasa365.configs import DATASET_DIRS  # noqa: E402
from robocasa365.registry import enable_fast_local_loading  # noqa: E402
from robocasa365.registry import register_configs  # noqa: E402


def _tasks(root: pathlib.Path) -> dict[int, str]:
    out = {}
    with (root / "meta/tasks.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rec = json.loads(line)
                out[int(rec["task_index"])] = rec["task"]
    return out


def _check_transformed(out: dict, cfg, expected_prompt: str) -> list[str]:
    problems = []
    if tuple(out["state"].shape) != (cfg.model.action_dim,):
        problems.append(f"state {tuple(out['state'].shape)} != ({cfg.model.action_dim},)")
    if tuple(out["actions"].shape) != (cfg.model.action_horizon, cfg.model.action_dim):
        problems.append(
            f"actions {tuple(out['actions'].shape)} != ({cfg.model.action_horizon}, {cfg.model.action_dim})"
        )
    for cam in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"):
        if cam not in out["image"]:
            problems.append(f"missing image[{cam}]")
    if not bool(out["image_mask"]["right_wrist_0_rgb"]):
        problems.append("right_wrist_0_rgb mask is False (pi0.5 needs it)")
    if str(out["prompt"]) != expected_prompt:
        problems.append(f"prompt mismatch: {str(out['prompt'])[:40]!r} != {expected_prompt[:40]!r}")
    return problems


def _structure_check_one(cfg, label: str, root: pathlib.Path, xform) -> tuple[bool, str]:
    """Check one subset: real parquet rows -> the given repack/policy transforms."""
    with (root / "meta/info.json").open(encoding="utf-8") as stream:
        info = json.load(stream)
    tasks = _tasks(root)

    video_keys = [k for k, f in info["features"].items() if f.get("dtype") == "video"]
    h, w, c = info["features"][video_keys[0]]["shape"]
    state_dim = int(np.prod(info["features"]["observation.state"]["shape"]))
    action_dim = int(np.prod(info["features"]["action"]["shape"]))

    shards = sorted((root / "data").glob("chunk-*/episode_*.parquet"))
    if not shards:
        return False, f"[{label}] no parquet shards under {root}/data"

    problems = []
    tested = [0, len(shards) // 2, len(shards) - 1]
    for ep in tested:
        table = pq.read_table(shards[ep], columns=["observation.state", "action", "task_index"])
        state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)[0]
        action = np.asarray(table["action"].to_pylist(), dtype=np.float32)[0]
        task_index = int(np.asarray(table["task_index"].to_pylist()).reshape(-1)[0])
        if state.shape != (state_dim,):
            problems.append(f"[{label}] ep{ep}: parquet state dim {state.shape} != {state_dim}")
        if action.shape != (action_dim,):
            problems.append(f"[{label}] ep{ep}: parquet action dim {action.shape} != {action_dim}")

        prompt = tasks[task_index]
        raw = {
            "observation.state": state,
            "action": np.repeat(action[None], cfg.model.action_horizon, axis=0),
            "prompt": prompt,
        }
        for key in video_keys:
            raw[key] = np.zeros((h, w, c), dtype=np.uint8)

        try:
            out = xform(raw)
        except Exception as exc:  # noqa: BLE001
            problems.append(f"[{label}] ep{ep}: transform failed {type(exc).__name__}: {exc}")
            continue
        problems.extend(f"[{label}] ep{ep}: {p}" for p in _check_transformed(out, cfg, prompt))

    detail = (
        f"[{label}] {len(shards)} episodes | state{state_dim} action{action_dim} | "
        f"tested ep{tested} | tasks={len(tasks)}"
    )
    return (not problems), detail if not problems else "; ".join(problems)


def structure_check(cfg) -> tuple[bool, str]:
    import openpi.transforms as _transforms  # noqa: PLC0415

    data_config = cfg.data.create(cfg.assets_dirs, cfg.model)
    components = getattr(data_config, "components", None)

    if components:
        # Mixed config: validate every component with its own transforms.
        results = []
        for component in components:
            root = pathlib.Path(component.repo_id)
            xform = _transforms.compose([*component.repack_transforms.inputs, *component.data_transforms.inputs])
            results.append(_structure_check_one(cfg, root.name, root, xform))
    else:
        root = pathlib.Path(data_config.repo_id)
        xform = _transforms.compose([*data_config.repack_transforms.inputs, *data_config.data_transforms.inputs])
        results.append(_structure_check_one(cfg, root.name, root, xform))

    ok = all(r[0] for r in results)
    detail = " || ".join(r[1] for r in results)
    return ok, detail


def full_check(cfg) -> tuple[bool, str]:
    import openpi.transforms as _transforms  # noqa: PLC0415
    import openpi.training.data_loader as _data_loader  # noqa: PLC0415

    data_config = cfg.data.create(cfg.assets_dirs, cfg.model)
    dataset = _data_loader.create_torch_dataset(data_config, cfg.model.action_horizon, cfg.model)
    mixed = bool(getattr(data_config, "components", None))
    xform = None if mixed else _transforms.compose(
        [*data_config.repack_transforms.inputs, *data_config.data_transforms.inputs]
    )

    n = len(dataset)
    problems = []
    for idx in (0, n // 2, n - 1):
        try:
            item = dataset[idx]
            # Mixed components are already transformed by their own pipeline.
            out = item if mixed else xform(item)
        except Exception as exc:  # noqa: BLE001
            problems.append(f"idx{idx}: {type(exc).__name__}: {exc}")
            continue
        expected = str(out["prompt"])
        base_img = out["image"]["base_0_rgb"]
        if base_img.max() == 0:
            problems.append(f"idx{idx}: base image is all zeros")
        problems.extend(f"idx{idx}: {p}" for p in _check_transformed(out, cfg, expected))
    detail = f"len={n} | tested idx (0, {n // 2}, {n - 1})"
    return (not problems), detail if not problems else "; ".join(problems)


def main(mode: str = "structure", config: str | None = None) -> None:
    register_configs(CONFIGS)
    if mode == "full":
        enable_fast_local_loading()
    selected = [c for c in CONFIGS if config is None or c.name == config]
    if not selected:
        raise SystemExit(f"unknown config: {config}")

    print(f"mode = {mode}", flush=True)
    failures = 0
    for cfg in selected:
        ok, detail = structure_check(cfg) if mode == "structure" else full_check(cfg)
        print(f"[{'PASS' if ok else 'FAIL'}] {cfg.name}\n        {detail}", flush=True)
        failures += 0 if ok else 1
    print("RESULT:", "ALL PASS" if failures == 0 else f"{failures} FAILED", flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    tyro.cli(main)

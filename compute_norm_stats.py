#!/usr/bin/env python
"""Compute normalization statistics for the RoboCasa365 target configs.

Thin wrapper around the omnisf ``scripts/compute_norm_stats.py`` that first
registers the RoboCasa365 configs.  The stats are written to
``$ROBOCASA365_ASSETS_DIR/<config_name>/norm_stats.json`` and are also copied
into every checkpoint by the trainer.

Usage::

    $OPENPI_PYTHON compute_norm_stats.py --config-name robocasa365_target_atomic

Or via ``scripts/compute_norm_stats.sh`` for all three splits.
"""

from __future__ import annotations

import os
import sys

# Keep the external openpi-omnisf checkout strictly read-only (no .pyc writes).
sys.dont_write_bytecode = True

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tyro  # noqa: E402

from robocasa365.configs import CONFIGS  # noqa: E402
from robocasa365.registry import load_compute_norm_stats_main  # noqa: E402
from robocasa365.registry import register_configs  # noqa: E402


def main(
    config_name: str,
    max_frames: int | None = None,
    seed: int = 0,
    batch_size: int | None = None,
    num_workers: int | None = None,
) -> None:
    register_configs(CONFIGS)
    omnisf_main = load_compute_norm_stats_main()
    omnisf_main(
        config_name=config_name,
        max_frames=max_frames,
        seed=seed,
        batch_size=batch_size,
        num_workers=num_workers,
    )


if __name__ == "__main__":
    tyro.cli(main)

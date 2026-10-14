#!/usr/bin/env python
"""RoboCasa365 post-training entry point (openpi-omnisf).

Registers the RoboCasa365 target configs into the installed openpi registry and
delegates to the omnisf alignment trainer (``scripts/train_align.py``), so this
is a drop-in replacement for ``scripts/train_align.py`` that also knows about
the RoboCasa365 configs.

Examples
--------
List the RoboCasa365 configs::

    $OPENPI_PYTHON train.py --help

Single-node, 8-GPU training on the composite-seen split::

    $OPENPI_PYTHON -m torch.distributed.run --nnodes=1 --nproc-per-node=8 \\
        --node-rank=0 --master-addr=127.0.0.1 --master-port=29500 \\
        train.py robocasa365_target_composite_seen --exp-name=run0 --overwrite

Better: use ``scripts/train_8gpu.sh``, which sets up the environment first.
"""

from __future__ import annotations

import os
import sys

# Keep the external openpi-omnisf checkout strictly read-only: never let the
# interpreter drop `__pycache__/*.pyc` files into it.
sys.dont_write_bytecode = True

# Make `import robocasa365` work when this file is executed from anywhere.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from robocasa365.configs import CONFIGS  # noqa: E402
from robocasa365.registry import enable_fast_local_loading  # noqa: E402
from robocasa365.registry import load_train_align_main  # noqa: E402
from robocasa365.registry import register_configs  # noqa: E402


def main() -> None:
    register_configs(CONFIGS)

    # Avoid the stock loader's per-episode file validation / full timestamp scan
    # for these 8k-episode local datasets (opt out with ROBOCASA365_FAST_LOCAL_LOADING=0).
    enable_fast_local_loading()

    # `cli()` exposes every config known to openpi (omnisf built-ins + external
    # configs + the RoboCasa365 ones registered above).
    from openpi.training import config as _config  # noqa: PLC0415

    config = _config.cli()

    train_align_main = load_train_align_main()
    train_align_main(config)


if __name__ == "__main__":
    main()

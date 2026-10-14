"""RoboCasa365 post-training entry point for the openpi-omnisf framework.

This package provides everything needed to post-train the omnisf pi0.5 model on
the RoboCasa365 *target* datasets (atomic / composite-seen / composite-unseen):

- :mod:`robocasa365.policy`      -- 12-D RoboCasa action policy (state/action padded to 32).
- :mod:`robocasa365.data_config` -- LeRobot data config factory for the v2.1 target datasets.
- :mod:`robocasa365.configs`     -- the concrete :class:`openpi.training.config.TrainConfig`s.
- :mod:`robocasa365.registry`    -- helpers to register configs into the installed openpi.

The training framework itself (``openpi`` + ``scripts/train_align.py``) is reused
from the omnisf checkout, which is located through ``OPENPI_ROOT`` (default
``/root/nas/lfh/omnisf``).  Nothing inside that checkout is modified.
"""

from robocasa365.configs import CONFIGS
from robocasa365.configs import DATASET_DIRS
from robocasa365.data_config import LeRobotRoboCasa365DataConfig

__all__ = ["CONFIGS", "DATASET_DIRS", "LeRobotRoboCasa365DataConfig"]

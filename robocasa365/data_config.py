"""LeRobot data config factory for the RoboCasa365 target datasets.

The datasets are the flat LeRobot **v3.0** mirrors from HuggingFace
(``ember-lab-berkeley/robocasa365-target-*``) after being converted back to
LeRobot **v2.1** (see ``scripts/convert_v30_to_v21.sh``), which is what the
installed openpi-omnisf ``lerobot`` expects.

Dataset schema (shared by all RoboCasa365 repos)::

    observation.state                                      float[16]
    action                                                 float[12]
    observation.images.robot0_agentview_left  (video 256x256)
    observation.images.robot0_agentview_right (video 256x256)
    observation.images.robot0_eye_in_hand     (video 256x256)
    task_index                                             int64

The repack transform below renames those keys to the layout expected by
:class:`robocasa365.policy.RoboCasa365Inputs`.
"""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import pathlib

from typing_extensions import override
import tyro

import openpi.models.model as _model
from openpi.training.config import DataConfig
from openpi.training.config import DataConfigFactory
from openpi.training.config import ModelTransformFactory
import openpi.transforms as _transforms

from robocasa365.policy import RoboCasa365Inputs
from robocasa365.policy import RoboCasa365Outputs


@dataclasses.dataclass(frozen=True)
class LeRobotRoboCasa365DataConfig(DataConfigFactory):
    """Data config for a single RoboCasa365 target split (12-D action space)."""

    # Number of native action dimensions kept by RoboCasa365 (base_motion(4) +
    # control_mode(1) + eef_delta_pos(3) + eef_delta_aa(3) + gripper(1)).
    native_action_dim: int = 12

    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default_factory=lambda: _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/state": "observation.state",
                        "observation/image": "observation.images.robot0_agentview_left",
                        "observation/right_image": "observation.images.robot0_agentview_right",
                        "observation/wrist_image": "observation.images.robot0_eye_in_hand",
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        )
    )

    action_sequence_keys: tyro.conf.Suppress[Sequence[str]] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[
                RoboCasa365Inputs(
                    action_dim=model_config.action_dim,
                    model_type=model_config.model_type,
                )
            ],
            outputs=[RoboCasa365Outputs(action_dim=self.native_action_dim)],
        )

        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


__all__ = ["LeRobotRoboCasa365DataConfig"]

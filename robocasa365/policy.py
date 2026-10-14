"""RoboCasa365 policy transforms for the openpi-omnisf framework.

This follows the official RoboCasa365 openpi recipe
(``robocasa-benchmark/openpi`` -> ``src/openpi/policies/robocasa_policy.py``):

* state is the raw 16-D ``observation.state``
  (``base_pos(3) + base_quat(4) + eef_pos_rel(3) + eef_quat_rel(4) + gripper_qpos(2)``),
  zero-padded to the model action dimension;
* actions are the raw 12-D ``action``
  (``base_motion(4) + control_mode(1) + eef_delta_pos(3) + eef_delta_aa(3) + gripper(1)``),
  zero-padded to the model action dimension;
* three cameras are consumed for pi0.5: left agent view -> ``base_0_rgb``,
  eye-in-hand -> ``left_wrist_0_rgb``, right agent view -> ``right_wrist_0_rgb``.

No gripper/rotation re-scaling is applied: the RoboCasa365 datasets are already
stored in the controller's delta command space, matching the official pipeline.
"""

from __future__ import annotations

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_robocasa365_example() -> dict:
    """Creates a random input example for the RoboCasa365 policy (for serving/tests)."""
    return {
        "observation/state": np.random.rand(16).astype(np.float32),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/right_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class RoboCasa365Inputs(transforms.DataTransformFn):
    """Convert RoboCasa365 dataset/environment observations to the model input format."""

    # The action dimension of the model. Used to pad state and actions for pi0/pi0.5.
    action_dim: int

    # Which model family is being trained (PI0 / PI05 / PI0_FAST).
    model_type: _model.ModelType = _model.ModelType.PI05

    def __call__(self, data: dict) -> dict:
        mask_padding = self.model_type == _model.ModelType.PI0

        state = transforms.pad_to_dim(data["observation/state"], self.action_dim)

        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])

        # pi0.5 consumes all three cameras; pi0 masks the right view.
        if self.model_type == _model.ModelType.PI05:
            if "observation/right_image" not in data:
                raise ValueError("observation/right_image is required for pi0.5 but not found in data")
            right_image = _parse_image(data["observation/right_image"])
            right_mask = np.True_
        else:
            right_image = np.zeros_like(base_image)
            right_mask = np.False_ if mask_padding else np.True_

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": right_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": right_mask,
            },
        }

        # Actions are only available during training.
        if "actions" in data:
            inputs["actions"] = transforms.pad_to_dim(data["actions"], self.action_dim)

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class RoboCasa365Outputs(transforms.DataTransformFn):
    """Trim model outputs (padded to ``action_dim``) back to the native 12-D RoboCasa action."""

    action_dim: int = 12

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"])[..., : self.action_dim]}
